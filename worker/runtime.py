"""Per-worker request handling shared by every strategy endpoint.

The first request a worker process sees acquires + loads the model (that is the
cold path being measured); later requests on the same process only run
inference (the warm path). Every response carries enough timing and provenance
to tell the two apart and to audit the run afterwards.
"""
from __future__ import annotations

import os
import time
import traceback

import strategies

PROCESS_STARTED = time.time()   # module import ~= worker process boot
_STATE: dict = {"calls": 0, "model": None, "tokenizer": None, "load": None}


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ("torch", "transformers", "huggingface_hub", "hf_xet", "runpod", "accelerate", "safetensors"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def _acquire(strategy: str, model_id: str, token: str | None, evict: bool) -> strategies.Acquired:
    if strategy == "nocache":
        return strategies.acquire_nocache(model_id, token)
    if strategy == "volume":
        return strategies.acquire_volume(model_id, token)
    if strategy == "volumecache":
        namespace = os.environ["BENCH_NAMESPACE"]
        evicted = strategies.evict_page_cache(strategies.volumecache_mirror_files(namespace)) if evict else 0
        acq = strategies.acquire_volumecache(model_id, token, namespace=namespace)
        acq.notes["evicted_mirror_bytes"] = evicted
        return acq
    if strategy == "gsvolumecache":
        namespace = os.environ["BENCH_NAMESPACE"]
        evicted = strategies.evict_page_cache(strategies.volumecache_mirror_files(namespace)) if evict else 0
        cls, impl = strategies.global_store_volumecache_class()
        acq = strategies.acquire_volumecache(model_id, token, namespace=namespace, cache_cls=cls)
        acq.notes.update(evicted_mirror_bytes=evicted, impl=impl)
        return acq
    if strategy == "modelstore":
        return strategies.acquire_modelstore(model_id, token)
    if strategy == "globalstore":
        return strategies.acquire_globalstore(model_id, token)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {strategies.STRATEGIES}")


def _load(path: str) -> tuple[object, object, dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(path)
    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    torch.cuda.synchronize()
    t2 = time.monotonic()
    return model, tok, {"tokenizer_load_s": t1 - t0, "model_load_s": t2 - t1}


def _generate(model, tok, prompt: str, max_new_tokens: int) -> dict:
    import torch

    inputs = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                     tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    torch.cuda.synchronize()
    t0 = time.monotonic()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    gen_s = time.monotonic() - t0
    new = out[0][prompt_tokens:]
    return {"generate_s": gen_s, "prompt_tokens": prompt_tokens, "new_tokens": int(new.shape[-1]),
            "tokens_per_s": new.shape[-1] / gen_s if gen_s > 0 else None,
            "text_preview": tok.decode(new, skip_special_tokens=True)[:160]}


def _import_ml_stack() -> float:
    """Time torch/transformers import on its own: it is identical across
    strategies and otherwise hides inside whichever step imports first."""
    t0 = time.monotonic()
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM  # noqa: F401  (lazy module; force the real import)
    return time.monotonic() - t0


def _cold_load(strategy: str, model_id: str, token: str | None, evict: bool) -> dict:
    import_s = _import_ml_stack()
    t0 = time.monotonic()
    acq = _acquire(strategy, model_id, token, evict)
    acquire_s = time.monotonic() - t0
    evicted = strategies.evict_page_cache(acq.shared_storage_files) if evict else 0
    model, tok, load_t = _load(acq.path)
    _STATE.update(model=model, tokenizer=tok)
    return {"import_s": import_s, "acquire_s": acquire_s, "acquire": acq.timings, "bytes": acq.bytes_total,
            "hub_online": acq.hub_online, "notes": acq.notes, "evicted_bytes": evicted, **load_t}


def handle(strategy: str, payload: dict) -> dict:
    """One benchmark request. Errors are returned as values so the driver can
    record them next to the successful trials instead of losing the row."""
    received = time.time()
    _STATE["calls"] += 1
    model_id = os.environ["BENCH_MODEL"]
    record = {
        "ok": True, "strategy": strategy, "model": model_id,
        "worker_id": os.environ.get("RUNPOD_POD_ID"), "dc": os.environ.get("RUNPOD_DC_ID"),
        "process_call": _STATE["calls"], "process_age_s": received - PROCESS_STARTED,
        "cold_process": _STATE["model"] is None,
    }
    try:
        t0 = time.monotonic()
        if _STATE["model"] is None:
            _STATE["load"] = _cold_load(strategy, model_id, os.environ.get("HF_TOKEN") or None,
                                        evict=os.environ.get("BENCH_EVICT", "1") == "1")
            record["cold"] = _STATE["load"]
        gen = _generate(_STATE["model"], _STATE["tokenizer"], payload.get("prompt", DEFAULT_PROMPT),
                        int(payload.get("max_new_tokens", 128)))
        record["generate"] = gen
        record["handler_s"] = time.monotonic() - t0
        if record["cold_process"]:
            import torch
            record["gpu"] = torch.cuda.get_device_name(0)
            record["versions"] = _versions()
    except Exception as e:
        record.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
    return record


DEFAULT_PROMPT = "Explain in three sentences why caching model weights reduces serverless cold-start latency."
