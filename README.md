# flash-model-cache-bench

Benchmark how a Hugging Face model's **storage strategy** affects cold-start latency
of an AI task (LLM text generation) on Runpod Serverless GPUs, deployed with
[Runpod Flash](https://github.com/runpod/flash).

```bash
git clone <this repo> && cd flash-model-cache-bench
make setup     # uv sync, create .env, check for a Runpod API key
make run       # deploy 4 endpoints -> benchmark -> report -> tear everything down
```

The report lands in `runs/<run-id>/report.md` (raw trials in `results.jsonl`).

## What is compared

Four queue-based GPU endpoints run **identical handler code** on the same GPU class in
the same datacenter. Only where the model bytes come from differs:

| strategy | where the model comes from on a cold start |
|---|---|
| `nocache` | downloaded from the Hugging Face Hub into container disk, every time |
| `volume` | HF cache lives **on the network volume**; weights are read straight off the mount |
| `volumecache` | HF cache on container disk; [`VolumeCache`](https://github.com/runpod/runpod-python) hydrates it from a mirror on the network volume first |
| `modelstore` | Runpod [cached model](https://docs.runpod.io/serverless/endpoints/model-caching) (`modelReferences`), staged by the platform at `/runpod-volume/huggingface-cache/hub` |

Each request does the full AI task: acquire model → load tokenizer + weights to GPU (bf16)
→ greedy-generate `MAX_NEW_TOKENS` tokens. The handler reports a breakdown
(import, acquire → hydrate/Hub/download, tokenizer/model load, generate, tokens/s);
the driver adds the client-observed end-to-end time and Runpod's `delayTime`
(queue + worker start) / `executionTime`.

## How a run works

1. **preflight** – API key present, no leftover `hf-cache-bench` app from an earlier run.
2. **deploy** – `flash deploy --app hf-cache-bench --env run-<id>` builds the four endpoints
   (`worker/bench_app.py`) and a network volume `hfbench-<id>` attached to `volume` and `volumecache`.
3. **attach cached model** – sets `modelReferences` on the `modelstore` endpoint (see *Flash gaps*).
4. **prime** – one request per strategy populates its cache (download to volume, VolumeCache
   mirror, Model Store staging). Reported separately as the one-time cost.
5. **rounds** × `COLD_TRIALS` – every endpoint is paused (`workersMax=0`) until `/health`
   shows zero workers, then restored, so the next request is a guaranteed cold start. Then per
   strategy: 1 cold request + `WARM_PER_COLD` warm requests. The handler's `cold_process` flag
   verifies each trial; a "cold" trial served by an already-loaded process is marked invalid and retried.
6. **report** – `runs/<id>/report.md`, `summary.json`.
7. **teardown** – always (success, failure, Ctrl-C, SIGTERM): `flash app delete`, delete the
   network volume, then verify no `hfbench-*` endpoint or the run's volume remains. A failed
   teardown exits non-zero and says what is left; `make clean` retries it.

## Knobs

Set in `.env` or on the command line (`make run MODEL=... COLD_TRIALS=5`). See `.env.example`.

| var | default | |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-3B-Instruct` | any HF causal LM that fits the GPU in bf16 |
| `GPU` | `ADA_24` | Flash `GpuGroup`/`GpuType` name(s), comma-separated |
| `DATACENTER` | `EU-RO-1` | must support network volumes |
| `VOLUME_GB` | `50` | needs ≈ 2.2 × model size (volume copy + VolumeCache mirror) |
| `STRATEGIES` | all four | subset, comma-separated |
| `COLD_TRIALS` / `WARM_PER_COLD` | `3` / `2` | |
| `MAX_NEW_TOKENS` | `128` | |
| `PARALLEL` | `1` | strategies run concurrently within a round; `0` = one at a time (no volume contention) |
| `EVICT_PAGE_CACHE` | `1` | `posix_fadvise(DONTNEED)` files read from shared storage before reading them |
| `HF_TOKEN` | – | only for gated/private models |
| `KEEP` | `0` | `1` skips teardown (debugging; run `make clean` after) |

## Cost

Every endpoint is `workers=(0, 1)`: at most four GPU workers exist at once, and only while a
trial runs. A default run (3B model, 3 cold × 2 warm) takes roughly 20–30 minutes of wall time
and a few GPU-minutes per strategy. The network volume is deleted at the end.

## Measurement choices (and why)

- **Forced cold, not idle-timeout cold.** Workers stayed alive for minutes past a 5 s
  `idleTimeout`, so waiting was unreliable. Pausing via `workersMax=0` is deterministic.
- **Local caches outside `/root/.cache`.** flash-worker tars new files under `/root/.cache`
  onto an attached network volume, which would quietly turn every strategy into a
  volume-backed one.
- **Leftover local state is wiped per cold process.** FlashBoot can restart the same worker
  with its container disk intact; `nocache` and `volumecache` delete leftovers first (and report
  how many bytes were there) so each cold start measures the strategy itself.
- **Offline first.** Every strategy resolves the model from its cache without the Hub and
  only falls back to the network if that fails; the report counts cold starts that
  still needed the Hub.
- **VolumeCache is vendored** (`worker/vendor/rp_volume_cache.py`, runpod-python 1.12.0,
  MIT). The Flash GPU image ships runpod 1.10.x, which the harness has imported before the
  handler runs, so a newer pin cannot take effect. The SDK version is used automatically
  once the image carries it (`notes.impl` records which).

## Flash gaps hit while building this

- No `Endpoint` parameter for Model Store cached models. `bench/runpod_api.py` sets
  `modelReferences` with the console's GraphQL `saveEndpoint` mutation.
- `execution_timeout_ms` passed to `Endpoint` did not reach the deployed endpoint (it
  stayed at `0`).
- `flash build` only picks up `dependencies=[...]` when it is a literal inside the
  `Endpoint(...)` call (AST scan), so each decorator repeats the list.
- `flash app delete` does not delete network volumes that the app's endpoints created.

## Layout

```
worker/                 Flash deploy root (becomes the worker's /app)
  bench_app.py          the four @Endpoint definitions
  runtime.py            request handling: cold load vs warm, timings, provenance
  strategies.py         the four acquisition strategies (pure, unit-tested)
  vendor/               VolumeCache from runpod-python 1.12.0
bench/                  orchestrator (runs on your machine)
  __main__.py           run / report / clean / preflight
  config.py             env-var config, validated up front
  deploy.py             flash CLI deploy/teardown + state file
  driver.py             prime, forced-cold rounds, warm follow-ups
  runpod_api.py         job/health API, GraphQL for modelReferences + workersMax, volumes
  report.py             results.jsonl -> report.md
tests/                  `make test` — no network, no spend
```
