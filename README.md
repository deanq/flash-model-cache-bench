# flash-model-cache-bench

**What is the fastest way to get a Hugging Face model onto a Runpod Serverless GPU on a cold start?**

This repo benchmarks four model-storage strategies side by side on real Runpod Serverless GPU
workers deployed with [Runpod Flash](https://github.com/runpod/flash). It measures the whole AI
task: getting the weights, loading them to the GPU, and generating text. Anyone with a Runpod
account can reproduce the numbers below with two commands:

```bash
make setup
make run
```

`make run` deploys the endpoints, runs the benchmark, writes a report, and **always deletes
everything it created** — including when the run fails or you press Ctrl-C.

---

## Findings

Measured 2026-09-25 on NVIDIA RTX 4090 workers in `EU-RO-1`, with Flash 1.20.0, transformers 5.17.0
and torch 2.9.1. All values are **medians over forced cold starts**; ranges and raw data are in
[`results/`](results/).

### Time the storage strategy controls (get the bytes + load them to GPU)

| strategy | Qwen2.5-3B-Instruct (5.8 GiB) | Qwen2.5-7B-Instruct (14.2 GiB) |
|---|---|---|
| `nocache`: download from HF + load | 18.9s | 28.3s |
| `volume`: load directly from network volume | 17.5s | 37.6s |
| `volumecache`: hydrate to local disk + load | **7.2s** (5.5 + 1.3) | **15.5s** (10.2 + 4.6) |
| `modelstore`: load from Runpod cached model | 2.7s | 14.1s |

### Whole cold request, as a client sees it

| strategy | 3B | 7B | of which: waiting for a worker (`delayTime`, 3B / 7B) |
|---|---|---|---|
| `volumecache` | **27.9s** | **49.6s** | 12.6s / 21.3s |
| `nocache` | 45.7s | 58.2s | 17.3s / 16.3s |
| `volume` | 44.9s | 78.9s | 16.4s / 31.0s |
| `modelstore` | 77.7s | 58.1s | **69.9s / 34.3s** |

### What this means

1. **Loading directly from the network volume is ~2.4× slower than copying to local disk first
   (VolumeCache), at both model sizes.** Reading safetensors straight off the mount ran at
   ~340–390 MiB/s. VolumeCache's hydrate copied the same files at ~1.0–1.4 GiB/s, and the load that
   followed read from local disk. For the 7B model, loading from the volume was even **slower than
   re-downloading from Hugging Face** on every cold start.
2. **VolumeCache had the fastest cold request overall** in both runs.
3. **Model Store loads fastest once the worker is running, but waiting for a worker dominates.** Its
   platform delay was the largest and least predictable of the four (3B: 36–97s; 7B: 9.5–115s,
   vs a median of ~16s for the other strategies). That is consistent with Runpod's documented
   behaviour of holding a worker until a host that already has the model is available. That
   wait is not billed as worker time, but the client still waits for it.
4. **VolumeCache still contacts the Hub on every cold start** (~0.5s, metadata only). It skips
   symlinks and `refs/`, so the Hugging Face cache it restores has the weight files but no
   snapshot folder, and `huggingface_hub` re-links them through the Hub. If the Hub is unreachable,
   that becomes a full download.
5. **Once the model is in VRAM, storage doesn't matter.** Warm throughput was the same for every
   strategy (≈44 tokens/s for 7B on a 4090). Differences in the warm table come from the host.

Caveats: 5 cold trials (3B) and 3 cold trials (7B) per strategy. Worker placement is noisy: one
`volume` trial waited 118s for a worker. Round by round, VolumeCache's storage time beat `volume`'s
in 7 of 8 rounds; the exception was a 3B round where one hydrate took 20.4s instead of ~5s.
Exact seconds will vary with datacenter, GPU host, and time of day.

Full reports: [`results/qwen3b/report.md`](results/qwen3b/report.md) ·
[`results/qwen7b/report.md`](results/qwen7b/report.md)

---

## Reproduce it yourself

### Prerequisites

- A [Runpod](https://www.runpod.io) account with credit and an API key.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (installs Python and every dependency).
- `make`, `git`. Tested on macOS; any Linux works too. Nothing runs on your GPU — the benchmark
  runs on Runpod.
- Optional: a Hugging Face token, only for gated/private models.

### Steps

```bash
git clone https://github.com/deanq/flash-model-cache-bench.git
cd flash-model-cache-bench

make setup            # uv sync (pins runpod-flash 1.20.0), creates .env, checks for an API key
$EDITOR .env          # set RUNPOD_API_KEY=...   (or run: uv run flash login)

make run              # ~15-20 min; report written to runs/<run-id>/report.md
```

To reproduce the two published runs exactly:

```bash
make run COLD_TRIALS=5 RUN_ID=qwen3b                                          # Qwen2.5-3B (default model)
make run MODEL=Qwen/Qwen2.5-7B-Instruct COLD_TRIALS=3 VOLUME_GB=60 RUN_ID=qwen7b
```

Other targets:

| command | what it does |
|---|---|
| `make report [RUN=<id>]` | re-render a report from `runs/<id>/results.jsonl` |
| `make clean` | tear down anything a crashed or interrupted run left behind (safe to run any time) |
| `make test` | unit tests — no network, no spend |

### Cost

Every endpoint is capped at one worker (`workers=(0, 1)`), so at most four GPU workers exist at a
time, and only during a trial. The published runs used about **12 GPU-minutes of handler time** each
(16.5 and 12.5 minutes of wall time), plus a network volume that exists only for the run. At
Serverless RTX 4090 prices that is a few dollars at most.

### Knobs

Set them in `.env` or on the command line (`make run MODEL=... COLD_TRIALS=5`).

| variable | default | notes |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-3B-Instruct` | any HF causal LM that fits the GPU in bf16 |
| `GPU` | `ADA_24` | Flash `GpuGroup`/`GpuType` name(s), comma-separated (e.g. `AMPERE_48`) |
| `DATACENTER` | `EU-RO-1` | must support network volumes |
| `VOLUME_GB` | `50` | needs ≈ 2.2 × model size (volume copy + VolumeCache mirror) |
| `STRATEGIES` | all four | comma-separated subset |
| `COLD_TRIALS` | `3` | forced-cold requests per strategy |
| `WARM_PER_COLD` | `2` | warm requests after each cold one |
| `MAX_NEW_TOKENS` | `128` | exact number of tokens generated per request |
| `PARALLEL` | `1` | strategies run concurrently within a round (in waves, see below); `0` = one at a time |
| `EVICT_PAGE_CACHE` | `1` | drop the kernel page cache for files read from shared storage before reading them |
| `HF_TOKEN` | – | gated/private models only |
| `KEEP` | `0` | `1` skips teardown for debugging — run `make clean` afterwards |

### Reading the report

`runs/<id>/report.md` contains:

- **Answer**: volume vs VolumeCache in one sentence, plus the ranking by cold request.
- **Cold start**: per strategy, the whole-request time; Runpod's `delayTime` (queue + worker
  start) and `executionTime`; Python import time; acquire (split into hydrate and Hub
  time); tokenizer and model load; **storage total**; and the first generation. Cells show
  `median [min–max]`.
- **Warm requests**: latency and tokens/s with the model already in GPU memory.
- **First-ever request**: the one-time cost of filling each cache.
- **Observations**: flags generated automatically, such as Hub use on cold starts and unusually long
  worker waits.

Raw data is in `results.jsonl`, one line per request, including everything the handler returned
(worker id, GPU, package versions, byte counts).

---

## How it works

### The four strategies

All four are queue-based GPU endpoints running **the same handler code** on the same GPU class in
the same datacenter. Only where the model bytes come from differs.

| strategy | on a cold start |
|---|---|
| `nocache` | `snapshot_download` into a fresh directory on container disk |
| `volume` | HF cache lives on the network volume (`/runpod-volume/hfbench/hub`); weights are loaded straight from the mount |
| `volumecache` | HF cache on container disk; [`VolumeCache`](https://github.com/runpod/runpod-python) `hydrate()`s it from a mirror on the network volume, then it loads locally |
| `modelstore` | Runpod [cached model](https://docs.runpod.io/serverless/endpoints/model-caching), staged by the platform at `/runpod-volume/huggingface-cache/hub` |

Each request then loads the tokenizer and model (`AutoModelForCausalLM`, bf16, `device_map="cuda"`)
and greedily generates exactly `MAX_NEW_TOKENS` tokens.

### A run, step by step

1. **Preflight**: API key present; no leftover `hf-cache-bench` app from an earlier run.
2. **Deploy**: `flash deploy --app hf-cache-bench --env run-<id>` creates the four endpoints and a
   network volume `hfbench-<id>`, attached to `volume` and `volumecache`.
3. **Attach cached model**: sets `modelReferences` on the `modelstore` endpoint.
4. **Prime**: one request per strategy fills its cache (download to the volume, VolumeCache
   mirror, Model Store staging). Reported separately.
5. **Rounds** (`COLD_TRIALS` of them):
   - Every endpoint is paused (`workersMax=0`) until `/health` reports zero workers, then restored,
     so the next request is guaranteed to hit a fresh worker.
   - Each strategy then gets 1 cold request and `WARM_PER_COLD` warm ones.
   - Within a round, `volume` and `volumecache` run in different waves, so they never compete for
     the network volume.
6. **Report**: `runs/<id>/report.md` and `summary.json`.
7. **Teardown** (always):
   - `flash app delete` removes the endpoints.
   - The network volume is deleted separately, because Flash leaves it behind.
   - A check confirms that no `hfbench-*` endpoint and no run volume remains. If anything is left,
     the run exits non-zero and says what; `make clean` retries the teardown.

### Measurement choices, and why

- **Cold starts are forced, not waited for.** Workers stayed alive for minutes past a 5s
  `idleTimeout`, and `/health` counts stopped FlashBoot workers as idle, so waiting was unreliable.
  Every trial is also checked in the handler with a `cold_process` flag. A "cold" trial served by
  an already-loaded process is marked invalid and retried.
- **The benchmark's own pause window is excluded.** After `workersMax` is restored, the endpoint
  answers `409 ENDPOINT_PAUSED` for a few seconds. The clock starts when the job is accepted.
- **Local caches live outside `/root/.cache`.** flash-worker tars new files under `/root/.cache`
  onto an attached network volume, which would silently turn every strategy into a volume-backed one.
- **Leftover local files are wiped on each cold process.** FlashBoot can restart the same worker
  with its container disk intact. `nocache` and `volumecache` delete leftovers first (and record
  how many bytes were there), so each cold start measures the strategy itself.
- **Every strategy tries its cache offline before calling the Hub.** The report counts the cold
  starts that still needed the Hub.
- **The page cache is evicted** (`posix_fadvise(DONTNEED)`) for files read from the volume or Model
  Store, so the benchmark reads from storage, not RAM. This does not reach a FUSE client's own cache
  on the host.
- **Medians instead of means.** Worker placement produces outliers (a 118s wait next to 10s ones)
  that pull a mean far from the typical case.
- **VolumeCache is vendored** from runpod-python 1.12.0
  (`worker/vendor/rp_volume_cache.py`, MIT). The Flash GPU image ships runpod 1.10.1, which is
  already imported before the handler runs, so pinning a newer version has no effect. The SDK's
  own class is used automatically once the image includes it; `notes.impl` records which one ran.

### Flash gaps found while building this

- **No `Endpoint` parameter for Model Store cached models,** and none in the REST API either.
  `bench/runpod_api.py` sets `modelReferences` with the same GraphQL `saveEndpoint` mutation the
  console uses (the API stores it as `https://huggingface.co/<Org>/<Name>:<commit>`).
- **`execution_timeout_ms` passed to `Endpoint` was not applied;** the deployed endpoint kept `0`.
- **`flash build` reads `dependencies=[...]` from the source code itself,** so the list must be written
  out literally in each `Endpoint(...)` call.
- **`flash app delete` does not delete network volumes** its endpoints created.
- **Stale `.flash/resources.pkl` state breaks the next deploy** ("Endpoint not found") after the
  endpoints are deleted, so each run clears it first.

### Troubleshooting

- **`app 'hf-cache-bench' already exists`**: a previous run didn't finish its teardown. Run
  `make clean`.
- **A strategy fails in the prime phase**: it is excluded from the rounds, and the error is shown
  in the report's first-request table. Common causes are a model too large for `VOLUME_GB`, or a
  gated model without `HF_TOKEN`.
- **Very long `modelstore` delays**: expected. See finding 3.

---

## Layout

```
worker/                 Flash deploy root (becomes the worker's /app)
  bench_app.py          the four @Endpoint definitions
  runtime.py            request handling: cold load vs warm, timings, provenance
  strategies.py         the four acquisition strategies (unit-tested)
  vendor/               VolumeCache from runpod-python 1.12.0 (MIT)
bench/                  orchestrator (runs on your machine)
  __main__.py           run / report / clean / preflight
  config.py             env-var config, validated before anything is deployed
  deploy.py             flash CLI deploy/teardown + state file for recovery
  driver.py             prime, forced-cold rounds, warm follow-ups
  runpod_api.py         job/health API; GraphQL for modelReferences and workersMax; volumes
  report.py             results.jsonl -> report.md
results/                the published runs (report, raw trials, summary, config)
tests/                  `make test` — no network, no spend
```

## License

MIT (see [`LICENSE`](LICENSE)). `worker/vendor/rp_volume_cache.py` is © Runpod, MIT, see
[`worker/vendor/LICENSE.runpod-python`](worker/vendor/LICENSE.runpod-python).
