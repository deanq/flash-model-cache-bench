# flash-model-cache-bench

**What is the fastest way to get a Hugging Face model onto a Runpod Serverless GPU on a cold start?**

This repo benchmarks eight model-storage strategies side by side on real Runpod Serverless GPU
workers deployed with Runpod Flash — the `runpod.apps` SDK and `rp flash` CLI built into
[runpod-python](https://github.com/runpod/runpod-python) 2.1+. It measures the whole AI
task: getting the weights, loading them to the GPU, and generating text. Anyone with a Runpod
account can reproduce the numbers below with two commands:

```bash
make setup
make run
```

`make run` deploys the endpoints, runs the benchmark, writes a report, and **always deletes
everything it created** (endpoints, network volume, Global Store) — including when the run fails
or you press Ctrl-C.

---

## Findings

Measured 2026-09-25 on NVIDIA RTX 4090 workers in `EU-RO-1`, with Flash 1.20.0, transformers 5.17.0
and torch 2.9.1. All values are **medians over forced cold starts**; ranges and raw data are in
[`results/`](results/).

### Time the storage strategy controls (get the bytes + load them to GPU)

| strategy | Qwen2.5-3B-Instruct (5.8 GiB) | Qwen2.5-7B-Instruct (14.2 GiB) |
|---|---|---|
| `nocache`: download from HF + load | 14.7s | 23.5s |
| `volume`: load directly from network volume | 16.3s | 34.8s |
| `volumecache`: hydrate from network volume + local load | **6.7s** (4.7 + 1.3) | **16.2s** (11.0 + 4.6) |
| `modelstore`: load from Runpod cached model | 2.3s | 14.2s |
| `globalstore`: load directly from Global Store | 48.1s | 98.9s |
| `gsvolumecache`: hydrate from Global Store + local load | 24.3s (21.1 + 1.5) | 81.4s ⚠️ see finding 4 |

### Whole cold request, as a client sees it

| strategy | 3B | 7B | of which: waiting for a worker (`delayTime`, 3B / 7B) |
|---|---|---|---|
| `volumecache` | **25.2s** | 43.4s | 13.8s / 16.3s |
| `nocache` | 35.2s | **41.0s** | 15.7s / 11.1s |
| `volume` | 44.6s | 62.1s | 18.0s / 17.8s |
| `gsvolumecache` | 46.5s | 108.4s | 15.8s / 18.2s |
| `globalstore` | 92.5s | 133.6s | 30.4s / 20.2s |
| `modelstore` | 106.0s | 81.2s | **98.6s / 69.3s** |

### What this means

1. **Copy to local disk first; don't load from shared storage.** On both backends, VolumeCache
   (hydrate to container disk, then load) beat loading directly from the mount. It won in 8 of 8
   rounds on the network volume and 6 of 7 on Global Store. Loading safetensors straight from the
   network volume ran at ~370–420 MiB/s; VolumeCache's hydrate ran at ~1.2–1.3 GiB/s.
2. **Global Store is the slowest place to load a model from.** Loading directly from it ran at
   ~125–150 MiB/s: about 3× slower than the network volume, and slower than downloading from
   Hugging Face. Hydrating from it (~280 MiB/s for 3B) helps, but was still ~4× slower than
   hydrating from a network volume. Global Store is object storage behind a FUSE client
   (geesefs), and loading safetensors from it through that client is a poor fit.
3. **Model Store loads fastest once the worker is running, but waiting for a worker dominates.**
   Its platform delay had a median of 99s (3B) and 69s (7B), vs ~11–30s for every other strategy,
   and one 3B trial never started within 20 minutes. That is consistent with Runpod's documented
   behaviour of holding a worker until a host that has the model is available. The wait isn't
   billed, but the client still waits for it. (A separate testbed also found that on **CPU**
   endpoints the declaration is accepted, but the model is never staged: Model Store is
   effectively GPU-only.)
4. **VolumeCache does not work on Global Store as shipped, and even patched it was unreliable for 7B.**
   - Stock VolumeCache copies big files with `shutil.copy2`, which fails on Global Store
     (EPERM on chmod/utime). The error is swallowed, so **every file ≥256 KiB is silently left
     out of the mirror** and every cold start re-downloads the weights. `gsvolumecache` uses a
     small `copyfile` patch (`worker/strategies.py`).
   - With the patch, the 7B mirror was complete (15.24 GB), yet each cold start hydrated only
     ~7.5 of 15.2 GB, silently. The worker re-downloaded the rest from HF and synced it back.
     For 3B (two shards) every hydrate was complete. The cause is not yet known; the patched
     copy now records its errors, so the next run will show them.
5. **HF cache folders don't survive on Global Store.** Symlinks written by one worker come back
   as 0-byte regular files on the next worker's mount. `snapshot_download` then reports the model
   as present, and loading fails. `globalstore` therefore stores the model as a plain folder of
   real files.
6. **VolumeCache still contacts the Hub on every cold start** (~0.5s, metadata only). It skips
   symlinks and `refs/`, so a hydrated HF cache has to be re-linked through the Hub. If the Hub is
   unreachable, that becomes a full download.
7. **Once the model is in VRAM, storage doesn't matter** — but the GPU host does. RTX 4090 workers
   came in two speed tiers (~36 vs ~90 tokens/s on 3B, consistent per worker). Whichever strategy
   drew slow hosts looks slower on warm requests, and its load times are inflated by ~25%. That
   is far smaller than the storage differences above.

Caveats: 5 cold trials (3B) and 3 (7B) per strategy. RTX 4090 capacity in EU-RO-1 was tight
during the 3B run, and some trials waited up to 18 minutes for a worker. Those waits are in
`delayTime`, not in storage time. Exact seconds will vary with datacenter, host, and time of day.

Full reports: [`results/qwen3b-gs/report.md`](results/qwen3b-gs/report.md) ·
[`results/qwen7b-gs/report.md`](results/qwen7b-gs/report.md). The earlier four-strategy runs from
the same day ([`results/qwen3b`](results/qwen3b/report.md),
[`results/qwen7b`](results/qwen7b/report.md)) agree with them.

### Re-run on runpod 2.1.0 (2026-11-03)

The tables above were measured with `runpod-flash` 1.20 and a vendored VolumeCache from runpod 1.12.
After moving the harness to the Flash built into runpod 2.1.0, a smaller 3B run (3 cold trials,
5 strategies, RTX 4090, `EU-RO-1`; [`results/qwen3b-rp21`](results/qwen3b-rp21/report.md)) gave
these median storage times (acquire + load):

| strategy | 2026-09-25 (Flash 1.20, VolumeCache 1.12) | 2026-11-03 (runpod 2.1.0) |
|---|---|---|
| `nocache` | 14.7s | 12.0s |
| `volume` | 16.3s | 21.2s |
| `volumecache` | **6.7s** (4.7 + 1.3) | 23.1s (17.8 hydrate + 4.2 load) |
| `globalstore` | 48.1s | 36.6s |
| `gshydrate` | – | **15.0s** (10.5 + 4.3) |

**VolumeCache no longer beats loading in place on a network volume, because of its new hydrate
defaults.** 2.1.0's VolumeCache hydrates with 64 concurrent 8 MiB range reads, tuned for Global
Store. On a network volume that ran at ~340–375 MiB/s. On the same worker and mirror, with the page
cache evicted before each try, `hydrate_workers=8, hydrate_chunk_size=64 MiB` ran at ~1000 MiB/s
(5.9s vs 15.8–17.3s, 2 repetitions each); one worker × 1 GiB ran at ~360 MiB/s and a plain sequential
`copyfile` at ~460 MiB/s. That is an SDK default to fix upstream; the harness measures stock VolumeCache
and doesn't tune it.

---

## Reproduce it yourself

### Prerequisites

- A [Runpod](https://www.runpod.io) account with credit and an API key.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (installs Python and every dependency).
- `make`, `git`. Tested on macOS; any Linux works too. Nothing runs on your GPU — the benchmark
  runs on Runpod.
- Global Store (global volumes) is in beta. If your account can't create one, leave the
  Global Store strategies out of `STRATEGIES` (see Troubleshooting).
- Optional: a Hugging Face token, only for gated/private models.

### Steps

```bash
git clone https://github.com/deanq/flash-model-cache-bench.git
cd flash-model-cache-bench

make setup            # uv sync (pins runpod 2.2.1), creates .env, checks for an API key
$EDITOR .env          # set RUNPOD_API_KEY=...   (or run: uv run rp login)

make run              # report written to runs/<run-id>/report.md
```

To reproduce the published six-strategy runs:

```bash
make run COLD_TRIALS=5 RUN_ID=qwen3b-gs                                          # Qwen2.5-3B (default model)
make run MODEL=Qwen/Qwen2.5-7B-Instruct COLD_TRIALS=3 VOLUME_GB=60 RUN_ID=qwen7b-gs
```

Other targets:

| command | what it does |
|---|---|
| `make report [RUN=<id>]` | re-render a report from `runs/<id>/results.jsonl` |
| `make billing [RUN=<id>]` | fetch Runpod-billed GPU-minutes and cost for a finished run (billing lags ~1 h); adds "Time and cost" to the report |
| `make clean` | tear down anything a crashed or interrupted run left behind (safe to run any time) |
| `make test` | unit tests — no network, no spend |

### Time and cost

Every endpoint is capped at one worker (`workers=(0, 1)`), so at most six GPU workers exist at a
time, and only during a trial. The published runs used **~20–25 GPU-minutes of handler time** each.
Wall time was 33 minutes for 7B and 92 minutes for 3B; the 3B run was stretched by GPU capacity
waits, which aren't billed as worker time. On top of that, a network volume and a Global Store
exist only for the run. At Serverless RTX 4090 prices this is a few dollars per run.

To go faster, run a subset: `make run STRATEGIES=volume,volumecache COLD_TRIALS=3`.

### Knobs

Set them in `.env` or on the command line (`make run MODEL=... COLD_TRIALS=5`).

| variable | default | notes |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-3B-Instruct` | any HF causal LM that fits the GPU in bf16 |
| `GPU` | `ADA_24` | `runpod.GpuGroup`/`GpuType` name(s) or shorthands, comma-separated (e.g. `AMPERE_48`, `4090`) |
| `DATACENTER` | `EU-RO-1` | must support network volumes |
| `VOLUME_GB` | `50` | needs ≈ 2.2 × model size (volume copy + VolumeCache mirror) |
| `STRATEGIES` | all eight | comma-separated subset of `nocache,volume,volumecache,modelstore,globalstore,gsvolumecache,gshydrate,volumeprefetch`; only these are deployed |
| `COLD_TRIALS` | `3` | forced-cold requests per strategy |
| `WARM_PER_COLD` | `2` | warm requests after each cold one |
| `MAX_NEW_TOKENS` | `128` | exact number of tokens generated per request |
| `PARALLEL` | `1` | strategies run concurrently within a round (in waves, see below); `0` = one at a time |
| `EVICT_PAGE_CACHE` | `1` | drop the kernel page cache for files read from shared storage before reading them |
| `HF_TOKEN` | – | gated/private models only |
| `KEEP` | `0` | `1` skips teardown for debugging — run `make clean` afterwards |
| `RUN_ID` | timestamp | names the run's app, network volume and global volume (`hfbench-<RUN_ID>`) |

### Reading the report

`runs/<id>/report.md` contains:

- **Answer**: for each storage backend, loading directly from it vs VolumeCache, plus the ranking
  by cold request.
- **Cold start**: per strategy, the whole-request time; Runpod's `delayTime` (queue + worker
  start) and `executionTime`; Python import time; acquire (split into hydrate and Hub time);
  tokenizer and model load; **storage total**; and the first generation. Cells show
  `median [min–max]`.
- **Warm requests**: latency and tokens/s with the model already in GPU memory.
- **First-ever request**: the one-time cost of filling each cache.
- **Observations**: flags generated automatically, such as Hub use on cold starts and unusually long
  worker waits.

Raw data is in `results.jsonl`, one line per request, including everything the handler returned:
worker id, GPU, package versions, byte counts, and VolumeCache's hydrated/fetched/mirrored bytes
and copy errors.

---

## How it works

### The seven strategies

All seven are queue-based GPU endpoints running **the same handler code** on the same GPU class in
the same datacenter. Only where the model bytes come from differs.

| strategy | on a cold start |
|---|---|
| `nocache` | `snapshot_download` into a fresh directory on container disk |
| `volume` | HF cache lives on the network volume (`/runpod-volume/hfbench/hub`); weights are loaded straight from the mount |
| `volumecache` | HF cache on container disk; [`VolumeCache`](https://github.com/runpod/runpod-python) `hydrate()`s it from a mirror on the network volume, then it loads locally |
| `modelstore` | Runpod [cached model](https://docs.runpod.io/serverless/endpoints/model-caching), staged by the platform at `/runpod-volume/huggingface-cache/hub` |
| `globalstore` | a [global volume](https://docs.runpod.io/storage/globalvolume/globalvolume-serverless) mounted at `/runpod-volume`; the model is loaded straight from a plain folder of files on it |
| `gsvolumecache` | like `volumecache`, but the mirror lives on the global volume |
| `gshydrate` | the `globalstore` folder, copied to container disk with 64 concurrent 8 MiB range reads, then loaded locally |

A global volume and a network volume can't be attached to the same endpoint, so the
Global Store strategies get their own endpoints. Global volumes don't tie an endpoint to a
datacenter, but here they are pinned to `DATACENTER` like the rest, so all of them are comparable.

Each request then loads the tokenizer and model (`AutoModelForCausalLM`, bf16, `device_map="cuda"`)
and greedily generates exactly `MAX_NEW_TOKENS` tokens.

### A run, step by step

1. **Preflight**: API key present; no app `hfbench-<id>` exists yet.
2. **Deploy**: `rp flash deploy --env run-<id>` in `worker/` creates app `hfbench-<id>` with one
   endpoint per selected strategy. Storage is declared in `worker/bench_app.py` and created by
   the deploy: a `NetworkVolume` `hfbench-<id>` for `volume`/`volumecache`, a `GlobalVolume`
   `hfbench-<id>` for the Global Store strategies, and `model=runpod.Model(MODEL)`
   (`modelReferences`) on `modelstore`. Every run is its own app, so teardown can't touch anything
   else in the account.
3. **Verify attachments**: reads each endpoint back and fails the run if the cached model or the
   global volume is missing.
4. **Prime**: one request per strategy fills its cache. Reported separately.
5. **Rounds** (`COLD_TRIALS` of them):
   - Every endpoint is paused (`workersMax=0`) until `/health` reports zero workers, then
     restored, so the next request is guaranteed to hit a fresh worker.
   - Each strategy then gets 1 cold request and `WARM_PER_COLD` warm ones.
   - Strategies that share storage (`volume`/`volumecache`, `globalstore`/`gsvolumecache`/`gshydrate`)
     run in different waves, so they never compete for bandwidth.
6. **Report**: `runs/<id>/report.md` and `summary.json`.
7. **Teardown** (always):
   - `rp flash app delete hfbench-<id> --yes` removes the endpoints and the app.
   - The network volume and the global volume are deleted separately, because `rp flash app delete`
     leaves them behind.
   - A check confirms that the app, the run's endpoint ids, its network volume and its global volume
     are gone. If anything is left, the run exits non-zero and says what; `make clean` retries the
     teardown.

### Measurement choices, and why

- **Cold starts are forced, not waited for.** Workers stayed alive for minutes past a 5s
  `idleTimeout`, and `/health` counts stopped FlashBoot workers as idle, so waiting was unreliable.
  Every trial is also checked in the handler with a `cold_process` flag. A "cold" trial served by
  an already-loaded process is marked invalid and retried.
- **The benchmark's own pause window is excluded.** After `workersMax` is restored, the endpoint
  answers `409 ENDPOINT_PAUSED` for a few seconds. The clock starts when the job is accepted.
- **`workersMax` is toggled with a partial update.** The harness PATCHes only the `workers` block
  (REST v2, as `rp endpoint scale` does), so a toggle can't detach `modelReferences`, the volume or
  the Flash environment binding.
- **Local caches live outside `/root/.cache`.** The Flash 1.x worker tarred new files under
  `/root/.cache` onto an attached volume, which would silently turn every strategy into a
  storage-backed one.
- **Leftover local files are wiped on each cold process.** FlashBoot can restart the same worker
  with its container disk intact. `nocache` and the VolumeCache strategies delete leftovers first
  (and record how many bytes were there).
- **Every strategy tries its cache offline before calling the Hub.** The report counts the cold
  starts that still needed the Hub.
- **The page cache is evicted** (`posix_fadvise(DONTNEED)`) for files read from shared storage,
  so the benchmark reads from storage, not RAM. This doesn't reach a FUSE client's own cache
  (MooseFS client, geesefs).
- **Medians instead of means.** Worker placement produces outliers (18-minute waits next to 10s
  ones) that pull a mean far from the typical case.
- **VolumeCache is the SDK's own.** `rp flash deploy` vendors the deploying client's runpod package
  into the build artifact, so workers run the same runpod (2.1.0) as your machine; `notes.impl`
  records the version. 2.1.0's VolumeCache hydrates big files with concurrent range reads (64 × 8 MiB
  by default) and treats file metadata as best-effort, so it no longer drops files on Global Store.

### Gaps found while building this

Status with runpod 2.1.0 (the harness was first built on standalone `runpod-flash` 1.20):

- **Model Store and Global Store are now first-class.** `model=runpod.Model(...)` sets
  `modelReferences`, and `mounts={"/runpod-volume": runpod.GlobalVolume(name)}` creates and attaches
  a global volume. The console GraphQL calls this harness needed with Flash 1.20 are gone.
- **`execution_timeout_ms` is applied** to the deployed endpoint (Flash 1.20 left it at `0`).
- **`dependencies=[...]` no longer has to be a literal.** The build reads it from the imported
  spec, not the source AST, so the decorator config is built by a function.
- **No local deploy state.** Deployed resources resolve server-side, so the Flash 1.20 problem of
  a stale `.flash/resources.pkl` breaking the next deploy is gone.
- **VolumeCache on Global Store no longer needs a patch.** The SDK now copies with `copyfile` and
  treats metadata as best-effort (unit-tested here), so the `copyfile` patch (finding 4) and the
  vendored copy are gone. `gsvolumecache` has not been re-measured live on 2.1.0 yet.
- **`rp flash app delete` still does not delete volumes** — neither the network volume nor the
  global volume its endpoints created.
- **The `409 ENDPOINT_PAUSED` window remains** after `workersMax` goes 0 → 1 (up to ~9s on
  2026-11-03); `bench/runpod_api.py` retries through it, and the clock starts when the job is accepted.
- **`runpod.__version__` on a worker reports the image's runpod (2.0.0), not the code that runs.**
  The artifact overlays the client's 2.1.0 package tree without its dist metadata, and `__version__`
  reads the metadata. The harness records `runpod._version` as `versions.runpod_code`.
- **VolumeCache's default hydrate (64 × 8 MiB ranges) is ~3× slower than 8 × 64 MiB on a network
  volume.** See "Re-run on runpod 2.1.0" above.
- **Flash endpoints still put your `RUNPOD_API_KEY` in the endpoint's environment,** where anyone
  who can read the endpoint config can see it. The harness never prints endpoint env.

### Troubleshooting

- **`app 'hfbench-<id>' already exists`**: that `RUN_ID` was used before. Pick another one, or run
  `make clean` if that run didn't finish its teardown.
- **Global Store creation fails** (e.g. beta not enabled): run without it,
  `make run STRATEGIES=nocache,volume,volumecache,modelstore`.
- **A strategy fails in the prime phase**: it is excluded from the rounds, and the error is shown
  in the report's first-request table. Common causes are a model too large for `VOLUME_GB`, or a
  gated model without `HF_TOKEN`.
- **Trials waiting many minutes for a worker**: that's GPU capacity in the datacenter; it shows
  up in `delayTime`. Try another `DATACENTER` or add a GPU class (`GPU=ADA_24,AMPERE_24`), which
  mixes hardware types.

---

## Layout

```
worker/                 `rp flash deploy` root (the build artifact's source)
  bench_app.py          runpod.App with one @app.queue endpoint per selected strategy, plus its storage
  runtime.py            request handling: cold load vs warm, timings, provenance
  strategies.py         the seven acquisition strategies (unit-tested)
bench/                  orchestrator (runs on your machine)
  __main__.py           run / report / clean / preflight
  config.py             env-var config, validated before anything is deployed
  deploy.py             rp flash deploy/teardown, attachment checks, volume cleanup, state file for recovery
  driver.py             prime, forced-cold rounds in shared-storage waves, warm follow-ups
  runpod_api.py         job/health API; workersMax (REST v2 PATCH); app/env lookup; volume cleanup
  report.py             results.jsonl -> report.md
results/                the published runs (report, raw trials, summary, config)
tests/                  `make test` — no network, no spend
```

## License

MIT (see [`LICENSE`](LICENSE)).
