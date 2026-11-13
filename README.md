# flash-model-cache-bench

**What is the fastest way to get a Hugging Face model onto a Runpod Serverless GPU on a cold start?**

This repo benchmarks eight model-storage strategies side by side on real Runpod Serverless GPU
workers deployed with Runpod Flash — the `runpod.apps` SDK and `rp flash` CLI built into
[runpod-python](https://github.com/runpod/runpod-python) 2.2+. It measures the whole AI
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

Measured on **NVIDIA H100 80GB HBM3** workers (`GPU=ADA_80_PRO`; every trial landed on H100 SXM) in
`US-CA-2`, with runpod-python 2.2.1, torch 2.9.1+cu128 and transformers 5.22.2. Eight strategies,
**5 valid forced-cold trials each**, page cache evicted before every storage read. The network-volume,
Model Store and download strategies ran 2026-11-11; the three Global Store strategies ran 2026-11-12
on the same setup (see caveats for the drift check).
Medians unless noted; ranges and raw data are in [`results/`](results/).

### Time the storage strategy controls (get the bytes + load them to GPU)

| strategy | Qwen2.5-3B-Instruct (5.8 GiB) | Qwen2.5-7B-Instruct (14.2 GiB) | read off shared storage (3B / 7B) |
|---|---|---|---|
| `nocache`: download from HF + load | 16.7s | 39.7s | – |
| `volume`: load from the network volume (lazy safetensors mmap) | 16.2s | 26.6s | 374 / 554 MiB/s |
| `volumeprefetch`: same files, parallel page-cache prefetch, then load in place | 7.0s (2.8 + 4.3) | 12.5s (5.1 + 6.1) | 2107 / 2840 MiB/s |
| `volumecache`: hydrate to container disk + local load | 6.6s (2.0 + 3.7) | **8.6s** (2.6 + 5.4) | 2999 / 5624 MiB/s |
| `modelstore`: load from Runpod cached model | **5.9s** | 9.7s | 1044 / 1535 MiB/s |
| `globalstore`: load directly from Global Store | 50.2s | 124.1s | 122 / 119 MiB/s |
| `gsvolumecache`: VolumeCache hydrate from Global Store + local load | 26.8s (22.9 + 3.5) | 47.4s (41.2 + 5.4) | 258 / 353 MiB/s |
| `gshydrate`: 64 parallel 8 MiB range reads from Global Store + local load | 26.4s (22.3 + 3.6) | 48.5s (42.4 + 5.4) | 265 / 343 MiB/s |

### Whole cold request, as a client sees it

Model Store stages the model before the container starts, so its cost is in Runpod's `delayTime`,
not in storage time. GPU capacity waits also land in `delayTime` and hit every strategy at random,
which makes medians noisy. The **best case** (minimum over 5 trials) approximates a cold start with
no capacity wait:

| strategy | best case (3B / 7B) | of which platform delay (3B / 7B) | median (3B / 7B) |
|---|---|---|---|
| `volumecache` | **24.5s / 23.6s** | 14.4s / 11.2s | 109.7s / **30.3s** |
| `volumeprefetch` | 26.9s / 28.1s | 13.2s / 11.8s | 114.1s / 69.3s |
| `volume` | 32.5s / 46.2s | 11.4s / 9.5s | **34.8s** / 104.8s |
| `nocache` | 33.6s / 51.5s | 12.6s / 9.9s | 114.0s / 58.9s |
| `modelstore` | 70.5s / 95.1s | **58.8s / 80.3s** | 83.2s / 107.0s |
| `gshydrate` | 40.0s / 106.4s | 12.8s / 53.6s | 89.1s / 160.9s |
| `gsvolumecache` | 45.6s / 107.0s | 13.5s / 55.3s | 85.0s / 177.3s |
| `globalstore` | 73.5s / 110.9s | 29.4s / 17.0s | 103.8s / 245.3s |

The Global Store rows come from a day with tighter H100 capacity (waits up to 19 minutes), so their
platform delay is less comparable than their storage time above.

### What this means

1. **Loading in place from a network volume is slow because of the access pattern, not the
   volume.** The default lazy (mmap) safetensors load read the volume at 374–554 MiB/s. Reading
   the *same files on the same volume* with 8 concurrent 64 MiB range reads first (`volumeprefetch`,
   the approach behind vLLM's `prefetch` load strategy) ran at 2.1–2.8 GiB/s, 5–6× faster, and the
   load that followed was served from RAM. That closed 96% (3B) and 78% (7B) of the storage-time gap
   between loading in place and VolumeCache.
2. **VolumeCache is still the fastest way off a network volume, end to end.** It had the best
   best-case cold request for both models (~24s) and the best 7B median. Its copy (3.0–5.6 GiB/s)
   beat the prefetch's pread-and-discard reader, which is untuned: treat 2.1–2.8 GiB/s as a lower
   bound for a parallel in-place reader.
3. **Model Store loads fastest once the worker runs, and is slowest end to end.** Its handler was the
   quickest (11–14s), but its platform delay never went below 59s (3B) or 80s (7B), against ~10–14s
   for every other strategy. Two of its 3B trials got no worker within 20 minutes. From these numbers
   we can't tell whether that time is staging the model or waiting for a host that has it; both
   land in `delayTime`.
4. **A network volume clearly beats a fresh download only with a fast reader.** Plain in-place
   loading was on par with downloading from Hugging Face for 3B (16.2 vs 16.7s) and ahead for 7B
   (26.6 vs 39.7s); prefetched or hydrated, it was 2.4–4.6× faster than the download.
5. **Global Store is the slowest place to keep a model, slower than downloading it.** Loading in
   place read it at ~120 MiB/s (3B 50s, 7B 124s). Parallel reads, either VolumeCache's hydrate or 64
   concurrent 8 MiB range reads, roughly doubled that to 260–350 MiB/s, but storage time (26–27s
   for 3B, 47–49s for 7B) was still behind a fresh Hugging Face download (16.7s / 39.7s) and 4–6×
   behind the network-volume caches. On a network volume the reader was the bottleneck; on Global
   Store the storage is.
6. **Once the model is in VRAM, storage doesn't matter.** Warm medians were 66–72 tokens/s (3B) and
   84–89 tokens/s (7B) across strategies.

Caveats:

- H100 capacity in `US-CA-2` was tight. Four cold trials got no worker within the 20-minute trial
  timeout; those strategies (`volume`, `volumeprefetch`, `modelstore` on 3B) got their fifth trial
  from a top-up run with the same GPU, datacenter and code, merged into the 3B report.
- The Global Store strategies ran a day later. That run re-ran `volumecache` as a drift check: median
  storage 6.1s (3B) and 9.1s (7B) against 6.6s and 8.6s on 2026-11-11. Its re-run trials are left out
  of the tables, and each report's header records the merge.
- A100s were not usable on 2026-11-11: A100 workers in `EU-RO-1` waited up to 21 minutes, and a
  probe of A100/H100 in `EU-RO-1`, `US-IL-1` and `US-CA-2` got workers only for H100 in `US-CA-2`.
- Exact seconds vary with datacenter, host and time of day. Compare strategies within a run.

Full reports: [`results/h100-qwen3b/report.md`](results/h100-qwen3b/report.md) ·
[`results/h100-qwen7b/report.md`](results/h100-qwen7b/report.md).

Cost:

| model | 2026-11-11 run | 2026-11-12 Global Store run | total |
|---|---|---|---|
| 3B | 109.5 min, 29.4 GPU-min ($2.64) | 72.9 min, 29.2 GPU-min ($2.62) | **182.4 min, 58.5 GPU-min ($5.26)** |
| 7B | 33.1 min, 25.8 GPU-min ($2.32) | 117.0 min, 39.9 GPU-min ($3.58) | **150.1 min, 65.6 GPU-min ($5.90)** |

Each cell is wall clock and billed GPU time; the 3B 2026-11-11 run includes its top-up. Billed time is
1.4–2.6× the summed handler time, because worker start-up is billed too.

---

## Earlier runs

### 2026-09-25: six strategies including Global Store (RTX 4090, `EU-RO-1`)

Measured on NVIDIA RTX 4090 workers in `EU-RO-1`, with Flash 1.20.0, transformers 5.17.0
and torch 2.9.1. All values are **medians over forced cold starts**; ranges and raw data are in
[`results/`](results/).

#### Time the storage strategy controls

| strategy | Qwen2.5-3B-Instruct (5.8 GiB) | Qwen2.5-7B-Instruct (14.2 GiB) |
|---|---|---|
| `nocache`: download from HF + load | 14.7s | 23.5s |
| `volume`: load directly from network volume | 16.3s | 34.8s |
| `volumecache`: hydrate from network volume + local load | **6.7s** (4.7 + 1.3) | **16.2s** (11.0 + 4.6) |
| `modelstore`: load from Runpod cached model | 2.3s | 14.2s |
| `globalstore`: load directly from Global Store | 48.1s | 98.9s |
| `gsvolumecache`: hydrate from Global Store + local load | 24.3s (21.1 + 1.5) | 81.4s ⚠️ see finding 4 |

#### Whole cold request, as a client sees it

| strategy | 3B | 7B | of which: waiting for a worker (`delayTime`, 3B / 7B) |
|---|---|---|---|
| `volumecache` | **25.2s** | 43.4s | 13.8s / 16.3s |
| `nocache` | 35.2s | **41.0s** | 15.7s / 11.1s |
| `volume` | 44.6s | 62.1s | 18.0s / 17.8s |
| `gsvolumecache` | 46.5s | 108.4s | 15.8s / 18.2s |
| `globalstore` | 92.5s | 133.6s | 30.4s / 20.2s |
| `modelstore` | 106.0s | 81.2s | **98.6s / 69.3s** |

#### What this means

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

### 2026-11-03: re-run on runpod 2.1.0 (RTX 4090, `EU-RO-1`)

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

runpod 2.2.1 keeps the 64 × 8 MiB default, yet on H100 workers in `US-CA-2` (above) VolumeCache hydrated
at ~3.0 GiB/s (3B) and ~5.6 GiB/s (7B). The 2.1.0 slowdown therefore depends on the host and volume
cluster, not only on the default.

### 2026-11-13: runpod 2.2.1, VolumeCache default vs explicit 8 × 64 MiB (RTX 4090, `EU-RO-1`)

Same small setup as the 2.1.0 re-run (3B, 3 cold trials, RTX 4090, `EU-RO-1`) on runpod 2.2.1, plus
`volumecache864`: the same VolumeCache mirror hydrated with `hydrate_workers=8,
hydrate_chunk_size=64 MiB`, the network-volume default on runpod-python main (#691, not in 2.2.1).
Stock `volumecache` still measures the SDK default (64 × 8 MiB in 2.2.1), so one run shows the
released default and the next release side by side
([`results/qwen3b-rp221`](results/qwen3b-rp221/report.md)). Median storage time (acquire + load):

| strategy | 2026-11-03 (runpod 2.1.0) | 2026-11-13 (runpod 2.2.1) |
|---|---|---|
| `nocache` | 12.0s | 13.2s |
| `volume` | 21.2s | 17.8s |
| `volumecache` (SDK default, 64 × 8 MiB) | 23.1s (17.8 + 4.2) | **16.8s** (13.1 hydrate + 3.1 load) |
| `volumecache864` (8 × 64 MiB) | – | 17.3s (12.7 hydrate + 3.9 load) |
| `globalstore`, `gshydrate` | 36.6s, 15.0s | not run |

Per-trial hydrate of the same 5.8 GiB mirror, page cache evicted first, the two variants never in
the same wave:

| | trial 1 | trial 2 | trial 3 |
|---|---|---|---|
| `volumecache` (64 × 8 MiB) | 12.8s | 13.1s | 20.7s |
| `volumecache864` (8 × 64 MiB) | 12.7s | **5.2s** | 14.0s |

**On this run, 8 × 64 MiB was not a reliable win.** It hydrated at ~1.1 GiB/s on one worker out of
three (matching the 2.1.0 microbenchmark) and at ~420–470 MiB/s on the other two, the same rate as
the default (285–462 MiB/s). That rate is close to the ~460 MiB/s of a plain sequential copy
measured on 2.1.0. A per-host volume read ceiling seems to cap most workers whatever the hydrate
settings. Three trials per variant can't separate them. The fast trials (here, and the H100 run
above) come from the host, not from the settings.

Also checked on 2.2.1:

- **Workers report the code that runs (#708).** `runpod.__version__` was 2.2.1 on every worker, so
  the harness dropped its `runpod._version` shim (`versions.runpod_code`). It now records
  `versions.runpod` (imported tree) and `versions.runpod_dist` (installed dist metadata, which can be
  the worker image's runpod). In this run both were 2.2.1 on every worker. The H100 runs above had
  2.2.0 metadata on some hosts.
- **Templates have unique names (#696)**, e.g. `hfbench-rp221-1113-hfbench-volume-template-4228d6`.
  All four outlived teardown. After the app, endpoints and volume were gone, the REST template list
  still showed the four `hfbench-rp221-1113-hfbench-{nocache,volume,volumecache,volumecache864}-template-4228d6`
  templates, and `DELETE /v1/templates/<id>` answered `500 template not found`. That fits #696's
  note that the platform removes a deleted endpoint's template later. With unique names, they no
  longer block a redeploy.
- No failures: 12/12 cold and 24/24 warm trials completed. Two cold trials (round 3 of `volume` and
  `volumecache`) waited ~100s for a worker. That is a platform capacity wait, inside `delayTime`, and
  it doesn't affect the storage times.

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

To reproduce the published H100 runs:

```bash
S=nocache,volume,volumecache,volumeprefetch,modelstore
make run GPU=ADA_80_PRO DATACENTER=US-CA-2 STRATEGIES=$S COLD_TRIALS=5 RUN_ID=h100-qwen3b
make run GPU=ADA_80_PRO DATACENTER=US-CA-2 STRATEGIES=$S COLD_TRIALS=5 VOLUME_GB=60 \
         MODEL=Qwen/Qwen2.5-7B-Instruct RUN_ID=h100-qwen7b
make billing RUN=h100-qwen3b    # an hour or more later: billed GPU-minutes and cost
```

`MAKEUP_ROUNDS` (default 3) re-runs only the strategies that end short of `COLD_TRIALS` valid
trials, e.g. when no worker started within the trial timeout.

Other targets:

| command | what it does |
|---|---|
| `make report [RUN=<id>]` | re-render a report from `runs/<id>/results.jsonl` |
| `make billing [RUN=<id>]` | fetch Runpod-billed GPU-minutes and cost for a finished run (billing lags ~1 h); adds "Time and cost" to the report |
| `make clean` | tear down anything a crashed or interrupted run left behind (safe to run any time) |
| `make test` | unit tests — no network, no spend |

### Time and cost

Every endpoint is capped at one worker (`workers=(0, 1)`), so at most one GPU worker per strategy
exists at a time, and only during a trial. The published H100 runs billed **25–30 GPU-minutes
($2.30–2.65) each**; wall time was 33 minutes (7B) and 110 minutes (3B, stretched by H100 capacity
waits, which aren't billed). A network volume (and a Global Store, if selected) exists only for the
run. `make billing RUN=<id>` adds wall clock, handler time and billed GPU-minutes/cost to the
report; Runpod's billing data can lag usage by several hours.

To go faster, run a subset: `make run STRATEGIES=volume,volumecache COLD_TRIALS=3`.

### Knobs

Set them in `.env` or on the command line (`make run MODEL=... COLD_TRIALS=5`).

| variable | default | notes |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-3B-Instruct` | any HF causal LM that fits the GPU in bf16 |
| `GPU` | `ADA_24` | `runpod.GpuGroup`/`GpuType` name(s) or shorthands, comma-separated (e.g. `AMPERE_48`, `4090`) |
| `DATACENTER` | `EU-RO-1` | must support network volumes |
| `VOLUME_GB` | `50` | needs ≈ 2.2 × model size (volume copy + VolumeCache mirror) |
| `STRATEGIES` | all eight | comma-separated subset of `nocache,volume,volumecache,modelstore,globalstore,gsvolumecache,gshydrate,volumeprefetch`, plus the optional `volumecache864`; only these are deployed |
| `COLD_TRIALS` | `3` | forced-cold requests per strategy |
| `MAKEUP_ROUNDS` | `3` | extra rounds, only for strategies short of `COLD_TRIALS` valid cold trials (e.g. no worker within the trial timeout) |
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
  It also shows each strategy's read rate off shared storage (MiB/s).
- **Overall cold request**: whole-request median, platform delay and handler time, plus the best
  case (minimum), ranked by best case. Use this to compare `modelstore`, whose staging happens
  before the container starts.
- **Warm requests**: latency and tokens/s with the model already in GPU memory.
- **First-ever request**: the one-time cost of filling each cache.
- **Observations**: flags generated automatically, such as Hub use on cold starts and unusually long
  worker waits.
- **Time and cost**: wall clock, summed handler time and (after `make billing`) billed GPU-minutes
  and cost, per strategy.

Raw data is in `results.jsonl`, one line per request, including everything the handler returned:
worker id, GPU, package versions, byte counts, and VolumeCache's hydrated/fetched/mirrored bytes
and copy errors.

---

## How it works

### The eight strategies

All eight are queue-based GPU endpoints running **the same handler code** on the same GPU class in
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
| `volumeprefetch` | the `volume` HF cache, read once into the page cache with 8 concurrent 64 MiB range reads, then loaded straight from the mount (pages come from RAM) |
| `volumecache864` (optional, not in the default set) | `volumecache` with explicit `hydrate_workers=8, hydrate_chunk_size=64 MiB`; shares its mirror |

A global volume and a network volume can't be attached to the same endpoint, so the
Global Store strategies get their own endpoints. Global volumes don't tie an endpoint to a
datacenter, but here they are pinned to `DATACENTER` like the rest, so all of them are comparable.

Each request then loads the tokenizer and model (`AutoModelForCausalLM`, bf16, `device_map="cuda"`)
and greedily generates exactly `MAX_NEW_TOKENS` tokens.

### A run, step by step

1. **Preflight**: API key present; no app `hfbench-<id>` exists yet.
2. **Deploy**: `rp flash deploy --env run-<id>` in `worker/` creates app `hfbench-<id>` with one
   endpoint per selected strategy. Storage is declared in `worker/bench_app.py` and created by
   the deploy: a `NetworkVolume` `hfbench-<id>` for `volume`/`volumecache`/`volumeprefetch`, a `GlobalVolume`
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
   - Strategies that share storage (`volume`/`volumecache`/`volumeprefetch`,
     `globalstore`/`gsvolumecache`/`gshydrate`) run in different waves, so they never compete for bandwidth.
   - Then up to `MAKEUP_ROUNDS` extra rounds, only for strategies short of `COLD_TRIALS` valid trials.
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
  so the benchmark reads from storage, not RAM. `volumeprefetch` evicts *before* its prefetch, so
  only its own read warms the cache; it records the container's cgroup `memory.max` (251 GB on the
  H100 workers), since a limit below the model size would evict prefetched pages before the load. This doesn't reach a FUSE client's own cache
  (MooseFS client, geesefs).
- **Medians instead of means.** Worker placement produces outliers (18-minute waits next to 10s
  ones) that pull a mean far from the typical case.
- **VolumeCache is the SDK's own.** `rp flash deploy` vendors the deploying client's runpod package
  into the build artifact, so workers run the same runpod (2.2.1) as your machine; `notes.impl`
  records the version. Since 2.1.0, VolumeCache hydrates big files with concurrent range reads (64 × 8 MiB
  by default) and treats file metadata as best-effort, so it no longer drops files on Global Store.

### Gaps found while building this

Status with runpod 2.1.0–2.2.1 (the harness was first built on standalone `runpod-flash` 1.20):

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
  Fixed in 2.2.1 (#708); see the 2026-11-13 section.
- **VolumeCache's default hydrate (64 × 8 MiB ranges) is ~3× slower than 8 × 64 MiB on a network
  volume.** See "Re-run on runpod 2.1.0" above.
  The 8 × 64 MiB default is on runpod-python main (#691), not in 2.2.1; `volumecache864` measures it.
- **`runpod.GpuType` has no L40 or L40S, and `GpuGroup.ADA_48_PRO` maps only to the RTX 6000 Ada**
  (2.2.1), so the SDK can't target L40S, a common datacenter card.
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
  mixes hardware types. To find capacity, run the harness itself as a probe in a few places:
  `make run STRATEGIES=nocache COLD_TRIALS=1 WARM_PER_COLD=0 GPU=... DATACENTER=... RUN_ID=probe-...`.
  Pod stock APIs don't reflect Serverless capacity.

---

## Layout

```
worker/                 `rp flash deploy` root (the build artifact's source)
  bench_app.py          runpod.App with one @app.queue endpoint per selected strategy, plus its storage
  runtime.py            request handling: cold load vs warm, timings, provenance
  strategies.py         the eight acquisition strategies (unit-tested)
bench/                  orchestrator (runs on your machine)
  __main__.py           run / report / billing / clean / preflight
  config.py             env-var config, validated before anything is deployed
  deploy.py             rp flash deploy/teardown, attachment checks, volume cleanup, state file for recovery
  driver.py             prime, forced-cold rounds in shared-storage waves, warm follow-ups
  runpod_api.py         job/health API; workersMax (REST v2 PATCH); app/env lookup; volume cleanup; billing
  report.py             results.jsonl -> report.md + summary.json
results/                the published runs (report, raw trials, summary, config)
tests/                  `make test` — no network, no spend
```

## License

MIT (see [`LICENSE`](LICENSE)).
