# Model-loading strategies on Runpod Flash — run `h100-qwen3b-all`

- **Model:** `Qwen/Qwen2.5-3B-Instruct` (5.76 GiB of weights/tokenizer)
- **GPU:** NVIDIA H100 80GB HBM3 · **Datacenter:** US-CA-2
- **Trials:** 5 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.8.0, huggingface_hub 2.4.0, runpod 2.2.0, runpod_code 2.2.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.22.2
- **Combined:** top-up cold/warm trials renumbered r6+; its prime excluded (main run's prime is the first-ever request). `globalstore`, `gsvolumecache`, `gshydrate` measured 2026-11-12 in run `h100gs-qwen3b-1112`; the rest measured 2026-11-11. Drift check: `volumecache` re-run alongside Global Store had median storage 6.1s / cold e2e 24.8s (n=5) vs 6.6s / 109.7s (n=5) here; its re-run rows are excluded from the tables.

## Answer

**network volume: volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `volume` 16.2s (loading straight off the mount: 16.1s [11.3–18.1]) vs `volumecache` 6.6s (hydrate 2.0s [1.7–8.1] + load from container disk 3.7s [3.3–4.5]) — volume is 2.45× slower. Client-observed cold request: volume 34.8s vs volumecache 109.7s (medians; includes scheduler placement, which the strategy does not control).

**network volume (prefetched): volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `volumeprefetch` 7.0s (loading straight off the mount: 4.3s [3.9–5.7]) vs `volumecache` 6.6s (hydrate 2.0s [1.7–8.1] + load from container disk 3.7s [3.3–4.5]) — volumeprefetch is 1.06× slower. Client-observed cold request: volumeprefetch 114.1s vs volumecache 109.7s (medians; includes scheduler placement, which the strategy does not control).

**Global Store: gsvolumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `globalstore` 50.2s (loading straight off the mount: 50.1s [26.8–64.2]) vs `gsvolumecache` 26.8s (hydrate 22.9s [22.2–24.9] + load from container disk 3.5s [3.1–3.8]) — globalstore is 1.87× slower. Client-observed cold request: globalstore 103.8s vs gsvolumecache 85.0s (medians; includes scheduler placement, which the strategy does not control).

**Global Store (parallel hydrate): gshydrate** gets the model onto the GPU faster. Median storage time (acquire + load): `globalstore` 50.2s (loading straight off the mount: 50.1s [26.8–64.2]) vs `gshydrate` 26.4s (hydrate 22.3s [16.5–28.3] + load from container disk 3.6s [3.4–4.0]) — globalstore is 1.90× slower. Client-observed cold request: globalstore 103.8s vs gshydrate 89.1s (medians; includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): volume 34.8s < modelstore 83.2s < gsvolumecache 85.0s < gshydrate 89.1s < globalstore 103.8s < volumecache 109.7s < nocache 114.0s < volumeprefetch 114.1s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate/prefetch | ↳ Hub | load (tok+model) | **storage total** | shared-storage read | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 114.0s [33.6–381.4] | 83.6s [12.6–354.7] | 21.2s [20.8–36.6] | 2.2s [2.0–2.7] | 12.87s [12.60–28.02] | — | 12.87s [12.59–28.01] | 3.79s [3.78–3.98] | **16.66s [16.37–31.79]** | — | 2.26s [2.17–2.38] | 5 |
| volume | 34.8s [32.5–197.1] | 13.2s [11.4–174.6] | 20.4s [15.6–23.0] | 2.1s [1.9–2.8] | 0.03s [0.02–0.04] | — | — | 16.12s [11.27–18.15] | **16.15s [11.29–18.18]** | 374 MiB/s [331–539] | 2.15s [2.07–2.59] | 5 (+1 invalid) |
| volumecache | 109.7s [24.5–236.4] | 97.3s [14.4–225.7] | 11.3s [10.1–17.5] | 2.0s [2.0–2.1] | 3.03s [2.43–8.50] | 1.97s [1.72–8.06] | 0.34s [0.27–0.99] | 3.75s [3.30–4.52] | **6.59s [5.74–13.02]** | 2999 MiB/s [732–3434] | 2.20s [2.13–2.39] | 5 |
| volumeprefetch | 114.1s [26.9–176.4] | 102.2s [13.2–162.1] | 11.8s [11.3–13.8] | 2.1s [2.0–2.4] | 3.11s [2.63–3.79] | 2.80s [2.55–3.27] | — | 4.26s [3.87–5.70] | **6.97s [6.89–8.85]** | 2107 MiB/s [1804–2310] | 2.28s [2.13–2.79] | 5 (+1 invalid) |
| modelstore | 83.2s [70.5–321.6] | 72.5s [58.8–310.4] | 10.8s [10.6–10.9] | 2.1s [1.9–2.1] | 0.00s [0.00–0.00] | — | — | 5.91s [5.69–6.23] | **5.91s [5.69–6.23]** | 1044 MiB/s [984–1083] | 2.17s [2.11–2.56] | 5 (+2 invalid) |
| globalstore | 103.8s [73.5–393.5] | 71.6s [29.4–338.6] | 54.7s [32.1–68.8] | 2.1s [2.0–2.3] | 0.08s [0.07–0.89] | — | — | 50.08s [26.83–64.19] | **50.16s [27.72–64.27]** | 122 MiB/s [95–222] | 2.16s [2.12–2.73] | 5 |
| gsvolumecache | 85.0s [45.6–165.8] | 53.8s [13.5–134.4] | 32.0s [30.9–34.2] | 2.1s [2.0–2.3] | 23.68s [23.03–25.41] | 22.88s [22.21–24.91] | 0.41s [0.39–0.69] | 3.52s [3.15–3.85] | **26.82s [26.48–29.26]** | 258 MiB/s [237–266] | 2.18s [2.10–2.71] | 5 (+1 invalid) |
| gshydrate | 89.1s [40.0–1114.0] | 54.2s [12.8–1077.8] | 30.7s [25.6–36.2] | 2.0s [1.9–2.1] | 23.06s [17.61–28.37] | 22.25s [16.50–28.30] | — | 3.58s [3.37–3.99] | **26.43s [21.18–31.94]** | 265 MiB/s [208–357] | 2.14s [2.08–2.22] | 5 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.
- *shared-storage read*: model bytes ÷ the step that reads the volume/store — the model load for in-place strategies (`volume`, `globalstore`, `modelstore`), the hydrate or prefetch otherwise.

## Overall cold request (what a client waits for)

| strategy | cold request e2e | platform delay | handler exec | best-case e2e | best-case delay | n |
|---|---|---|---|---|---|---|
| volumecache | 109.7s [24.5–236.4] | 97.3s [14.4–225.7] | 11.3s [10.1–17.5] | 24.5s | 14.4s | 5 |
| volumeprefetch | 114.1s [26.9–176.4] | 102.2s [13.2–162.1] | 11.8s [11.3–13.8] | 26.9s | 13.2s | 5 |
| volume | 34.8s [32.5–197.1] | 13.2s [11.4–174.6] | 20.4s [15.6–23.0] | 32.5s | 11.4s | 5 |
| nocache | 114.0s [33.6–381.4] | 83.6s [12.6–354.7] | 21.2s [20.8–36.6] | 33.6s | 12.6s | 5 |
| gshydrate | 89.1s [40.0–1114.0] | 54.2s [12.8–1077.8] | 30.7s [25.6–36.2] | 40.0s | 12.8s | 5 |
| gsvolumecache | 85.0s [45.6–165.8] | 53.8s [13.5–134.4] | 32.0s [30.9–34.2] | 45.6s | 13.5s | 5 |
| modelstore | 83.2s [70.5–321.6] | 72.5s [58.8–310.4] | 10.8s [10.6–10.9] | 70.5s | 58.8s | 5 |
| globalstore | 103.8s [73.5–393.5] | 71.6s [29.4–338.6] | 54.7s [32.1–68.8] | 73.5s | 29.4s | 5 |

Medians [min–max]; ranked by best case (minimum), which approximates a cold start without a GPU-capacity wait. Platform delay includes everything before the handler runs — queueing, worker start, and for `modelstore` the platform staging the model or placing the worker on a host that has it — so compare strategies here, not on storage time alone.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 2.3s [2.2–2.4] | 66.1 [63.7–71.5] | 10 |
| volume | 2.3s [2.1–3.0] | 66.9 [54.9–73.7] | 10 |
| volumecache | 2.3s [2.2–2.3] | 67.1 [58.7–71.4] | 10 |
| volumeprefetch | 2.3s [2.2–2.9] | 68.4 [55.7–72.2] | 10 |
| modelstore | 2.3s [2.2–3.0] | 70.6 [60.9–73.1] | 10 |
| globalstore | 2.2s [2.1–3.0] | 71.0 [54.7–73.6] | 10 |
| gsvolumecache | 2.3s [2.0–3.0] | 70.6 [56.6–72.0] | 10 |
| gshydrate | 2.3s [2.1–2.3] | 72.2 [69.5–72.9] | 10 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 108.2s | 78.4s | 21.0s | download_s 21.0s |
| volume | 121.6s | 79.6s | 21.4s | download_s 21.4s |
| volumecache | 52.5s | 28.9s | 15.5s | hub_s 11.6s, hydrate_s 0.0s, sync_s 3.8s |
| volumeprefetch | 177.8s | 166.2s | 3.1s | prefetch_s 3.1s, resolve_s 0.0s |
| modelstore | 82.8s | 71.6s | 0.0s | resolve_s 0.0s |
| globalstore | 130.0s | 23.1s | 54.6s | download_s 12.2s, upload_s 42.3s |
| gsvolumecache | 703.2s | 651.3s | 43.8s | hub_s 13.0s, hydrate_s 0.5s, sync_s 29.1s |
| gshydrate | 82.7s | 52.6s | 22.0s | hydrate_s 21.6s, resolve_s 0.0s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volume`: 1 cold trial(s) invalid: ['no result after 1200s']
- `volumecache` needed the Hugging Face Hub on 5/5 cold starts (0.34s [0.27–0.99] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `volumeprefetch`: 1 cold trial(s) invalid: ['no result after 1200s']
- `modelstore`: 2 cold trial(s) invalid: ['no result after 1200s']
- `gsvolumecache` needed the Hugging Face Hub on 5/5 cold starts (0.41s [0.39–0.69] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `gsvolumecache`: 1 cold trial(s) invalid: ['status of job 3b00969c-7e33-4e4a-b885-5f7f170bfc0c-u1 on x2scdd4zlqpb2e: ConnectTimeout: timed out (after 5 attempts)']

## Strategies

- **nocache** — no cache (HF download every cold start)
- **volume** — network volume, loaded in place
- **volumecache** — VolumeCache (volume -> container disk)
- **volumeprefetch** — network volume, parallel page-cache prefetch, loaded in place
- **modelstore** — Runpod Model Store (cached model)
- **globalstore** — Runpod Global Store (global volume), loaded in place
- **gsvolumecache** — VolumeCache (Global Store -> container disk)
- **gshydrate** — parallel range hydrate (Global Store -> container disk)

## Caveats

- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. Look at the [min–max] range before reading small differences.
- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.
- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.
- `GPU` names a class, not a machine: hosts in the same class differ (see the warm tokens/s range), which shows up in import, load and generate times alike.
- Cold rounds ran strategies concurrently in waves; strategies sharing storage (`volume`/`volumecache`/`volumeprefetch`, `globalstore`/`gsvolumecache`/`gshydrate`) were never in the same wave. (The prime phase runs all at once.)
- Model Store is declared with `model=runpod.Model(...)` on the endpoint (sets `modelReferences`).
- Global Store (beta) is declared as a `runpod.GlobalVolume` mount. Its endpoints are pinned to the same datacenter as the rest for comparability, although a global volume would let them run anywhere.
- Global Store is object storage behind a FUSE client (geesefs), which keeps its own memory cache that `posix_fadvise` cannot evict.
- `globalstore` keeps a plain directory of real files, not an HF cache: HF snapshot symlinks written by one worker come back as 0-byte regular files on the next worker's mount.
- `gsvolumecache` uses stock VolumeCache (runpod>=2.1.0 copies with copyfile; older releases' copy2 failed with EPERM on chmod/utime there and silently left every large file out of the mirror).

## Time and cost

Trials and time/cost combined from `h100-qwen3b-1111`, `h100-qwen3b-1111-topup`, `h100gs-qwen3b-1112` (sums over those runs).

- wall clock (deploy → teardown): 182.4 min
- handler time (sum of `executionTime`): 26.6 GPU-min
- billed GPU time: 58.5 GPU-min, $5.26 (Runpod billing API, this run's endpoints only)
- billed by strategy: globalstore 9.4, gshydrate 5.3, gsvolumecache 10.4, modelstore 4.4, nocache 5.6, volume 5.9, volumecache 8.2, volumeprefetch 9.1
