# Model-loading strategies on Runpod Flash — run `qwen3b-gs`

- **Model:** `Qwen/Qwen2.5-3B-Instruct` (5.76 GiB of weights/tokenizer)
- **GPU:** NVIDIA GeForce RTX 4090 · **Datacenter:** EU-RO-1
- **Trials:** 5 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.6.0, huggingface_hub 1.33.0, runpod 1.10.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.17.0

## Answer

**network volume: volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `volume` 16.3s (loading straight off the mount: 16.1s [12.0–21.0]) vs `volumecache` 6.7s (hydrate 4.7s [4.6–6.3] + load from container disk 1.3s [1.2–1.5]) — volume is 2.44× slower. Client-observed cold request: volume 44.6s vs volumecache 25.2s (medians; includes scheduler placement, which the strategy does not control).

**Global Store: gsvolumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `globalstore` 48.1s (loading straight off the mount: 48.0s [37.9–63.0]) vs `gsvolumecache` 24.3s (hydrate 21.1s [17.1–26.1] + load from container disk 1.5s [1.3–2.6]) — globalstore is 1.98× slower. Client-observed cold request: globalstore 92.5s vs gsvolumecache 46.5s (medians; includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): volumecache 25.2s < nocache 35.2s < volume 44.6s < gsvolumecache 46.5s < globalstore 92.5s < modelstore 106.0s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate | ↳ Hub | load (tok+model) | **storage total** | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 35.2s [26.1–1115.8] | 15.7s [9.3–1073.3] | 20.2s [16.1–41.8] | 3.2s [2.5–5.8] | 13.00s [10.04–28.57] | — | 13.00s [10.04–28.57] | 1.72s [1.32–2.72] | **14.72s [11.40–31.29]** | 2.18s [1.94–4.47] | 5 |
| volume | 44.6s [25.8–49.3] | 18.0s [8.6–18.7] | 26.0s [16.7–31.0] | 4.8s [2.5–5.0] | 0.21s [0.15–0.29] | — | — | 16.06s [12.00–21.00] | **16.31s [12.15–21.18]** | 4.26s [1.77–4.43] | 5 |
| volumecache | 25.2s [23.0–594.6] | 13.8s [11.8–583.6] | 11.1s [10.8–13.8] | 2.5s [2.5–3.1] | 5.47s [5.06–6.87] | 4.74s [4.58–6.28] | 0.43s [0.41–0.52] | 1.25s [1.21–1.54] | **6.67s [6.31–8.41]** | 1.75s [1.64–2.20] | 5 |
| modelstore | 106.0s [78.2–144.8] | 98.6s [70.8–127.5] | 7.1s [6.8–16.9] | 2.7s [2.6–5.0] | 0.00s [0.00–0.00] | — | — | 2.28s [2.02–6.94] | **2.28s [2.02–6.94]** | 1.77s [1.73–4.26] | 4 (+1 invalid) |
| globalstore | 92.5s [68.2–144.2] | 30.4s [18.0–79.8] | 57.9s [48.6–72.6] | 5.1s [5.0–5.6] | 0.05s [0.04–0.15] | — | — | 48.05s [37.85–63.04] | **48.10s [38.01–63.08]** | 4.38s [4.20–4.68] | 4 (+1 invalid) |
| gsvolumecache | 46.5s [40.3–632.3] | 15.8s [11.9–600.6] | 31.1s [24.1–33.8] | 2.9s [2.7–5.0] | 21.68s [17.55–26.56] | 21.10s [17.08–26.08] | 0.43s [0.40–0.53] | 1.46s [1.27–2.58] | **24.26s [18.82–28.02]** | 2.04s [1.76–4.21] | 5 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 3.0s [2.2–5.1] | 73.6 [32.5–91.4] | 10 |
| volume | 4.8s [2.3–5.2] | 36.5 [34.7–92.1] | 10 |
| volumecache | 2.9s [2.3–3.0] | 91.3 [77.6–94.4] | 10 |
| modelstore | 3.0s [2.2–5.1] | 89.4 [33.9–93.0] | 8 |
| globalstore | 5.0s [4.4–5.2] | 35.6 [33.9–37.1] | 8 |
| gsvolumecache | 3.4s [2.4–5.6] | 59.3 [33.8–92.5] | 10 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 75.3s | 51.9s | 13.8s | download_s 13.8s |
| volume | 34.4s | 10.3s | 7.3s | download_s 7.0s |
| volumecache | 103.1s | 27.2s | 62.8s | hub_s 50.8s, hydrate_s 0.0s, sync_s 11.8s |
| modelstore | 78.8s | 71.2s | 0.0s | resolve_s 0.0s |
| globalstore | 122.0s | 34.1s | 33.3s | download_s 10.7s, upload_s 22.1s |
| gsvolumecache | 82.3s | 18.6s | 51.2s | hub_s 21.7s, hydrate_s 0.3s, sync_s 28.6s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volumecache` needed the Hugging Face Hub on 5/5 cold starts (0.43s [0.41–0.52] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `modelstore`: 1 cold trial(s) invalid: ['no result after 1200s']
- `globalstore`: 1 cold trial(s) invalid: ['no result after 1200s']
- `gsvolumecache` needed the Hugging Face Hub on 5/5 cold starts (0.43s [0.40–0.53] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `modelstore` waited 98.6s [70.8–127.5] for a worker on cold starts vs a median of 15.8s for the other strategies. That platform delay, not loading, dominates its cold start — consistent with the documented behaviour of holding workers until a host that has the cached model is available (the delay is not billed as worker time).

## Strategies

- **nocache** — no cache (HF download every cold start)
- **volume** — network volume, loaded in place
- **volumecache** — VolumeCache (volume -> container disk)
- **modelstore** — Runpod Model Store (cached model)
- **globalstore** — Runpod Global Store (global volume), loaded in place
- **gsvolumecache** — VolumeCache (Global Store -> container disk)

## Caveats

- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. Look at the [min–max] range before reading small differences.
- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.
- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.
- `GPU` names a class, not a machine: hosts in the same class differ (see the warm tokens/s range), which shows up in import, load and generate times alike.
- Cold rounds ran strategies concurrently in waves; strategies sharing storage (`volume`/`volumecache`, `globalstore`/`gsvolumecache`) were never in the same wave. (The prime phase runs all at once.)
- Model Store is attached with the console's GraphQL `modelReferences` field; Flash 1.20 has no parameter for it.
- Global Store (beta) is attached with the console's saveEndpoint `volumes` input (`OBJECT_STORE_VOLUME`); Flash 1.20 has no parameter for it. Its endpoints are pinned to the same datacenter as the rest for comparability, although a global volume would let them run anywhere.
- Global Store is object storage behind a FUSE client (geesefs), which keeps its own memory cache that `posix_fadvise` cannot evict.
- `globalstore` keeps a plain directory of real files, not an HF cache: HF snapshot symlinks written by one worker come back as 0-byte regular files on the next worker's mount.
- `gsvolumecache` uses VolumeCache with a copyfile patch: stock VolumeCache's copy2 fails (EPERM on chmod/utime) and silently leaves every large file out of the mirror.

