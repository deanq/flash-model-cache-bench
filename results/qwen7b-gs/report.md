# Model-loading strategies on Runpod Flash — run `qwen7b-gs`

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (14.20 GiB of weights/tokenizer)
- **GPU:** NVIDIA GeForce RTX 4090 · **Datacenter:** EU-RO-1
- **Trials:** 3 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.6.0, huggingface_hub 1.33.0, runpod 1.10.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.17.0

## Answer

**network volume: volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `volume` 34.8s (loading straight off the mount: 34.5s [34.5–35.4]) vs `volumecache` 16.2s (hydrate 11.0s [10.9–11.4] + load from container disk 4.6s [3.0–5.4]) — volume is 2.15× slower. Client-observed cold request: volume 62.1s vs volumecache 43.4s (medians; includes scheduler placement, which the strategy does not control).

**Global Store: gsvolumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `globalstore` 98.9s (loading straight off the mount: 98.8s [91.2–113.1]) vs `gsvolumecache` 81.4s (hydrate 40.7s [27.4–94.1] + load from container disk 4.3s [4.2–4.8]) — globalstore is 1.22× slower. Client-observed cold request: globalstore 133.6s vs gsvolumecache 108.4s (medians; includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): nocache 41.0s < volumecache 43.4s < volume 62.1s < modelstore 81.2s < gsvolumecache 108.4s < globalstore 133.6s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate | ↳ Hub | load (tok+model) | **storage total** | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 41.0s [28.4–59.5] | 11.1s [9.6–14.7] | 29.6s [18.4–44.3] | 2.9s [2.6–4.8] | 19.45s [9.87–27.44] | — | 19.45s [9.87–27.43] | 4.09s [3.06–7.81] | **23.54s [12.92–35.25]** | 2.78s [2.60–3.92] | 3 |
| volume | 62.1s [58.5–102.7] | 17.8s [14.2–56.7] | 43.5s [43.5–45.2] | 4.9s [4.8–5.6] | 0.26s [0.24–0.34] | — | — | 34.52s [34.47–35.42] | **34.81s [34.79–35.66]** | 3.56s [3.49–3.60] | 3 |
| volumecache | 43.4s [36.3–44.4] | 16.3s [15.8–19.2] | 25.0s [20.4–26.6] | 5.0s [2.7–5.0] | 11.85s [11.53–11.92] | 11.03s [10.85–11.35] | 0.54s [0.42–0.77] | 4.64s [2.97–5.35] | **16.16s [14.82–17.27]** | 3.56s [2.56–3.94] | 3 |
| modelstore | 81.2s [41.6–626.7] | 69.3s [17.0–602.7] | 23.6s [11.5–24.3] | 4.7s [2.6–4.9] | 0.00s [0.00–0.00] | — | — | 14.21s [5.53–15.21] | **14.21s [5.53–15.22]** | 3.56s [2.57–3.65] | 3 |
| globalstore | 133.6s [117.7–143.3] | 20.2s [17.5–24.7] | 108.5s [100.0–122.3] | 5.0s [4.9–5.6] | 0.05s [0.05–0.14] | — | — | 98.81s [91.23–113.10] | **98.95s [91.28–113.15]** | 3.63s [3.49–3.83] | 3 |
| gsvolumecache | 108.4s [77.6–164.6] | 18.2s [14.7–19.9] | 89.8s [62.7–144.5] | 4.7s [2.5–5.0] | 77.07s [53.12–130.80] | 40.71s [27.39–94.08] | 8.36s [7.44–10.08] | 4.30s [4.24–4.76] | **81.37s [57.36–135.56]** | 3.42s [2.53–3.62] | 3 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 3.7s [3.7–4.4] | 55.5 [43.0–55.8] | 6 |
| volume | 4.0s [3.6–4.5] | 43.9 [42.9–44.7] | 6 |
| volumecache | 4.1s [3.6–4.5] | 44.8 [42.9–54.5] | 6 |
| modelstore | 4.3s [3.7–4.4] | 44.5 [42.9–56.3] | 6 |
| globalstore | 4.4s [3.7–4.5] | 43.8 [42.9–45.1] | 6 |
| gsvolumecache | 4.3s [3.7–4.5] | 43.3 [42.9–56.2] | 6 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 558.4s | 534.5s | 14.2s | download_s 14.2s |
| volume | 107.5s | 39.2s | 17.1s | download_s 16.6s |
| volumecache | 42.1s | 9.7s | 24.2s | hub_s 14.3s, hydrate_s 0.0s, sync_s 9.8s |
| modelstore | 31.1s | 15.7s | 0.0s | resolve_s 0.0s |
| globalstore | 204.2s | 16.5s | 85.1s | download_s 23.2s, upload_s 60.7s |
| gsvolumecache | 115.9s | 54.6s | 53.1s | hub_s 14.8s, hydrate_s 0.3s, sync_s 37.5s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volumecache` needed the Hugging Face Hub on 3/3 cold starts (0.54s [0.42–0.77] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `gsvolumecache` needed the Hugging Face Hub on 3/3 cold starts (8.36s [7.44–10.08] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `modelstore` waited 69.3s [17.0–602.7] for a worker on cold starts vs a median of 17.8s for the other strategies. That platform delay, not loading, dominates its cold start — consistent with the documented behaviour of holding workers until a host that has the cached model is available (the delay is not billed as worker time).

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

