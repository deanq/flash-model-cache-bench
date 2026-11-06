# Model-loading strategies on Runpod Flash — run `rp21-a`

- **Model:** `Qwen/Qwen2.5-3B-Instruct` (5.76 GiB of weights/tokenizer)
- **GPU:** NVIDIA GeForce RTX 4090 · **Datacenter:** EU-RO-1
- **Trials:** 3 forced-cold per strategy, 1 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.8.0, huggingface_hub 2.4.0, runpod 2.0.0, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.22.1

## Answer

**network volume: volume** gets the model onto the GPU faster. Median storage time (acquire + load): `volume` 21.2s (loading straight off the mount: 20.9s [19.8–23.8]) vs `volumecache` 23.1s (hydrate 17.8s [16.1–19.2] + load from container disk 4.2s [4.1–4.7]) — volumecache is 1.09× slower. Client-observed cold request: volume 43.3s vs volumecache 59.8s (medians; includes scheduler placement, which the strategy does not control).

**Global Store (parallel hydrate): gshydrate** gets the model onto the GPU faster. Median storage time (acquire + load): `globalstore` 36.6s (loading straight off the mount: 36.6s [35.5–36.8]) vs `gshydrate` 15.0s (hydrate 10.5s [10.0–11.4] + load from container disk 4.3s [3.2–4.5]) — globalstore is 2.44× slower. Client-observed cold request: globalstore 61.6s vs gshydrate 80.4s (medians; includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): nocache 32.2s < volume 43.3s < volumecache 59.8s < globalstore 61.6s < gshydrate 80.4s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate | ↳ Hub | load (tok+model) | **storage total** | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 32.2s [25.7–35.5] | 12.2s [9.3–12.4] | 19.9s [15.8–22.7] | 3.2s [1.9–3.2] | 8.06s [7.97–10.26] | — | 8.06s [7.97–10.26] | 3.99s [3.48–4.61] | **11.96s [11.54–14.88]** | 4.35s [2.08–4.45] | 3 |
| volume | 43.3s [42.1–45.1] | 13.0s [12.5–13.6] | 30.2s [27.9–32.1] | 3.2s [3.2–3.2] | 0.36s [0.30–0.45] | — | — | 20.87s [19.76–23.78] | **21.17s [20.11–24.23]** | 4.28s [4.19–4.33] | 3 |
| volumecache | 59.8s [45.7–86.4] | 28.2s [12.3–56.1] | 30.8s [29.8–33.2] | 3.2s [3.1–4.0] | 19.01s [18.11–19.93] | 17.79s [16.08–19.22] | 0.46s [0.45–0.46] | 4.16s [4.13–4.67] | **23.13s [22.27–24.60]** | 4.20s [4.20–4.28] | 3 |
| globalstore | 61.6s [52.7–63.9] | 20.1s [12.8–22.4] | 40.6s [39.7–41.0] | 2.0s [2.0–2.1] | 0.04s [0.04–0.05] | — | — | 36.60s [35.48–36.82] | **36.64s [35.53–36.86]** | 1.90s [1.77–1.95] | 3 |
| gshydrate | 80.4s [29.8–99.9] | 56.5s [11.2–76.6] | 22.8s [18.4–23.8] | 3.0s [1.9–3.1] | 10.95s [10.55–11.99] | 10.50s [10.02–11.40] | — | 4.32s [3.16–4.46] | **15.01s [14.11–16.30]** | 4.19s [2.10–4.44] | 3 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 4.4s [2.3–4.9] | 33.4 [32.9–72.0] | 3 |
| volume | 4.4s [4.3–5.0] | 33.0 [32.8–35.2] | 3 |
| volumecache | 4.5s [4.4–4.5] | 34.6 [34.0–35.1] | 3 |
| globalstore | 2.2s [2.2–2.2] | 83.6 [77.9–88.0] | 3 |
| gshydrate | 4.3s [2.2–4.3] | 34.2 [34.1–81.2] | 3 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 60.6s | 40.0s | 12.7s | download_s 12.7s |
| volume | 90.6s | 61.3s | 9.0s | download_s 8.7s |
| volumecache | 110.5s | 75.5s | 21.5s | hub_s 7.9s, hydrate_s 0.0s, sync_s 13.4s |
| globalstore | 180.1s | 49.4s | 66.0s | download_s 20.5s, upload_s 45.4s |
| gshydrate | 85.2s | 57.9s | 14.9s | hydrate_s 13.3s, resolve_s 0.0s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volumecache` needed the Hugging Face Hub on 3/3 cold starts (0.46s [0.45–0.46] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `gshydrate` waited 56.5s [11.2–76.6] for a worker on cold starts vs a median of 16.6s for the other strategies. That platform delay, not loading, dominates its cold start.

## Strategies

- **nocache** — no cache (HF download every cold start)
- **volume** — network volume, loaded in place
- **volumecache** — VolumeCache (volume -> container disk)
- **globalstore** — Runpod Global Store (global volume), loaded in place
- **gshydrate** — parallel range hydrate (Global Store -> container disk)

## Caveats

- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. Look at the [min–max] range before reading small differences.
- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.
- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.
- `GPU` names a class, not a machine: hosts in the same class differ (see the warm tokens/s range), which shows up in import, load and generate times alike.
- Cold rounds ran strategies concurrently in waves; strategies sharing storage (`volume`/`volumecache`, `globalstore`/`gsvolumecache`) were never in the same wave. (The prime phase runs all at once.)
- Model Store is declared with `model=runpod.Model(...)` on the endpoint (sets `modelReferences`).
- Global Store (beta) is declared as a `runpod.GlobalVolume` mount. Its endpoints are pinned to the same datacenter as the rest for comparability, although a global volume would let them run anywhere.
- Global Store is object storage behind a FUSE client (geesefs), which keeps its own memory cache that `posix_fadvise` cannot evict.
- `globalstore` keeps a plain directory of real files, not an HF cache: HF snapshot symlinks written by one worker come back as 0-byte regular files on the next worker's mount.
- `gsvolumecache` uses stock VolumeCache (runpod>=2.1.0 copies with copyfile; older releases' copy2 failed with EPERM on chmod/utime there and silently left every large file out of the mirror).

