# Model-loading strategies on Runpod Flash — run `qwen7b-gshydrate`

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (14.20 GiB of weights/tokenizer)
- **GPU:** NVIDIA GeForce RTX 4090 · **Datacenter:** EU-RO-1
- **Trials:** 3 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.7.0, huggingface_hub 2.2.0, runpod 1.10.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.19.0

## Answer

**Global Store (parallel hydrate): gshydrate** gets the model onto the GPU faster. Median storage time (acquire + load): `globalstore` 75.1s (loading straight off the mount: 74.9s [74.1–79.1]) vs `gshydrate` 28.4s (hydrate 22.6s [20.6–39.8] + load from container disk 7.8s [3.2–10.6]) — globalstore is 2.64× slower. Client-observed cold request: globalstore 148.9s vs gshydrate 62.3s (medians; includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): nocache 56.2s < gshydrate 62.3s < globalstore 148.9s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate | ↳ Hub | load (tok+model) | **storage total** | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 56.2s [42.8–112.2] | 17.5s [16.7–43.1] | 38.1s [25.8–68.3] | 5.2s [3.1–5.6] | 24.27s [15.16–53.48] | — | 24.27s [15.15–53.47] | 4.40s [4.16–4.45] | **28.68s [19.32–57.92]** | 3.74s [2.69–3.87] | 3 |
| globalstore | 148.9s [95.6–161.6] | 61.5s [14.7–74.8] | 86.3s [80.0–87.2] | 4.8s [3.0–7.1] | 0.04s [0.04–0.24] | — | — | 74.89s [74.12–79.09] | **75.13s [74.15–79.13]** | 2.97s [2.58–3.78] | 3 |
| gshydrate | 62.3s [47.7–201.2] | 24.0s [14.7–142.6] | 37.6s [32.2–58.1] | 4.7s [3.1–5.1] | 22.96s [20.60–39.81] | 22.59s [20.55–39.77] | — | 7.84s [3.22–10.62] | **28.44s [26.18–50.43]** | 2.75s [2.70–3.71] | 3 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 4.2s [3.7–4.8] | 43.8 [41.5–55.8] | 6 |
| globalstore | 3.7s [3.0–3.9] | 55.6 [54.5–56.1] | 6 |
| gshydrate | 3.7s [3.0–4.0] | 54.7 [41.3–56.0] | 6 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 149.5s | 90.8s | 45.7s | download_s 45.7s |
| globalstore | 227.9s | 33.9s | 78.4s | download_s 20.5s, upload_s 57.7s |
| gshydrate | 94.2s | 55.8s | 23.6s | hydrate_s 21.4s, resolve_s 0.0s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `globalstore` waited 61.5s [14.7–74.8] for a worker on cold starts vs a median of 20.7s for the other strategies. That platform delay, not loading, dominates its cold start.

## Strategies

- **nocache** — no cache (HF download every cold start)
- **globalstore** — Runpod Global Store (global volume), loaded in place
- **gshydrate** — parallel range hydrate (Global Store -> container disk)

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

