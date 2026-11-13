# Model-loading strategies on Runpod Flash — run `rp221-1113`

- **Model:** `Qwen/Qwen2.5-3B-Instruct` (5.76 GiB of weights/tokenizer)
- **GPU:** NVIDIA GeForce RTX 4090 · **Datacenter:** EU-RO-1
- **Trials:** 3 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.8.0, huggingface_hub 2.4.0, runpod 2.2.1, runpod_dist 2.2.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.22.2

## Answer

**network volume: volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `volume` 17.8s (loading straight off the mount: 17.5s [11.1–17.9]) vs `volumecache` 16.8s (hydrate 13.1s [12.8–20.7] + load from container disk 3.1s [2.8–5.3]) — volume is 1.06× slower. Client-observed cold request: volume 126.9s vs volumecache 29.6s (medians; includes scheduler placement, which the strategy does not control).

**network volume (VolumeCache 8 x 64 MiB): volumecache864** gets the model onto the GPU faster. Median storage time (acquire + load): `volume` 17.8s (loading straight off the mount: 17.5s [11.1–17.9]) vs `volumecache864` 17.3s (hydrate 12.7s [5.2–14.0] + load from container disk 3.9s [2.3–5.3]) — volume is 1.03× slower. Client-observed cold request: volume 126.9s vs volumecache864 38.5s (medians; includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): volumecache 29.6s < volumecache864 38.5s < nocache 39.1s < volume 126.9s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate/prefetch | ↳ Hub | load (tok+model) | **storage total** | shared-storage read | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 39.1s [22.2–44.9] | 12.9s [8.1–17.7] | 20.7s [14.0–31.3] | 3.1s [1.5–3.2] | 9.26s [8.22–19.67] | — | 9.25s [8.21–19.67] | 3.80s [2.22–3.92] | **13.18s [10.44–23.47]** | — | 4.14s [1.89–4.40] | 3 |
| volume | 126.9s [27.4–155.2] | 100.5s [10.9–129.2] | 25.5s [15.9–25.9] | 3.2s [2.5–3.2] | 0.29s [0.27–0.36] | — | — | 17.51s [11.06–17.88] | **17.79s [11.33–18.24]** | 348 MiB/s [340–547] | 4.16s [1.85–4.22] | 3 |
| volumecache | 29.6s [28.4–132.3] | 8.5s [7.5–98.9] | 20.4s [20.3–33.1] | 2.0s [1.5–2.4] | 13.67s [13.30–21.30] | 13.14s [12.78–20.67] | 0.45s [0.42–0.48] | 3.14s [2.85–5.28] | **16.81s [16.15–26.58]** | 449 MiB/s [285–462] | 2.01s [1.78–3.88] | 3 |
| volumecache864 | 38.5s [36.1–41.8] | 13.0s [12.5–23.8] | 24.9s [12.1–28.8] | 3.1s [2.0–4.2] | 13.35s [5.77–14.73] | 12.69s [5.22–13.97] | 0.46s [0.46–0.56] | 3.91s [2.27–5.26] | **17.26s [8.04–19.99]** | 465 MiB/s [422–1131] | 4.29s [1.92–4.42] | 3 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.
- *shared-storage read*: model bytes ÷ the step that reads the volume/store — the model load for in-place strategies (`volume`, `globalstore`, `modelstore`), the hydrate or prefetch otherwise.

## Overall cold request (what a client waits for)

| strategy | cold request e2e | platform delay | handler exec | best-case e2e | best-case delay | n |
|---|---|---|---|---|---|---|
| nocache | 39.1s [22.2–44.9] | 12.9s [8.1–17.7] | 20.7s [14.0–31.3] | 22.2s | 8.1s | 3 |
| volume | 126.9s [27.4–155.2] | 100.5s [10.9–129.2] | 25.5s [15.9–25.9] | 27.4s | 10.9s | 3 |
| volumecache | 29.6s [28.4–132.3] | 8.5s [7.5–98.9] | 20.4s [20.3–33.1] | 28.4s | 7.5s | 3 |
| volumecache864 | 38.5s [36.1–41.8] | 13.0s [12.5–23.8] | 24.9s [12.1–28.8] | 36.1s | 12.5s | 3 |

Medians [min–max]; ranked by best case (minimum), which approximates a cold start without a GPU-capacity wait. Platform delay includes everything before the handler runs — queueing, worker start, and for `modelstore` the platform staging the model or placing the worker on a host that has it — so compare strategies here, not on storage time alone.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 4.5s [2.2–4.8] | 35.0 [34.3–91.1] | 6 |
| volume | 4.2s [2.2–4.5] | 34.5 [33.9–87.3] | 6 |
| volumecache | 2.3s [2.3–4.3] | 82.7 [37.7–90.6] | 6 |
| volumecache864 | 4.3s [2.2–4.8] | 34.7 [33.2–86.1] | 6 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 77.0s | 57.7s | 12.6s | download_s 12.6s |
| volume | 64.3s | 16.4s | 21.8s | download_s 21.3s |
| volumecache | 28.6s | 10.0s | 11.5s | hub_s 6.9s, hydrate_s 0.0s, sync_s 4.5s |
| volumecache864 | 41.2s | 26.1s | 7.2s | hub_s 0.4s, hydrate_s 6.8s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volumecache` needed the Hugging Face Hub on 3/3 cold starts (0.45s [0.42–0.48] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `volumecache864` needed the Hugging Face Hub on 3/3 cold starts (0.46s [0.46–0.56] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `volume` waited 100.5s [10.9–129.2] for a worker on cold starts vs a median of 12.9s for the other strategies. That platform delay, not loading, dominates its cold start.

## Strategies

- **nocache** — no cache (HF download every cold start)
- **volume** — network volume, loaded in place
- **volumecache** — VolumeCache (volume -> container disk)
- **volumecache864** — VolumeCache with explicit 8 x 64 MiB hydrate (volume -> container disk)

## Caveats

- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. Look at the [min–max] range before reading small differences.
- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.
- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.
- `GPU` names a class, not a machine: hosts in the same class differ (see the warm tokens/s range), which shows up in import, load and generate times alike.
- Cold rounds ran strategies concurrently in waves; strategies sharing storage (`volume`/`volumecache`/`volumeprefetch`/`volumecache864`, `globalstore`/`gsvolumecache`/`gshydrate`) were never in the same wave. (The prime phase runs all at once.)
- Model Store is declared with `model=runpod.Model(...)` on the endpoint (sets `modelReferences`).

## Time and cost

- wall clock (deploy → teardown): 17.3 min
- handler time (sum of `executionTime`): 7.3 GPU-min
- billed GPU time: not in the billing API yet (`make billing RUN=<id>` later; billing can lag usage by several hours)
