# Model-loading strategies on Runpod Flash — run `h100-qwen7b-1111`

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (14.20 GiB of weights/tokenizer)
- **GPU:** NVIDIA H100 80GB HBM3 · **Datacenter:** US-CA-2
- **Trials:** 5 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.8.0, huggingface_hub 2.4.0, runpod 2.2.0, runpod_code 2.2.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.22.2

## Answer

**network volume: volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `volume` 26.6s (loading straight off the mount: 26.6s [23.3–38.2]) vs `volumecache` 8.6s (hydrate 2.6s [1.9–2.8] + load from container disk 5.4s [5.4–5.7]) — volume is 3.08× slower. Client-observed cold request: volume 104.8s vs volumecache 30.3s (medians; includes scheduler placement, which the strategy does not control).

**network volume (prefetched): volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): `volumeprefetch` 12.5s (loading straight off the mount: 6.1s [5.9–6.5]) vs `volumecache` 8.6s (hydrate 2.6s [1.9–2.8] + load from container disk 5.4s [5.4–5.7]) — volumeprefetch is 1.45× slower. Client-observed cold request: volumeprefetch 69.3s vs volumecache 30.3s (medians; includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): volumecache 30.3s < nocache 58.9s < volumeprefetch 69.3s < volume 104.8s < modelstore 107.0s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate/prefetch | ↳ Hub | load (tok+model) | **storage total** | shared-storage read | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 58.9s [51.5–174.4] | 13.5s [9.9–116.0] | 43.8s [40.2–58.1] | 2.1s [1.9–2.1] | 33.86s [30.60–47.70] | — | 33.85s [30.59–47.70] | 5.58s [5.53–6.37] | **39.75s [36.18–54.07]** | — | 1.85s [1.73–1.88] | 5 |
| volume | 104.8s [46.2–160.7] | 74.5s [9.5–132.4] | 31.7s [27.6–43.8] | 2.0s [1.9–2.2] | 0.02s [0.02–0.03] | — | — | 26.55s [23.28–38.17] | **26.57s [23.32–38.19]** | 554 MiB/s [386–633] | 1.95s [1.90–2.14] | 5 |
| volumecache | 30.3s [23.6–71.7] | 17.2s [11.2–58.8] | 12.7s [11.7–12.9] | 2.0s [2.0–2.2] | 3.16s [2.32–3.31] | 2.58s [1.91–2.77] | 0.31s [0.29–0.37] | 5.45s [5.42–5.70] | **8.62s [7.74–8.86]** | 5624 MiB/s [5257–7613] | 1.77s [1.73–1.89] | 5 |
| volumeprefetch | 69.3s [28.1–132.3] | 52.5s [11.8–116.5] | 16.4s [15.7–17.1] | 2.0s [1.9–2.2] | 6.02s [5.74–6.86] | 5.12s [4.92–6.35] | — | 6.05s [5.89–6.51] | **12.47s [11.63–12.85]** | 2840 MiB/s [2290–2954] | 1.76s [1.74–1.95] | 5 |
| modelstore | 107.0s [95.1–161.8] | 92.1s [80.3–146.8] | 14.4s [13.7–14.8] | 2.0s [1.9–2.0] | 0.00s [0.00–0.00] | — | — | 9.70s [9.21–10.06] | **9.70s [9.21–10.06]** | 1535 MiB/s [1484–1622] | 1.91s [1.85–2.18] | 5 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.
- *shared-storage read*: model bytes ÷ the step that reads the volume/store — the model load for in-place strategies (`volume`, `globalstore`, `modelstore`), the hydrate or prefetch otherwise.

## Overall cold request (what a client waits for)

| strategy | cold request e2e | platform delay | handler exec | best-case e2e | best-case delay | n |
|---|---|---|---|---|---|---|
| volumecache | 30.3s [23.6–71.7] | 17.2s [11.2–58.8] | 12.7s [11.7–12.9] | 23.6s | 11.2s | 5 |
| volumeprefetch | 69.3s [28.1–132.3] | 52.5s [11.8–116.5] | 16.4s [15.7–17.1] | 28.1s | 11.8s | 5 |
| volume | 104.8s [46.2–160.7] | 74.5s [9.5–132.4] | 31.7s [27.6–43.8] | 46.2s | 9.5s | 5 |
| nocache | 58.9s [51.5–174.4] | 13.5s [9.9–116.0] | 43.8s [40.2–58.1] | 51.5s | 9.9s | 5 |
| modelstore | 107.0s [95.1–161.8] | 92.1s [80.3–146.8] | 14.4s [13.7–14.8] | 95.1s | 80.3s | 5 |

Medians [min–max]; ranked by best case (minimum), which approximates a cold start without a GPU-capacity wait. Platform delay includes everything before the handler runs — queueing, worker start, and for `modelstore` the platform staging the model or placing the worker on a host that has it — so compare strategies here, not on storage time alone.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 2.2s [1.6–2.3] | 85.2 [80.6–91.6] | 10 |
| volume | 2.3s [1.6–2.3] | 88.6 [77.1–89.2] | 10 |
| volumecache | 2.2s [1.6–2.3] | 89.2 [73.8–91.6] | 10 |
| volumeprefetch | 1.6s [1.5–2.3] | 89.1 [77.7–91.5] | 10 |
| modelstore | 2.2s [1.6–2.3] | 88.1 [82.5–91.9] | 10 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 80.0s | 38.2s | 31.0s | download_s 31.0s |
| volume | 134.9s | 71.2s | 33.1s | download_s 33.1s |
| volumecache | 174.2s | 124.6s | 39.6s | hub_s 33.9s, hydrate_s 0.0s, sync_s 5.7s |
| volumeprefetch | 30.4s | 13.0s | 6.1s | prefetch_s 4.9s, resolve_s 0.0s |
| modelstore | 185.7s | 171.0s | 0.0s | resolve_s 0.0s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volumecache` needed the Hugging Face Hub on 5/5 cold starts (0.31s [0.29–0.37] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `volume` waited 74.5s [9.5–132.4] for a worker on cold starts vs a median of 34.8s for the other strategies. That platform delay, not loading, dominates its cold start.
- `modelstore` waited 92.1s [80.3–146.8] for a worker on cold starts vs a median of 34.8s for the other strategies. That platform delay, not loading, dominates its cold start — consistent with the documented behaviour of holding workers until a host that has the cached model is available (the delay is not billed as worker time).

## Strategies

- **nocache** — no cache (HF download every cold start)
- **volume** — network volume, loaded in place
- **volumecache** — VolumeCache (volume -> container disk)
- **volumeprefetch** — network volume, parallel page-cache prefetch, loaded in place
- **modelstore** — Runpod Model Store (cached model)

## Caveats

- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. Look at the [min–max] range before reading small differences.
- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.
- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.
- `GPU` names a class, not a machine: hosts in the same class differ (see the warm tokens/s range), which shows up in import, load and generate times alike.
- Cold rounds ran strategies concurrently in waves; strategies sharing storage (`volume`/`volumecache`/`volumeprefetch`, `globalstore`/`gsvolumecache`/`gshydrate`) were never in the same wave. (The prime phase runs all at once.)
- Model Store is declared with `model=runpod.Model(...)` on the endpoint (sets `modelReferences`).

## Time and cost

- wall clock (deploy → teardown): 33.1 min
- handler time (sum of `executionTime`): 14.6 GPU-min
- billed GPU time: 25.8 GPU-min, $2.32 (Runpod billing API, this run's endpoints only)
- billed by strategy: modelstore 4.5, nocache 7.0, volume 6.2, volumecache 4.1, volumeprefetch 3.9
