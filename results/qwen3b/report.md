# Model-loading strategies on Runpod Flash — run `qwen3b`

- **Model:** `Qwen/Qwen2.5-3B-Instruct` (5.76 GiB of weights/tokenizer)
- **GPU:** NVIDIA GeForce RTX 4090 · **Datacenter:** EU-RO-1
- **Trials:** 5 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.6.0, huggingface_hub 1.33.0, runpod 1.10.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.17.0

## Answer

**volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): volume 17.8s (loading straight off the mount: 17.5s [12.8–18.7]) vs VolumeCache 7.2s (hydrate 5.5s [4.7–20.4] + load from container disk 1.3s [1.3–2.5]) — volume is 2.47× slower. Client-observed cold request: volume 44.9s vs VolumeCache 27.9s (medians; this also includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): volumecache 27.9s < volume 44.9s < nocache 45.7s < modelstore 77.7s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate | ↳ Hub | load (tok+model) | **storage total** | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 45.7s [33.3–69.3] | 17.3s [9.5–22.4] | 28.6s [19.8–46.3] | 4.7s [2.9–5.3] | 16.40s [13.15–29.23] | — | 16.39s [13.15–29.23] | 2.51s [1.36–6.54] | **18.91s [14.55–35.78]** | 4.27s [2.07–4.90] | 5 |
| volume | 44.9s [28.4–146.4] | 16.4s [9.6–118.0] | 28.2s [18.2–28.9] | 4.6s [2.5–5.5] | 0.23s [0.13–0.28] | — | — | 17.49s [12.84–18.67] | **17.77s [12.97–18.90]** | 4.12s [1.76–4.17] | 5 |
| volumecache | 27.9s [21.4–41.0] | 12.6s [9.6–16.4] | 11.7s [11.1–27.9] | 2.7s [2.6–4.9] | 5.93s [5.29–21.38] | 5.46s [4.66–20.35] | 0.53s [0.41–0.93] | 1.29s [1.26–2.49] | **7.18s [6.56–22.73]** | 1.81s [1.70–4.33] | 5 |
| modelstore | 77.7s [44.1–116.6] | 69.9s [36.0–97.3] | 7.8s [7.2–19.2] | 2.7s [2.5–5.1] | 0.00s [0.00–0.00] | — | — | 2.65s [2.58–9.21] | **2.65s [2.58–9.22]** | 1.96s [1.70–4.25] | 5 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 4.4s [2.8–5.2] | 36.2 [32.4–89.2] | 10 |
| volume | 4.7s [2.3–5.2] | 36.8 [34.4–89.8] | 10 |
| volumecache | 2.9s [2.3–5.2] | 82.0 [34.4–94.2] | 10 |
| modelstore | 3.0s [2.3–5.1] | 78.8 [35.1–93.5] | 10 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 88.2s | 23.1s | 49.0s | download_s 49.0s |
| volume | 41.9s | 11.0s | 10.1s | download_s 9.8s |
| volumecache | 102.5s | 21.3s | 68.0s | hub_s 54.7s, hydrate_s 0.0s, sync_s 13.3s |
| modelstore | 116.1s | 98.1s | 0.0s | resolve_s 0.0s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volumecache` needed the Hugging Face Hub on 5/5 cold starts (0.53s [0.41–0.93] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.
- `modelstore` waited 69.9s [36.0–97.3] for a worker on cold starts vs a median of 16.4s for the other strategies. That platform delay, not loading, dominates its cold start — consistent with the documented behaviour of holding workers until a host that has the cached model is available (the delay is not billed as worker time).

## Strategies

- **nocache** — no cache (HF download every cold start)
- **volume** — network volume, loaded in place
- **volumecache** — VolumeCache (volume -> container disk)
- **modelstore** — Runpod Model Store (cached model)

## Caveats

- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. Look at ± spread before reading small differences.
- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.
- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.
- `GPU` names a class, not a machine: hosts in the same class differ (see the warm tokens/s range), which shows up in import, load and generate times alike.
- Cold rounds ran strategies concurrently in waves; `volume` and `volumecache` were never in the same wave, so they did not contend for the network volume. (The prime phase runs all four at once.)
- Model Store is attached with the console's GraphQL `modelReferences` field; Flash 1.20 has no parameter for it.

