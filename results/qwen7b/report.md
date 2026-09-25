# Model-loading strategies on Runpod Flash — run `qwen7b`

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (14.20 GiB of weights/tokenizer)
- **GPU:** NVIDIA GeForce RTX 4090 · **Datacenter:** EU-RO-1
- **Trials:** 3 forced-cold per strategy, 2 warm after each; strategies run concurrently in waves; page cache evicted before storage reads: yes
- **Versions:** accelerate 1.15.0, hf_xet 1.6.0, huggingface_hub 1.33.0, runpod 1.10.1, safetensors 0.8.0, torch 2.9.1+cu128, transformers 5.17.0

## Answer

**volumecache** gets the model onto the GPU faster. Median storage time (acquire + load): volume 37.9s (loading straight off the mount: 37.6s [35.9–37.9]) vs VolumeCache 15.5s (hydrate 10.2s [8.2–10.3] + load from container disk 4.6s [2.3–4.7]) — volume is 2.44× slower. Client-observed cold request: volume 78.9s vs VolumeCache 49.6s (medians; this also includes scheduler placement, which the strategy does not control).

Cold-request ranking (client-observed, median): volumecache 49.6s < modelstore 58.1s < nocache 58.2s < volume 78.9s

## Cold start (forced scale-to-zero, first request on a new worker)

| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate | ↳ Hub | load (tok+model) | **storage total** | first generate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nocache | 58.2s [32.9–58.7] | 16.3s [9.6–19.7] | 38.0s [22.6–41.6] | 4.9s [2.6–5.5] | 19.65s [14.57–24.09] | — | 19.65s [14.57–24.09] | 4.15s [2.39–13.09] | **28.25s [16.96–32.74]** | 3.65s [2.77–3.98] | 3 |
| volume | 78.9s [69.2–80.9] | 31.0s [18.0–34.1] | 47.2s [46.2–50.6] | 5.2s [4.8–6.0] | 0.27s [0.26–0.38] | — | — | 37.60s [35.91–37.89] | **37.88s [36.29–38.16]** | 3.59s [3.52–3.75] | 3 |
| volumecache | 49.6s [28.3–50.6] | 21.3s [11.1–25.5] | 24.7s [16.4–27.8] | 5.0s [2.6–5.0] | 10.93s [8.67–14.25] | 10.16s [8.16–10.30] | 0.59s [0.42–0.65] | 4.56s [2.33–4.65] | **15.49s [11.00–18.91]** | 3.53s [2.57–3.93] | 3 |
| modelstore | 58.1s [20.9–151.0] | 34.3s [9.5–115.0] | 23.1s [11.1–35.5] | 5.1s [2.5–5.5] | 0.00s [0.00–0.00] | — | — | 14.13s [5.74–25.15] | **14.13s [5.74–25.15]** | 3.67s [2.57–4.05] | 3 |

Cells are **median [min–max]** over valid cold trials.

- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own un-pause window).
- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. Model Store's staging/placement shows up here, not in the handler.
- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.

## Warm requests (model already in GPU memory)

| strategy | warm request e2e | tokens/s | n |
|---|---|---|---|
| nocache | 4.1s [3.0–4.5] | 44.2 [39.0–56.2] | 6 |
| volume | 4.0s [3.7–4.4] | 44.2 [42.5–45.0] | 6 |
| volumecache | 4.3s [3.0–5.0] | 43.0 [40.2–56.0] | 6 |
| modelstore | 3.8s [3.7–4.5] | 44.1 [41.4–55.9] | 6 |

Warm throughput should be identical across strategies — once weights are in VRAM, storage is out of the picture. A gap here points at GPU/host variance, not the caching strategy.

## First-ever request (cache population)

| strategy | e2e | platform delay | acquire | detail |
|---|---|---|---|---|
| nocache | 75.8s | 32.4s | 20.5s | download_s 20.5s |
| volume | 118.9s | 44.4s | 19.4s | download_s 19.1s |
| volumecache | 80.7s | 23.5s | 43.0s | hub_s 28.3s, hydrate_s 0.0s, sync_s 14.7s |
| modelstore | 62.6s | 50.9s | 0.0s | resolve_s 0.0s |

One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).

## Observations

- `volumecache` needed the Hugging Face Hub on 3/3 cold starts (0.59s [0.42–0.65] each). VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). A Hub outage would therefore fall back to a full download.

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

