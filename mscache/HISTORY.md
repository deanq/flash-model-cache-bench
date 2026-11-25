<!-- Verbatim archive of ~/.claude/roles/topics/MSCACHE.md as of 2026-11-24, before it was trimmed to current state. -->

# MSCACHE: Model Store cache test program, customer path, every enabled DC

This file is the topic's memory. A session opened for this topic starts with
no conversation history and reads this file first. Keep it current: write every
decision, open item and next trigger here in the same turn it happens.

Opened 2026-11-21 by DQ on Dean's choice. It takes over the Model Store
peer-cache / LRU testing that the SDK topic ran as waves 1-3 plus TX-3/MO-1.
After reading this file, do not start any paid run. First step: design and
build (free, local), then report to DQ.

## Owns
- The Model Store cache test program described below: scenario harness, runs, per-run reports.
- Code: ~/Desktop/flash-model-cache-bench (git, main) and the probe at ~/Desktop/handoff-bundles/tx3-modelstore-peer-probe (cycle.py, redeploy.py, create.py, probe_handler.py, rounds.py, saturate.py, set_model.py; logs rounds*.log, results.jsonl). Decide whether the new harness lives in the bench repo (recommended: a new directory there, committed on a branch) and record it here.
- Not owned: runpod-python SDK work (SDK topic). Not owned: any Runpod platform fix or ticket (see Rules).

## Rules that hold here
- THIS MACHINE ACTS AS A RUNPOD USER, NOT AN EMPLOYEE (Dean, 2026-11-21). Customer path only: public API, SDK, console-equivalent calls with Dean's user account. No internal repos, no tickets, no platform fixes. Platform findings go to DQ, which sends them to Dean's main-Mac session.
- Paid runs start only on Dean's go typed in THIS session (`claude attach MSCACHE`). A go relayed through DQ or another session is not enough; SDK's permission check denied exactly that on 2026-11-21. Say the cost before asking.
- Stop and report if a DC gives no workers for 15 minutes.
- Clean up every endpoint after each scenario; verify nothing is left.
- Keep runs inside 3-4 h (see prototype-benchmark-rigor.md).
- No em-dashes and no Claude attribution in anything that leaves this machine.

## The strategy (from Dean via his main-Mac DQ, 2026-11-21, with MS's update merged)

GOAL: measure how the Model Store cache really performs for a customer and find its gaps, in every DC where it is enabled.

LESSONS FROM WAVES 1-3 (design around them):
- Cached hits were confounded by other pods (incl. other customers') holding the model ("Cannot unassign - in use").
- LRU keeps only copies the DB marks DEPLOYED; a round that ends before the host reports DEPLOYED leaves the copy ASSIGNED and teardown deletes it. Round length is a variable.
- Peer pulls ran slower than origin in CA-MTL-1 (p50 39 vs 216 MB/s) and share 4 download slots per host.
- The 40-60 s scale-from-zero wait is ai-api (40 s min queue time + 20 s cache reload; fixes in review), not the cache.
- In-container timing alone cannot tell peer from origin; the main Mac joins host logs.

PRINCIPLES:
1. One variable per scenario.
2. Measure phases: submit -> pod resumed -> container start -> first model file write -> last file write -> handler ready -> first job done.
3. Every load joinable: endpoint id, pod id, machine id, model repo + revision + size, UTC timestamps per phase.
4. A control DC with LRU off.
5. At least 3 repeats per scenario per DC; report median and worst.
6. Clean up every endpoint after each scenario.

DC MATRIX (MS, 2026-11-21):
- Model Store for serverless is on for every account in every DC; no flag needed.
- LRU ON: US-TX-3 (26 machines), US-MO-1 (56), CA-MTL-1 (~236), EU-SE-1 (171), US-NC-1 (104). Skip CEPH-POC and VAST-POC (not reachable). Everything else is LRU OFF.
- CONTROL (LRU off): EUR-NO-1 (118 machines, no peer). Alternatives US-NE-1 or EUR-IS-2. Avoid US-CA-2 and EUR-IS-1 (~30% CPU machines). Disk-cap eviction still runs in LRU-off DCs; only "keep the copy after the last pod" is LRU.
- Peer pull live (partly): US-TX-3, CA-MTL-1, US-MO-1 (7 MO-1 machines serve but do not pull, incl. 7b03). EUR-IS-2 is NOT live. Run S4/S5 in those three only.
- GPU types: read live stock per DC from the capacity API at test time. TX-3 had no L40S or 4090-only capacity on 11-21. "No worker" outcomes are findings too.

SCENARIOS:
- S1 Cold first pull: a model never seen in the DC.
- S2 Warm reuse after teardown (LRU core): deploy, wait for handler ready + 2 min so the copy is fully on disk, scale to 0, wait until no pods, redeploy; log whether it lands on the same machine. Idle gaps 1, 10 and 60 min to find the retention horizon.
- S3 Short-round teardown: like S2 but tear down 10 s, 30 s, 60 s after container start. Corrected expectation: teardown mid-download does NOT cancel the download; the redeploy waits on it. Measure time to ready on the redeploy and whether the same bytes are pulled twice (one case on 11-21 lost ~18 min).
- S4 Peer vs origin: model cached on other machines in the DC but not the target. Peer applies only to public HF content: add one private model as a negative control (always origin).
- S5 Slot contention: 3-6 endpoints with different large models landing on one machine at once. S5+: one warm deploy (model already on disk) while the same machine is cold-pulling 2+ other models; warm files still wait for one of 4 download slots (queue wait p50 75-102 s, p95 306-373 s that week in TX-3/MO-1).
- S6 Disk pressure: the cap is 20% of disk, model bytes only; at the cap a deploy goes to another machine rather than evicting in place; eviction takes the oldest copy with no minimum age, so a just-finished copy can be evicted. Repeat the 11-21 7vmo Devstral case (rewritten ~10 min after first pull). Run in one large and one small DC.
- S7 Size spread: one small (<10 GB), one medium (20-40 GB), one large (100+ GB) model per DC.
- S8 Failure paths: an unreadable gated/private repo and a non-existent revision; time to failure and the error shown. Once.
- S9 Scale-from-zero isolation: already-cached model, endpoint at 0 workers; everything after "pod resumed" is cache time. Keep the ai-api wait out of cache numbers.
- S10 Teardown then immediate redeploy on the same machine: does the new worker start cleanly, and how long does it take.

COVERAGE: per LRU-on DC at least S1, S2, S3, S7, S9 (and S10); S4 and S5 in TX-3, CA-MTL-1, MO-1; S6 in one large and one small DC; S8 once; the control DC for the S2/S9 comparison.

OUTPUT PER RUN (send to DQ, which forwards to the main Mac): window, cost estimate, DC, scenario id, round; per load: endpoint id, pod id, machine id, model + revision + size, phase timestamps, outcome (cached / partial / full pull / failed / no worker); anomalies labeled as in-container observations. The main Mac joins host and graphql logs (peer vs origin per file on about half of downloads, download-slot wait, version completed/failed per machine; it cannot see free disk per machine) and sends back the combined read.

CADENCE: pilot S2 + S9 in one DC first to check the joins, then the full matrix.

## Open pull requests
- None.

## Flags, rollouts and watches
- None held here. LRU flag state is the main Mac's; ask DQ if unsure.

## Plan (written 2026-11-21)
HARNESS: ~/Desktop/flash-model-cache-bench, dir mscache/, branch feature/mscache-harness, commit ce85719 (local only: push was denied by the permission classifier; Dean decides). Worktree path in "Worktrees". Tests: tests/test_mscache.py (19, fake Runpod, no spend); full suite 93 pass.
- spec.py: scenarios are data (Deploy(role, hold_s, scale_up) / Teardown / AwaitScaleDown / Idle). One endpoint, one DC, one narrow GPU pool, ONE worker per scenario. s2(gap_s) = seed + 3x[teardown, gap, redeploy], hold READY_HOLD_S=120 on every load (ready + 2 min). s9 = seed + 3x[idle scale-down, 60 s, submit with workersMax left at 1].
- driver.py: runs steps, endpoint deleted in finally; no_worker (300 s no pod) / stall (600 s all exited) / timeout (30 min) end the scenario. Gap is measured from teardown_done (all pods EXITED/TERMINATED). Pods placed for a load but not serving it are written as pod_only taint records.
- phases.py: one JSON record per load: endpoint, pod, machine, gpu, model, model_ref (resolved commit from saveEndpoint), revision (snapshot dir), staged_bytes, UTC phases submit/pod_created/pod_last_started/container_start/handler_ready/job_start/job_done_seen, offsets, staging_s = container_start - pod_created, outcome estimate (cached <= 75 s, else pull; no_worker; failed). Host-log join is final.
- cost.py, plan.py, report.py, __main__.py: `python -m mscache plan|smoke|run --spend|report|leftovers` (run from the worktree with ../../../.venv/bin/python; output runs/mscache/<run_id>/records.jsonl + report.md). Endpoint names msc-<dc>-<sid><arm>-<gpu>-<run_id>; leftover check filters "msc-".
- $0 smoke passed 2026-11-21 in CA-MTL-1/a40: 3 endpoints created+deleted, refs resolved to commits, leftovers none.

PILOT (awaiting Dean's go): CA-MTL-1, A40 pool ($1.22/h), 3 scenarios in parallel on 3 endpoints, distinct low-traffic models so arms cannot taint each other:
- S2 gap 60 s: allenai/OLMo-2-1124-13B-Instruct (27.4 GB)
- S2 gap 600 s: tiiuae/Falcon3-10B-Instruct (20.6 GB)
- S9: ibm-granite/granite-3.3-8b-instruct (16.3 GB)
- 12 loads; likely $0.67, worst $2.22; wall ~45 min. Seeds double as S1 samples (likely first pull in DC). Calibration: 11-21 billing was ~$0.05/load across ~200 loads.
- Command: cd <worktree>; ../../../.venv/bin/python -m mscache run --dc CA-MTL-1 --gpu a40 --spend
- Pilot success = the main Mac can join every load (endpoint/pod/machine/UTC phases) to host logs; report goes to DQ.

FULL MATRIX (rough, likely; worst ~3x): per LRU-on DC S1+S2(gaps 1/10/60 min)+S3+S7+S9+S10 ~60 loads, ~4 h worker-time: MTL A40 ~$5, SE-1 A40 ~$5, NC-1 L40S ~$8, TX-3 L40S/4090 ~$7, MO-1 A100 ~$13 (H100 ~$25). S4+S5 in TX-3/MTL/MO-1 ~$10-15. Control S2/S9 ~$2. S8 <$1. Total ~$50-60 likely, ~$170 worst. The gap-60-min arm sets wall time to ~3.3 h per DC; split the matrix into 2 windows of <=4 h. Scenarios S1/S3/S4/S5/S7/S8/S10 not coded yet (after the pilot validates the joins).

SPEC CORRECTIONS TO RAISE (need Dean or main Mac):
1. S3 "teardown 10/30/60 s after container start" cannot hit mid-download: Model Store stages the copy BEFORE the container starts. Proposal: anchor on pod_created.
2. "first/last model file write" is not observable in-container (blobs keep upstream mtimes; copy is mounted before start). These phases must come from host logs.
3. S6 disk pressure cannot be targeted from the customer side (no machine pinning, no free-disk view). Proposal: observational only, or drop.
4. Control DC EUR-NO-1 lists no GPU stock in the catalog right now. Fallback EUR-IS-2 (4090/L4, LRU off, peer not live) or US-NE-1; check at test time with a $0 abandon.
5. Parallel scenarios in one DC can co-land on one machine and share its 4 download slots. The harness records machine per load so co-landing is visible; serialize if it shows up.

## Next trigger
- Pilot done 2026-11-21 18:46; main Mac join received; fresh-endpoint change committed bab67f4 and reported to DQ.
- 2026-11-21: Dean replied "ok" to the sitrep. Read as: go ahead with the free work, NOT a paid go (no run named, no cost accepted).
- Done: main Mac's 64q5 check, commit 84880c0. staging_s is anchored on pod_created by design (placement wait excluded); records now carry staging_from and submit_to_container_s, and the report labels both (64q5: 23.4 s from pod_created, 48.3 s from submit). 103 passed, 1 skipped.
- Done 2026-11-22: full matrix coded, commit 7a25262 (branch feature/mscache-harness, NOT pushed; 128 passed, 1 skipped).
  - Shape: plan = lanes (parallel) -> batches (serial) -> scenarios (parallel). Window 1 = S2gap3600 lane beside a serial core (S2gap60, S2gap600, S9, S10, S1, S7 small/medium/large, S3 t10/t30/t60). Window 2 = contention lane (S4 holder+3 peers, then S5 warm redeploy during 3 cold 72B pulls) in TX-3/MTL/MO-1 only, plus S8 (gated, bad revision) in CA-MTL-1 only. Control = serial S2gap60, S2gap600, S9 in EUR-IS-2 (fallback US-NE-1).
  - One model per scenario, same models in every DC, none from the pilot (map in mscache/plan.py `M`).
  - S3 cuts at max(pod_created, submit) + 10/30/60 s and records whether the container had already started (then it is an S2 round, not S3). S4/S5 arms line up via in-run signals (set when the job goes IN_PROGRESS).
  - S6 not coded: observational only (spec correction #3).
  - S4 private-model negative control NOT coded: needs a private HF repo + token from Dean.
  - $0 smoke CA-MTL-1 w1+w2 2026-11-22: all 20 refs resolve; S8 gated and bad revision are both REJECTED AT CREATE ("Failed to validate model reference ... Unable to access model"), so S8 costs $0 and is a create-time result. Leftovers none.
  - Plan costs (likely / worst): W1 per DC: MTL a40 $2.30/$8.69 (195 min, set by S2gap3600), SE-1 a40 same, NC-1 l40s $3.56/$13.46, TX-3 l40s $3.56/$13.46 (4090: $2.52/$9.54), MO-1 a100 $5.63/$21.29. W2: MTL $1.20/$6.23 (18 min), TX-3 l40s $1.82/$9.59 (4090 $1.29/$6.80), MO-1 $2.88/$15.17. Control EUR-IS-2 4090 $0.71/$1.60 (74 min). Everything, TX-3 on l40s: about $25 likely / $98 worst. DCs run in parallel; W1 then W2 is about 3.5 h wall.
- 2026-11-22: Dean typed "go" HERE, right after the matrix summary that quoted "everything: ~$25 likely / $98 worst, ~3.5 h". Read as a go for exactly that: W1 in the 5 LRU DCs (MTL a40, SE-1 a40, NC-1 l40s, TX-3 l40s, MO-1 a100) + control EUR-IS-2 4090, all in parallel; then W2 in MTL/TX-3/MO-1 once that DC's W1 ends. No GPU swaps or extra arms outside this without asking. If a DC's first load gets no worker (15 min), kill that DC's run and report. Wait on PIDs, not pgrep -f (the pilot's pgrep loops matched themselves and never exited).
- Matrix run log (launched 07:50 local; logs ~/.claude/jobs/00ee7a35/tmp/matrix, PIDs in matrix/pids):
  - 08:24: S2 seeds done in all 6. US-NC-1 S2gap60 measure THROTTLED 15 min (both NC-1 seeds landed on machine ae3z; 60 s later nothing placed). Seeds placed, so not a "DC gives no workers" stop; NC-1 goes on. If NC-1 S2gap600 also throttles, kill NC-1 and report. US-TX-3 gap3600 seed: pod +183 s, container +1338 s (~19 min to stage 16 GB).
  - 08:31: US-TX-3 S2gap60 measure also THROTTLED 15 min (seed on 623z). Both throttles are l40s DCs (TX-3 l40s also throttled in the pilot). Same rule for TX-3: if its S2gap600 throttles too, kill it and report. The other 4 DCs are fine: cached re-deploys stage in 3.8-6.2 s, or about 45-65 s behind the ai-api floor. EUR-IS-2 (LRU off) labels some loads "cached" (52 s); probably the ai-api floor, so the main Mac's host-log join decides.
  - 08:43: US-NC-1 STOPPED (2nd throttle: S2gap600 seed placed 3 pods on 3 machines, then throttled 15 min). Killed PID 22042/22041; deleted its 2 live endpoints (S2gap3600 mid-gap, S9 just created); no nc1 endpoints left. Partial report runs/mscache/11221550-US-NC-1-l40s-w1/report.md. NC-1 has 2 seeds (pull, ae3z) and nothing else. Rerun on another GPU only with Dean's go.
  - 08:52: US-TX-3 STOPPED (2nd throttle: S2gap600 seed, pod on 358f, throttled 15 min). Killed PID 22044/22043 (W2 in TX-3 will not run); deleted its 2 live endpoints; no tx3 left. Partial report runs/mscache/11221550-US-TX-3-l40s-w1/report.md: 2 seed pulls (623z 630 s, hyxm 1158 s staging). Both l40s DCs are out; l40s capacity, not the cache. Rerun on another GPU only with Dean's go.
  - ~09:2x: EUR-IS-2 control DONE, all 3 ok; report runs/mscache/11221550-EUR-IS-2-4090-control/report.md. Its exit 1 is the global leftovers check seeing the other DCs' live endpoints (none euris2). Harness nit: scope leftovers to the run's DC.
  - 09:12: S2 60-min measures. MTL: cached, but on 73y4 while the seed was on i96l (another machine had it; check peer/LRU in the host join). SE-1: cached, same machine (b1l6). MO-1: THROTTLED 15 min (seed on 7b03): MO-1's 1st throttle; a 2nd stops MO-1.
  - 09:17: US-MO-1 STOPPED (2nd: S2gap600 measure no_worker 15 min, after the 60-min throttle). Killed PID 22046/22045 (MO-1 W2 will not run); deleted S9 endpoint; no mo1 left. Partial report runs/mscache/11221550-US-MO-1-a100-w1/report.md: S2gap60 complete (seed + 3 measures), gap600 seed pull, 60-min seed pull. Now only MTL (W1 then W2 = only S4/S5 + S8 DC) and SE-1 run.
  - 11:04 MTL W1 12/12 ok; ~11:10 SE-1 W1 12/12 ok (exit 1 = global leftovers nit). MTL W2: S8 gated + badrev rejected at create ($0); S4 done 11:21 (peer3 pulled, ~530 s); S5 started 11:21. Stop S5 at 11:50 (4 h cap) if not done; Dean did NOT extend the cap.
  - ~11:30: Dean typed "go" again; asked to scope (AskUserQuestion), Dean picked "Rerun dropped DCs": US-TX-3 4090 W1+W2 and US-NC-1 a40 W1, about $6.11 likely / $25.03 worst, ~3.5 h. US-MO-1 stays out. Launched via tmp/rerun.sh; logs matrix/US-TX-3-4090-*.out, US-NC-1-a40-w1.out; PIDs appended to matrix/pids. Same 2-throttle stop rule.
  - 11:4x: MTL W2 DONE inside the cap (run dir runs/mscache/11221904-CA-MTL-1-a40-w2; original matrix.sh exited 0). S4: holder seed pulled on i96l (173 s); all 3 peers co-landed on ANOTHER machine (90vb) and each pulled ~527-531 s, so no peer copy from the holder was seen. S5: warm redeploy cached 3.8 s on h3ny, but the 3 cold 72B pulls all co-landed on tm9s (363 / 603 / 848 s staging, ~245 s apart: they look queued, not parallel). Warm and cold were on different machines, so S5 did NOT test a warm load behind busy download slots. Both are for the main Mac's host-log join. Original-run leftovers: none (only the rerun's live endpoints).
  - 11:52: Rerun STOPPED. US-TX-3 4090 and US-NC-1 a40: every first load (S2gap60, S2gap3600 in both) got NO WORKER for 15 min, so the "DC gives no workers for 15 min" rule applies. Killed 50370-50373, deleted the 2 live endpoints; leftovers none (verified). Rerun spend about $0 (nothing placed). Reports: runs/mscache/11221937-US-TX-3-4090-w1, 11221937-US-NC-1-a40-w1 (no loads).
- MATRIX DONE 2026-11-22 11:53. Full coverage: CA-MTL-1 (W1 12/12 + W2), EU-SE-1 (W1 12/12), EUR-IS-2 control (3/3). Partial: US-MO-1 (S2 60 s complete + 2 seeds). Seeds only: US-NC-1, US-TX-3. Capacity on 11-22 was the blocker in 3 of 5 LRU DCs (l40s throttled, a100 throttled/no worker, 4090 and a40 no worker).
- BILLING DONE ~13:00 (MCP list-serverless-billing, rows summed by the endpoint_id values in each run's records.jsonl). Matrix actual $4.19 vs quoted $25 likely / $98 worst: CA-MTL-1 W1 $1.16, EU-SE-1 $1.09, US-MO-1 $0.98, EUR-IS-2 control $0.63, US-NC-1 l40s $0.17, US-TX-3 l40s $0.15, CA-MTL-1 W2 $0.00 at the time (rows lagging; see BILLING FINAL below), rerun $0.00. Pilot (11220157) $1.47. 11-22 day total $5.66 = matrix + pilot, so it reconciles.
- BILLING FINAL (2026-11-23, 19:00Z hour rows posted): CA-MTL-1 W2 $0.53 (9 endpoints: S5 cold 2gai $0.11, the rest ~$0.05 each). Two more 19:00Z rows belong to W1 (7bay, gn4y; MTL/SE-1 S2gap3600) for +$0.10. Rerun endpoints have no rows ($0). MATRIX TOTAL $4.82 vs quoted $25 likely / $98 worst; with the pilot ($1.47), 11-22 is $6.29.
- (superseded) Next: (1) billing recheck; (2) host-log join; (3) Dean push/retry/HF repo; (4) leftovers nit.
- PARKED 2026-11-23 on Dean's "park MSCACHE" (via DQ). State: nothing in flight, no crons or monitors held, no mscache processes, leftovers none (verified 11-22 11:52). Main Mac's combined read is in the Inbox (2026-11-22 entry): LRU confirmed; MTL gap3600 hits were local, not peer; 73y4 16:54 was an origin pull misread as cached.
- OPEN ITEMS:
  1. Dean: push feature/mscache-harness (commits bab67f4, 84880c0, 7a25262; not on the remote): `! git -C ~/Desktop/flash-model-cache-bench/.claude/worktrees/mscache-harness push -u origin feature/mscache-harness`
  2. Retry US-NC-1, US-TX-3 and US-MO-1 once the main Mac's ai-api floor fix and teardown (in-flight drop) fix ship. Add an S3 rerun to verify those fixes.
  3. Harness: the 75 s cached/pull cutoff misclassified 73y4 16:54 (a ~15 GB origin pull in 24 s inside the 74.5 s floor load). Reclassify from the host join where available; don't treat the cutoff as final.
  4. Optional: a private HF repo for S4's negative control.
  5. Harness nit: scope the end-of-run leftovers check to the run's DC.
- NEXT TRIGGER: DQ or the main Mac reports that the ai-api floor and teardown fixes have shipped. Then quote the cost for NC-1/TX-3/MO-1 W1(+W2) plus an S3 rerun, and wait for Dean's go typed in this session (`claude attach MSCACHE`). Fix item 3 (and 5) before that run.
- 2026-11-24: session reopened by DQ; read this file. Inbox 11-24: host v4.9.0 GA (CA-MTL-1 233/233, US-MO-1 56/56); RunPod#7549 not yet in prod. NEXT: (a) free: harness items 3 (reclassify cached/pull from host join, not the 75 s cutoff) and 5 (scope leftovers to run DC); (b) quote v4.9.0 after-run for S4/S5 (CA-MTL-1, US-MO-1; warm+cold must co-land) + S2 gap600 + EUR-IS-2 control, which don't need #7549; S3/S10 wait for #7549 in prod. Paid run only on Dean's go typed here.
- (superseded) Next: Dean's go typed HERE, naming what to run (all, or by DC/window) and accepting the cost; push of feature/mscache-harness; a private HF repo for S4's negative control (optional). Pilot-gap retry ($1.27/$4.13) is superseded by window 1, which reruns S2 60/600 + S9 on new models.

## Quote 2026-11-24: v4.3.0 -> v4.9.0 after-run (asked by DQ; quote only, nothing run)
From `python -m mscache plan` at 7a25262 ($0). Same models/arms as 11-22. 11-22 actuals came in ~5x under "likely" ($4.82 vs $25), so "likely" is conservative.
- (2) S4/S5 (no S8; S8 is a create-time reject, nothing to re-measure): CA-MTL-1 a40 $1.17 / $6.19, 18 min. US-MO-1 l40s $1.82 / $9.59, 18 min (a100 would be $2.88 / $15.17 but MO-1 lists no A100). Note: MO-1 W2 never ran 11-22, so MO-1 S4/S5 has no "before"; only MTL is a true before/after.
- (3) S2gap600: CA-MTL-1 a40 $0.25 / $0.59, 45 min. Control EUR-IS-2 4090 S2gap600 $0.28 / $0.65, 45 min (full control S2gap60+600+S9 $0.71 / $1.60, 74 min).
- (1) S3 t10/t30/t60 + S10: CA-MTL-1 a40 $0.80 / $3.37, 43 min; US-MO-1 l40s $1.24 / $5.22, 43 min (a100 $1.96 / $8.25).
- Recommended now (2)+(3): $3.52 likely / $17.02 worst. Wall ~65 min (MTL serial: S4/S5 then S2gap600; MO-1 and IS-2 in parallel). Adding (1) host-only now: +$2.04 / +$8.59, +43 min (total $5.56 / $25.61, ~1h50).
- $0 capacity read (get-data-center GPU_AVAILABILITY, pod catalog, 2026-11-24): MO-1 L40S LOW + H100 LOW, NO A100; TX-3 nothing listed; NC-1 only H200 LOW; MTL A40 LOW; IS-2 4090 LOW. Verdict: MO-1 A100 no (use L40S), TX-3 skip, NC-1 skip. Catalog is pod stock, serverless may differ; the 15-min no-worker stop rule is the real $0 check.
- 2026-11-24: quote sent to DQ. NEXT TRIGGER: Dean's go typed here for (2)+(3) [and optionally (1)]; or DQ asks for fixes 3/5/6 first ($0). #7549 in prod unblocks (1) as the main target.
- Harness fixes before the run (free, not yet done): (3) cached/pull outcome from the 75 s cutoff is an estimate only; label it so and let the host join decide. (5) scope end-of-run leftovers to the run's DC. (6) S5 co-landing: warm and cold must share a machine (11-22 they did not). Customer path can't pin machines; plan: start cold loads first, then redeploy warm until it lands on a cold load's machine (max 3 tries), record co_landed; adds ~$0.2-0.5 likely. Without it S5 repeats 11-22's non-result.

## Pilot run log (done)
- 2026-11-21: Dean, typed in this session: "For each of the pilot DC, run a sub agent to execute the tests". Asked which DCs; Dean picked the 3 peer-live DCs. Stated cost: likely $3.35, worst $11.10, ~45 min.
- One subagent per DC, in parallel: CA-MTL-1 a40 ($0.67/$2.22), US-MO-1 a100 ($1.64/$5.44), US-TX-3 l40s ($1.04/$3.44; if every scenario gets no worker, retry once with 4090, $0.74/$2.44). No H100 fallback in MO-1 without asking (outside the stated cost).
- Output dirs: <worktree>/runs/mscache/<MMDDHHMM>-<DC>-<gpu>/ (records.jsonl, report.md).
- 3 background subagents launched 2026-11-21 (commit c3c90ed: per-DC run dir fix). Each checks leftovers for its own prefix (msc-camtl1-, msc-mo1-, msc-tx3-). Logs: $CLAUDE_JOB_DIR/tmp/mscache-pilot-<DC>*.out.
- Next: when all 3 report, verify leftovers none for msc-*, write results here, send DQ the summary + report paths for the main Mac's join.
- Launched 17:57 local (run_id 11220157). The MTL and TX-3 subagents ended early, before their runs finished, so they never reported. MSCACHE watches the logs itself now (Monitor on the 3 .out files).
- 18:03 live findings: (a) all 3 scenarios in each DC landed on ONE machine (MTL vys4, MO-1 7b03, TX-3 623z); this is spec correction #5, so reruns go --serial. (b) A pod that is staging a model, or throttled waiting for a GPU, is listed desiredStatus EXITED; only /health shows it alive. The old stall rule therefore cancels live loads at 10 min.
- 18:05: gap60 seed loads finished as pulls (MTL ~318 s staging, MO-1 ~385 s). TX-3: all 3 throttled, no load yet.
- Fix c496049: /health liveness, a 900 s throttled limit (the program's 15-min no-worker rule), and run --only/--serial. The running processes still use the old code.
- 18:07-18:09: all MTL and MO-1 seeds staged as pulls (MTL 318/441/450 s, MO-1 385/571/574 s), just under the old stall cutoff. S9 measure loads were logged "unknown": scale-from-zero restarts the stopped worker under its OLD pod id, which the stale filter dropped. Fixed in f3b64d8 (join the stale pod, anchor staging on max(createdAt, submit), record pod_reused). Re-read the pilot's unknown S9 rows: container_start about 2 s after submit means cached.
- 18:10 TX-3: S2gap600 placed on hyxm after 470 s and pulled in ~165 s. The old stall rule failed S2gap60 and S9 (throttled about 800 s). Their endpoints are deleted.
- 18:11: TX-3 rerun started on the fixed code: `--serial --only S2gap60,S9` (likely $0.63, worst $2.28, inside the TX-3 budget). Log: $CLAUDE_JOB_DIR/tmp/mscache-rerun-US-TX-3.out.
- 18:16: rerun 1 abandoned S2gap60 as no_worker at 300 s (TX-3 placement takes ~470 s). Killed it and deleted its S9 endpoint by hand. Lesson: SIGINT does not stop a run, because the worker thread keeps going; use SIGTERM and then delete leftovers. Fix 44850f6: ABANDON_S=900 (the 15-min rule), and /health workers count as placed.
- 18:17: TX-3 rerun 2 started on 44850f6 (log mscache-rerun2-US-TX-3.out).
- 18:24 MO-1 pilot exited (report runs/mscache/11220157-US-MO-1-a100/report.md).
  - S2gap60: 3/3 cached on 7b03, staging median 23 s, worst 50 s.
  - S9: 3 measure loads, submit to job start ~22 s; container started ~2 s after submit, so cached (rows say "unknown" because of the pod-reuse bug).
  - S2gap600: no_worker on its first measure after the 600 s gap (old 300 s rule).
  - The pod-only "taint" pods (9) are all on 7b03, the same machine: endpoint worker churn, not cross-machine taint.
  - MO-1 leftovers: none.
- 18:25: MO-1 S2gap600 rerun on 44850f6 (likely $0.64, worst $1.85, ~45 min; log mscache-rerun-US-MO-1.out).
- 18:36 TX-3 STOPPED under the 15-min rule. The rerun-2 S2gap60 seed was placed on 623z (pod at +190 s), sat throttled for 900 s and was cancelled. That is the same machine where all 3 original TX-3 scenarios sat throttled. SIGTERM'd the run; its S9 endpoint was deleted by hand. TX-3 l40s S2gap60 and S9 have NO data. The 4090 fallback was not used: its condition (no TX-3 scenario gets a worker) is not met, because S2gap600 runs on hyxm. Retrying TX-3 needs Dean.
- 18:40 MO-1 STOPPED under the 15-min rule: the S2gap600 rerun seed got no pod and no /health worker for 900 s (A100, MO-1). Endpoint deleted by the harness. MO-1 S2gap600 has NO data. Retrying needs Dean.
- Next: once each DC's run exits, check leftovers. For any arm that ended stall or throttled, rerun it serially inside the approved cost, e.g. `run --dc CA-MTL-1 --gpu a40 --spend --serial --only S2gap600,S9`.

## Pilot results (2026-11-21 local, 17:57-18:46; 01:57-02:46Z 11-22)
Run dirs, under <worktree>/runs/mscache/: 11220157-CA-MTL-1-a40, 11220157-US-MO-1-a100, 11220157-US-TX-3-l40s. Rerun dirs (failed only): 11220211/11220217-US-TX-3-l40s, 11220225-US-MO-1-a100. Leftovers: none, verified 18:46. Spend upper bound $3.32 (staging counted as billed); `make billing` about 1 h after the run.

**CA-MTL-1 A40: complete, 9/9 measure loads served from cache.**
- S2gap60 on vys4: staging 48.8, 2.9, 3.0 s. The first includes the ~45 s ai-api wait.
- S2gap600: staging 3.0 and 43.5 s on vys4. The third load ran on a DIFFERENT machine, 73y4, at 51.4 s. Was that a peer pull or a local copy? The host-log join must say.
- S9: container start 1.5-2.0 s after submit. A pod (izhe49qnrc09mm) already existed before submit and was restarted (rows say "unknown", re-read as cached).

**US-MO-1 A100: partial.**
- S2gap60 3/3 cached on 7b03: 49.8, 23.4, 4.4 s.
- S9 3/3 cached (cs about 2 s; restarted pods huv8/g3hn).
- S2gap600: no data. no_worker after the 600 s gap (old 300 s rule), then the rerun seed got no worker for 900 s and stopped under the 15-min rule.

**US-TX-3 L40S: mostly no data; stopped under the 15-min rule.**
- Placement is slow: pods appeared 190-470 s after submit.
- Machine 623z took workers and kept them throttled for 13-15+ min, three times.
- S2gap600 seed was a pull on hyxm (165 s). Then 1 measure after a 600 s gap was cached on hyxm (23.6 s). The next measure was no_worker (old rule).
- S2gap60 and S9: no data (pilot, rerun 1 and rerun 2 all failed).

Seeds: every seed was a pull (MTL 317-450 s, MO-1 384-574 s, TX-3 165 s). In each DC all 3 parallel seeds co-landed on one machine.

Platform findings for the main Mac (via DQ):
1. The GraphQL pods query lists a pod that is staging a model, or throttled waiting for a GPU, as desiredStatus EXITED. Only /health shows it alive.
2. TX-3 machine 623z keeps accepting workers it cannot give a GPU to (throttled for 15+ min, 3 times, 11-22 02:00-02:36Z).
3. Parallel endpoints with different models in one DC co-land on one machine (all 3 DCs).
4. Scale-from-zero restarts a stopped worker under its old pod id, and a pod can exist before the job is submitted (MTL S9).

Harness fixes made live: c496049 (/health liveness, throttle limit, --only/--serial), f3b64d8 (restarted pods keep their id), 44850f6 (no_worker after 15 min, counting /health). Lesson: SIGINT does not stop `run` (the worker thread continues); SIGTERM, then delete leftovers.

## Harness change: fresh endpoint per redeploy (2026-11-21, commit bab67f4, $0)
Asked by the main Mac via DQ after its joined read (every cached load used the local disk copy, no transfers; 73y4's 51.4 s was spent before the container was created; 623z had 8/8 GPUs rented; finding 4 = expected behavior).
- spec.Recreate step + spec.fresh_redeploy(gap): Teardown, delete endpoint, create a new one (same DC, pool, model; name suffix -1/-2/-3), Idle, Deploy. S2 uses it now; S3 and S10 must use it when coded. Recreate adds about 10 s wall per redeploy, no worker cost.
- Every load records `resumed` (pod_reused, or a pod id seen on an earlier load) and `floor_wait`.
- floor_wait = cached, and ai-api's 40 s scale-from-zero floor falls inside the staging window (window opens before +38 s, container starts at or after +38 s). The report keeps floor_wait loads out of staging stats, prints "cached behind ai-api floor: n", "new pod=n resumed=n", and marks rows "(ai-api floor)". A late-placed pod (TX-3 kzlv, pod +238 s) is not floor_wait.
- Tests: 102 passed, 1 skipped.
- Counter-evidence sent to the main Mac: in the pilot every S2 measure load ran on a NEW pod id, created 0.5-1.0 s after submit (one at 24.9 s). Only S9 resumed pod ids. So pilot S2 did not measure a resume; the change removes the doubt anyway.
- Pilot re-read with the floor rule: 5 of 8 cached S2 measures were floor_wait (MTL aj82 48.8, tpr5 43.5, 9656 51.4 on 73y4; MO-1 a33p 49.8, 64q5 23.4). The others staged in 2.9-5.0 s (TX-3 kzlv 23.6 s, placed late). Real cached copy time is about 3-5 s.
- Expect: with a new endpoint per redeploy, most S2 measures will be scale-from-zero and hit the floor, so S2 reads as cached vs pull (75 s cutoff), not as copy time.
- Costs (python -m mscache plan): retry MO-1 S2gap600 likely $0.64, worst $1.85; TX-3 S2gap60+S9 on L40S $0.63/$2.28 (on 4090 $0.45/$1.61). Both retries $1.27/$4.13. Full matrix unchanged, about $50-60 likely, about $170 worst, 2 windows of 4 h or less.

## Waiting on Dean
- Pilot gaps: retry MO-1 S2gap600 (A100 had no capacity at 02:25-02:40Z) and TX-3 S2gap60/S9 (623z throttles)? Now on bab67f4 (fresh endpoint per redeploy). Cost: both $1.27 likely / $4.13 worst; TX-3 on 4090 instead $0.45/$1.61. Needs a new go typed here.
- Whether to push feature/mscache-harness to origin (push denied in this session).
- Answers on spec corrections 1-5 above (before the full matrix, not before the pilot).

## Decisions made (and why)
- 2026-11-21: separate topic instead of SDK; Dean's choice. Keeps the test program out of SDK's long context and lets SDK do the Wasm SDK work in parallel.
- 2026-11-21: session opened, read this file, announced ownership to DQ.
- 2026-11-21: Dean typed "go" before any plan/cost existed; treated as go for the free design+build only, not a paid run (the rule needs a stated cost first).
- 2026-11-21: harness lives in the bench repo (mscache/), not handoff-bundles: git history, reuse of bench.runpod_api, SDK agreed.
- 2026-11-21: one worker per scenario, narrow GPU pool, one distinct model per parallel scenario (taint lesson from waves 1-3).
- 2026-11-21: pilot DC CA-MTL-1 on A40: largest LRU DC, cheapest pool, 79/81 placement in wave 2, peer pull live.
- 2026-11-21: outcome classified by staging time (container_start - pod_created), not file mtimes (upstream mtimes made them useless in waves 1-3).

## Worktrees
- ~/Desktop/flash-model-cache-bench/.claude/worktrees/mscache-harness, branch feature/mscache-harness (harness lives in mscache/ there). Created 2026-11-21.

## Knowledge the next session needs
Memory files in /Users/deanq/.claude/projects/-Users-deanq-Desktop/memory/ (loaded automatically; cwd is ~/Desktop):
- machine-role-runpod-user.md: the user-not-employee rule.
- flash-model-cache-bench.md: the bench, `make billing`, GraphQL modelReferences and forced-cold tricks.
- tx3-modelstore-peer-cache-probe.md: TX-3 probe facts (Flash cannot run in TX-3: no flash volume; plain-image probe).
- prototype-benchmark-rigor.md: keep runs to 3-4 h.
Full wave logs and the main Mac's joined reads: /Users/deanq/.claude/roles/topics/SDK.md, sections "2026-11-19 CA-MTL-1 wave", "2026-11-21 CA-MTL-1 wave 2", "wave 3", "LRU waves for US-TX-3 and US-MO-1", and the Inbox entries from the main Mac dated 2026-11-21.

## Handoff from SDK
Written by SDK, 2026-11-21. The full logs are in SDK.md (the sections named in "Knowledge" above); this is only what you need to run things.

LAYOUT
- ~/Desktop/flash-model-cache-bench (git): supplies the venv (.venv, uv) and `bench.runpod_api.Runpod` (REST + GraphQL client; key from ~/.runpod/config.toml). The probe imports it.
- ~/Desktop/handoff-bundles/tx3-modelstore-peer-probe (NOT git; backups of earlier versions are in runs-2026-11-21/backups/):
  - probe_handler.py: the worker. It runs from env PROBE_SRC inside python:3.12-slim and does no GPU work. It inspects the staged HF cache at /runpod-volume/huggingface-cache/hub and returns machine identity, container_age_s, handler_age_s, now, staged_mtime_min/max and first_job_on_worker.
  - cycle.py: the loop driver. It holds MODELS (alias -> repo, label tag, read flag), ROTATIONS, VARIANTS (GPU configs per kind), create() and summarize().
    - create() makes the endpoint via REST with dataCenterIds=[PEER_DC], then a GraphQL saveEndpoint full upsert to set modelReferences and instanceIds. It fails loudly if the ref did not save.
  - rounds.py: run_round(), pods(), timing() and log(). Writes one JSON line per job to results.jsonl.
  - redeploy.py: the scale-to-0-and-back test.
  - runs-2026-11-21/: the .out logs for CA-MTL-1 wave 3 (mtl3-*), TX-3 (tx3l-*) and MO-1 (mo1l-*), plus endpoint IDs by model.
  - create.py, set_model.py, saturate.py: older one-off tools (single endpoint, swap model ref, hold N workers). Not used since wave 1.
- Endpoint names are `<dc>-peer-*`, e.g. mtl-peer-, mo1-peer-, tx3-peer-. Leftover checks filter on "-peer-". Endpoint mjit43r8xtub5o is not a probe endpoint; leave it.

HOW TO RUN (from the bench dir)
  cd ~/Desktop/flash-model-cache-bench; P=~/Desktop/handoff-bundles/tx3-modelstore-peer-probe; export PYTHONPATH=.:$P
  PEER_DC=US-MO-1 PEER_OFFSET=0 nohup .venv/bin/python -u $P/cycle.py <hours> 0 <kind> <workers> <model|rotation> > out 2>&1 &
  PEER_DC=US-MO-1 nohup .venv/bin/python -u $P/redeploy.py <hours> <model_alias> <variant> <kind> > out 2>&1 &
- kind is one of gpu (TX-3: 4090/l40s/mix), gpu-other, cpu, mo1 (h100/a100/l40s/mix) or mtl (a40/h100/mix). Rotations: popular, mo1, mtl, tx3-lru, mo1-lru. The -lru rotations leave out the redeploy model.
- cycle.py: each cycle makes one new endpoint, sends `workers` jobs, stops at the first COMPLETED job, cancels the rest and deletes the endpoint. A config that gets no pod in ABANDON_S (300 s) is dropped, and the next endpoint rotates to the next VARIANT. PEER_OFFSET staggers the start model and variant between parallel loops.
- redeploy.py: one endpoint. Each round sets workersMax 0, waits until all pods are EXITED/TERMINATED or gone (GONE_TIMEOUT_S 600), sleeps SETTLE_S (60 s), then runs a round of 1 worker. In `finally` it deletes the endpoint and logs REDEPLOY DONE. It does NOT wait for "handler ready + 2 min" before teardown and has no configurable idle gap. S2/S3 need both, so add them.
- Leftover check:
  .venv/bin/python -c "import cycle; rp=cycle.Runpod(); e=[x for x in rp.graphql(cycle.ENDPOINTS_QUERY,{})['myself']['endpoints'] if '-peer-' in x['name']]; print(e or 'none')"

WHAT IS LOGGED PER LOAD (results.jsonl plus the .out lines)
- Each job record: round label, endpoint, job id, worker (pod) id, machineId, location, gpu, delay_ms, exec_ms, submitted (epoch s), and the handler output.
- `r<label> TIMING pod= machine= {offsets}`: offsets in seconds from submit for pod_created and pod_last_started (GraphQL createdAt/lastStartedAt), container_start (now - container_age_s; container age is computed from pid 1's start time against uptime), handler_start, first_file_write and last_file_write (min/max mtime over the model's files), and job_start.
- POD-ONLY lines: pods placed but killed before finishing a job. They still pull the model, so count them as holders of the copy.
- CYCLE summary:
  - fresh_pulls = workers whose staged_mtime_max is later than submit.
  - pre_container = delay_ms/1000 - container_age_s, i.e. the time before the container started. It is about 2-5 s when cached and about 45-60 s when it includes the ai-api queue wait.
- Classify a load with BOTH mtimes:
  - first and last write both before submit = cached;
  - first before submit, last after = partial (a top-up of an existing copy);
  - first after submit = full pull.
  - Many blobs keep upstream mtimes, so "no write after submit" is strong evidence of cached, but a file write alone does not tell peer from origin.
- Gaps against your spec: no model revision or size per load (the saveEndpoint response holds the resolved ref and commit), and no "pod resumed" phase separate from lastStartedAt.

GOTCHAS
- lastStartedAt and the in-container start sometimes disagree by about 20 s. Log both.
- Stalls: a pod staging a large model is not desiredStatus RUNNING yet. The stall cut (STALL_S 600 s) fires only when every placed pod is EXITED/TERMINATED for that long. A 180 s cut gave false abandons on 58 GB pulls.
- Teardown: EXITED pods stay listed for a while. redeploy.wait_gone now ignores them (it used to hang 600 s).
- Taint: any pod holding the model on the landing machine during a redeploy gap spoils the round. That includes our own loops and POD-ONLY pods, and other customers you cannot see. Keep the redeploy model out of any parallel loop.
- Abandons: an endpoint with no pod in 5 min costs $0 and is deleted. A100 in MO-1 placed only on 7b03 and dfl0; host 287n crash-looped A100 pods on 11-19.
- TX-3 capacity: the catalog lists no GPU or CPU availability for US-TX-3, yet 4090 pods still place. On 11-21 everything ran on 7vmo (loops) and 7sfc (redeploy); 4090-only and L40S never placed. CPU never places in TX-3.
- Monitor pipelines: do not put `cut` or `head` after grep in a Monitor; they buffer and the monitor stays silent. For "tell me when done", use a background `while pgrep -f cycle.py ...; do sleep 20; done`.
- Worker count: keep it at 2. Every extra worker can mean another full pull that is killed when the first job finishes.

BILLING
- MCP tool list-serverless-billing with startTime/endTime and bucketSize hour|day; lastN cannot be combined with start/end. Records carry serverlessId, so sum the rows for your endpoint IDs.
- Rows lag by more than 1 h. For the bench's own runs, `make billing` in the bench repo works.
- Known costs: the 11-20 UTC day was $9.28 (CA-MTL-1 waves plus the tail of MO-1). My live estimates ran high.

IN FLIGHT
- Billing recheck, not posted at 19:40Z on 11-21:
  - CA-MTL-1 wave 3, 17:05-17:52Z, 108 endpoints; IDs in runs-2026-11-21/wave3-by-model.txt.
  - TX-3 + MO-1, 18:48-19:37Z, 18 + 89 endpoints; IDs in runs-2026-11-21/{tx3l,mo1l}-by-model.txt.
  - Estimates: about $10, $1-2 and $8-10. Sum by serverlessId once the rows post.
- Open for the main Mac's join (already sent to DQ):
  - the MO-1 redeploy top-ups (70ia r6, pptb r10/r11);
  - the 7vmo Devstral rewrite about 10 min after the first pull (ftw118v017hygv, then orrcs9qlad46or);
  - wave-3 questions on container-start gating and lastStartedAt skew.
- SDK holds no crons, monitors or running processes. Nothing is left deployed (leftover check: none, 19:37Z).

## Inbox
- 2026-11-24 main-Mac DQ: host v4.9.0 is GA. DD 22:39-22:49Z 11-24: 4,811/4,811 machines, CA-MTL-1 233/233, US-MO-1 56/56 (fleet flipped 22:10-22:20Z). Host half of the before/after test is ready. Pending: RunPod#7549 (graphql half of SLS-1099, keep in-flight copies under LRU) in RunPod 1.772.0, not yet tagged; main Mac will say when it is in prod. S4/S5 and the S2 regression check don't need #7549; S3/S10 now would measure host-only. Paid runs need Dean's go typed here.
- 2026-11-23 main-Mac DQ CORRECTION: host v4.9.0 is NOT released. The 68 machines are the RC canary set (CA-MTL-1 1 of 233, US-MO-1 2 of 56), not a prod rollout. The v4.9.0 before/after test waits for GA; main Mac will say when it lands. RunPod#7549/#7547 are not in a RunPod release either (last is v1.771.0, 11-20). The 11-25 to 11-29 window is off.
- 2026-11-23 main-Mac DQ: RunPod#7549 status known only at Tuesday 11-24 release cut; main Mac will send whether #7549/#7547 made it and when v4.9.0 covers most of CA-MTL-1 and US-MO-1. Accepted order S3/S10, then S4/S5, then S2 gap600 + control; DQ's rough range $5-10 likely / ~$30 worst (to be replaced by MSCACHE's own quote).
- 2026-11-23 from main-Mac DQ [cf55d1] (planning only, no paid go): merge freeze after Tue 11-24 deploys until Mon 11-30. Host v4.9.0 is the only new Model Store code: canaries since 11-24 00:10Z on 68 machines (incl. CA-MTL-1 pdadanvocqnq, US-MO-1 99b3jg83jalb + 7b0328fzr7pb), wider rollout Tue. Contains #3303 (removing a version stops its in-flight downloads; re-add reuses blobs on disk), #3296 (separate origin/peer slots), #3302 (hedge to origin when a peer is slow). Graphql teardown fix RunPod#7549 (keep in-flight copies under LRU) is merged, maybe not released Tue; main Mac will confirm. Proposed window: 11-25 to 11-29, after main Mac confirms v4.9.0 on most CA-MTL-1 and US-MO-1 machines. Before/after v4.3.0 -> v4.9.0, same models/GPUs/arms: (1) MAIN: S3 short-round teardown + S10 teardown and immediate redeploy (11-22: 6/6 full pulls 266-711 s); report time to ready and same-machine or not. (2) S4 peer vs origin + S5 slot contention in CA-MTL-1/US-MO-1: one warm deploy while the same machine cold-pulls 2+ models. (3) S2 gap600 in one LRU DC + EUR-IS-2 control as a regression check. Skip LAN transport, probe cap, #3315, TX-3/NC-1 if no capacity. Per run: endpoint, pod, machine, model+revision, phase timestamps to DQ. Ask: reply with a cost estimate and scenario pick; do not start until Dean's go typed here AND main Mac confirms the version spread.
- 2026-11-21 SDK reply (Dean asked me to check with SDK):
  - Billing posted, 11-21 day: MTL wave 3 $4.20 (108 eps), TX-3 $0.35 (11/19 billed), MO-1 $5.49 (87/89). Waves ~$10.04 for ~200 endpoints => ~$0.05/load; SDK's live estimates ran ~2x high. No-worker endpoints cost $0.
  - Main-Mac joins still open: MO-1 70ia/pptb top-ups; 7vmo Devstral rewrite (ftw118v017hygv then orrcs9qlad46or).
  - Gotchas: pre-container wait bimodal 2-5 s or 45-60 s (ai-api); vys4/hww1/9lk2/rsjp often start in 2 s. TX-3 7sfc cached yet 3-261 s waits (unexplained, maybe slot contention). MO-1 A100 only on 7b03/dfl0; 287n crash-loops. 7vmo spawns 3-6 POD-ONLY pods per endpoint (they pull, inflate holders). TX-3 4090-only never places; mix does. saveEndpoint needs instanceIds. rest.runpod.io does not resolve here; use api.runpod.io. Billing: rows lag >1 h, lastN not with start/end, match serverlessId.
  - Same-machine redeploys: believed placement prefers holders, not proven (thin capacity confound). Test: fresh endpoint, same model, GPU mix, no other load, compare landings vs get-capacity snapshot at submit; main Mac confirms from scheduler logs.
  - Harness asks (all adopted in the design): configurable gap; teardown at ready+2 min; revision+size per load; POD-ONLY pods recorded as taint; structured JSON per load; cost from GPU-seconds; one worker, no parallel loops in same DC; ask main Mac to join "other holders on landing machine"; harness in the git bench repo.
- 2026-11-21, from main-Mac DQ via DQ: calls on the 5 spec corrections (apply before the full matrix; pilot unchanged):
  1. S3: anchor on pod_created. Tear down at pod_created + 10 s, 30 s and 60 s, with a large model (100+ GB) so the staging window is long enough. Record whether the container had started before the teardown; if it had, that round counts as S2, not S3. Redeploy immediately and log staging_s on the redeploy; they read duplicate bytes and download wait from host logs.
  2. File-write phases come from their host logs. Your phase list plus staging_s is what they need; include pod id and machine id per load.
  3. S6: observational only. Do not target disk pressure; they read evictions and DISK_LIMIT from host and graphql logs across the matrix and report per machine.
  4. Control DC: EUR-IS-2 by default (busiest LRU-off DC, no peer). Match the GPU type: compare each LRU DC against a control run on the same GPU type if the control DC has it; if EUR-IS-2 lacks the type, try US-NE-1 with the $0 abandon check. Note in the output which GPU type each control run used.
  5. Agreed: record the machine per load, and serialize if loads collide.
  Pilot (CA-MTL-1, A40, S2 at 60 s and 600 s + S9, ~$0.67) confirmed as right; it starts on Dean's go typed in this session.
- 2026-11-22, from main-Mac DQ via DQ: MS retracts the resume caveat; MSCACHE was right.
  - All 9 S2 loads ran on NEW pod ids, and each found the model on disk with zero holders and no removal. So the S2 hits are real LRU retention for a new worker on the same machine (vys4: released 61-68 s and 602-607 s before submit; 7b03: 62-67 s). The 600 s gap was survived purely by retention.
  - The fresh-endpoint change is fine to keep but not required. Keep the resume-vs-new logging.
  - Correction 1: the slow 43-50 s loads had their pod within ~1 s. The delay is ai-api's ResumePod arriving at +42-49 s instead of +2-3 s (ai-api, not host or copy time; floor path likely).
  - Correction 2: "Disk utilization validation failed" fires on all 9 pods, so it did not delay 9656 on 73y4.
  - Check: 64q5on3hbd3zjf's staging_s 23.4 s looks measured from pod create (+24.7 s), not submit. Fix the definition or label it.
  - The floor-hit vs copy-time split (copy ~3-5 s) stands.
- 2026-11-22, from main-Mac DQ via DQ: host-log read of CA-MTL-1 window 2 (MS + DD, host v4.3.0, 19:04-19:50Z).
  - S4 (Yi-1.5-34B on 90vb): one shared transfer set served all 3 peer pods (16 blobs, mounts 19:19:10-16). 15 of 16 blobs came by PEER: 68.8 GB in 632 s = 109 MB/s total, ~24-30 MB/s per blob, 4 at a time; 9 blobs waited 172-540 s for a slot. The holder i96l pulled the same model from ORIGIN in 126 s = 544 MB/s, so peer was ~5x slower. The peer source isn't logged on v4.3.0, but timing fits i96l (finished 19:07:09; 90vb started 19:08:21). "context canceled" lines are the losing origin leg of the peer/origin race, not errors. A platform fix (hedge to origin on slow throughput) is in progress on their side.
  - S5 (tm9s): the three 72B pulls QUEUED; they did not share bandwidth. Order Smaug, Qwen1.5-72B, Qwen2.5-72B; each ~141-145 GB at ~470-536 MB/s from origin. The 4 slots were full 93% of the time. Slot wait per model (total/max): Smaug 3,785/249 s, Qwen1.5 16,718/538 s, Qwen2.5 27,422/770 s. The ~245 s gaps are slot queueing. Pods: Smaug i0w3wb3vg2b8q1 (gwf2pmy7e5h06r), Qwen2.5 5ban657lqgwbty (ztqhipx1qpd8p4); Qwen1.5's endpoint is not mappable on their side (send it).
  - Design note: to cover "warm behind busy slots", warm and cold must land on the same machine. Consider sizing the cold load to fill a small DC, or retrying until they co-land; record the machine.
- 2026-11-22, from main-Mac DQ via DQ: combined read of the full matrix (MS: host, graphql and ai-api logs).
  1. LRU CONFIRMED. Control EUR-IS-2: after each pod ended, graphql unassigned the copy and the host removed it ~17 s later; each measure full-pulled. LRU DCs: all 7 gap600 measures (MTL 73y4, SE-1 86fs and i8a9, MO-1 287n) reused retained copies: zero holders for ~10 min, no removal, 0 blobs moved.
  2. MTL gap3600 hits were local copies, not peer (tm9s from a non-matrix pod's pull that LRU kept; i96l kept the seed). Correction: 73y4 at 16:54 was NOT cached: it pulled ~15 GB from origin in 24 s inside the 74.5 s floor load. Our 75 s cutoff misclassified it.
  3. S3 6/6 = in-flight copy dropped at teardown. Same machine (MTL t30, t60; SE t60): partial removed, full re-pull. Different machine (MTL t10; SE t10, t30): the first transfer kept running to completion while the redeploy pulled the whole model on a second machine (~280-290 GB moved for a ~140 GB model), leaving an untracked copy. Platform fix in progress (theirs).
  4. The 45-75 s "cached" loads are ai-api's 40 s catch-up floor (fires at 42.6-49.2 s queue time) when no resumable worker exists. Host time from container sync to model mount is 0.5-1.7 s on every cached load. S10 the same.
  5. failed/no_worker: GPU capacity (SUPPLY_CONSTRAINT, 0 candidates), 0 disk refusals; 2 TX-3 "model is being prepared for this region's shared storage" errors.
  6. Slow TX-3 seeds: the transfers were short (15 GB in 126 s, 16 GB in 47 s); the pods sat idle until a stop/resume, with the copy ready 8-18 min before the pod started. NC-1 the same. Not the cache; cause not in logs.
  7. Per DC (cached mount on host | cold pull aggregate MB/s median | origin per blob | peer share/speed): MTL 0.55 s | 468 | 133 | 5%/29 MB/s; SE-1 1.39 s | 249 | 140 | 0; IS-2 control 1.72 s | 433 | 191 | 0; MO-1 0.54 s | 250 | 154 | 0; NC-1 n/a | 590 (n=2) | 241 | 0; TX-3 n/a | 318 (n=2) | 132 | ~0/12 MB/s.
  Gaps ranked: ai-api 40 s floor (biggest cost on hits; fixes in review); in-flight drops (fix in progress); pods idle before reaching the host in TX-3/NC-1; peer ~5x slower than origin (fix in progress); the 4-slot queue serializes cold pulls.
  Harness follow-up for us: the 75 s cached/pull cutoff misclassifies a fast origin pull inside the floor (73y4 16:54); reclassify from the host join where available.
