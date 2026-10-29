# Pipeline evolution log (line I)

Every attempt is logged here, including failures, abandoned runs and bugs. Numbers come from files under
outputs/evolution/; code under code/evolution/. Baseline = pipeline_best.yaml v0.

## E0 — baseline v0 (from lines E–H)
- Pipeline: rules + Qwen3-VL-32B → ViPE pose/K + Pi3X depth (median-ratio scale to ViPE) → rule-score QC + flags.
- A group (51 clips, no ETH3D), H:vipe+pi3x+med: pc F@5cm 0.809, traj-aligned pc F 0.618, metric AbsRel 0.061,
  g5/g20 EPE 4.82/11.1 px, fail rates scale 28.6% / depth 7.1% / FOV 19.6%.
- Known gaps: ViPE scale failures, feed-forward FOV bias in the wild, ViPE pose failures on driving/wrist cams,
  low yield (~36% filter pass, large QC rejection in the wild).

## E1a — which intrinsics source to condition on? (2026-10-17 23:40 – 10-18 00:30, CPU + 0.1 GPU·h)
- Hypothesis: GeoCalib (single-image, no motion needed) is a good independent K source; "ViPE if |ViPE−GeoCalib| < 3°
  else GeoCalib" fixes ViPE's K failures.
- Done: `code/evolution/geocalib_bench.py` (ViPE's GeoCalib copy, 12 frames/clip, median focal) on A (no ETH3D) / B /
  ScanNet++ / C = 164 clips (~1.1 s/clip on one H200); `fov_sources.py` → `E1/fov_sources.csv`; `k_select.py` →
  `E1/k_select.json` (HFOV error vs GT on the 92 GT clips; fail = error > 5°).
- Results (GT clips, fail rate / mean error):

| K source / rule | fail >5° | mean err | notes |
|---|---|---|---|
| ViPE | 10.9% | 3.26° | all failures but one on Sintel (CG, fast motion) |
| GeoCalib | 41.3% | — | bad on KITTI (21°), Sintel (13°), vKITTI2 (11°) |
| Pi3X self | 38.0% | 5.74° | |
| DA3 self | 42.4% | — | |
| median(ViPE, GeoCalib, Pi3X, DA3) | 30.4% | 4.06° | |
| **gate**: ViPE unless HFOV ∉ [25°, 130°] → median(GeoCalib, Pi3X, DA3) | 12.0% | **2.73°** | fixes 3 Sintel collapses (16–22°) |
| vote (switch when others agree <5° and ViPE is >5° off) | 10.9% | 2.92° | |

- Disagreement as a QC signal for a ViPE FOV failure: |ViPE − median(others)| AUROC **0.87**, |ViPE − GeoCalib| 0.83,
  GeoCalib's own uncertainty only 0.67.
- **Wild (C) finding**: ViPE's focal collapses on 10/72 clips (14%): negative focal (DROID ep028), HFOV 7.7–18.8°
  (DROID wrist ep039/ep053, embodied YouTube, hard cases, synthetic zoom) or ~160° (one driving, one Sekai).
  Embodied median HFOV: ViPE 26.8° vs GeoCalib 63.7° / Pi3X 73.2°. Feeding these K's to a depth network would be
  harmful, so the gate is required before any K conditioning. The gate switches 15% of C (embodied 50%, hard 30%).
- Rejected: "ViPE if close to GeoCalib else GeoCalib" (would hand 40%+ of clips to a worse source).
- Adopted (pending E1b end-to-end check): K rule = gate; disagreement |ViPE − others| becomes a QC flag.
- The planned GeoCalib-K Pi3X run (`pi3x_kgeo`) is kept only as an error-propagation probe (bad K in → ?).

## E6a — ViPE at half frame rate (zero GPU, re-uses outputs/qc/variants/*_sub2) (2026-10-18 00:20)
- Hypothesis (ideas.md #4): ViPE on every 2nd frame keeps accuracy and halves the main cost.
- Done: `code/evolution/e6_sub2_eval.py` → `E6/vipe_sub2.csv`; both runs evaluated on the same frame subset vs GT.
- Results (median; A n=50, B n=13):

| | ATE Sim3 % | scale fail | FOV err | metric AbsRel | wall s/clip |
|---|---|---|---|---|---|
| A full | 0.588 | 16% | 1.21° | 0.113 | 73.0 |
| A sub2 | 0.578 | 18% | 1.04° | 0.120 | 72.7 |
| B full | 0.346 | 69% | 0.18° | 0.208 | 118 |
| B sub2 | 0.428 | 85% | 0.17° | 0.248 | 106 |

- Accuracy: A unchanged; B (TartanAir, vKITTI2: fast forward motion) slightly worse ATE and scale.
- **Cost surprise**: wall time barely drops (73 → 73 s on A). The QC variant runs were made under heavy CPU load
  (load ~240), so the timing is confounded, but ETH3D clips with 7–8 frames also took 31–32 s → a large fixed
  per-clip overhead (process start + model loading in `vipe infer`). Next (E6b): controlled timing of full vs sub2
  vs a persistent worker that loads the models once.
- Decision: pending E6b (accuracy is acceptable for static/slow scenes, not for fast driving).

## E1b — K-conditioned Pi3X / DA3 runs (2026-10-17 23:45 – 10-18 01:10, ≈4.5 GPU·h on GPUs 1–4)
- Code: `code/evolution/pi3x_cond.py` (variants kgt / kvipe / kgeo / pkvipe / kgate), `da3_cond.py` (ViPE poses+K),
  eval `e1_eval.py` → `E1/raw`, `E1/tables.md`. 164 clips each (A 51, B 13, ScanNet++ 28, C 72), 0 failures.
- Sanity: Pi3X obeys the conditioning K (recovered HFOV within 0.3° of the input). DA3 ignores intrinsics unless
  extrinsics are given too (model/da3.py), so DA3 was only run pose+K conditioned.
- Cost: Pi3X 19 s model time / 200 frames (+~20 s image loading on CPU), peak 84 GB; pose-conditioned 29 s / 34 GB;
  DA3 pose-conditioned 16 s / 31 GB.
- A group (51 clips), medians and failure rates:

| label | ATE Sim3 % | scale fail | FOV fail | metric AbsRel | pc F@5cm | traj-aligned pc F | g5 / g20 EPE px |
|---|---|---|---|---|---|---|---|
| v0 H:vipe+pi3x+med | 0.592 | 28.6% | 19.6% | 0.061 | 0.809 | 0.618 | 4.82 / 11.1 |
| H:vipe+pi3x_kvipe+med (depth only conditioned) | 0.592 | 28.6% | 19.6% | 0.058 | 0.825 | 0.619 | 4.75 / 10.8 |
| H:vipe+pi3x_kgt+med (oracle K) | 0.592 | 28.6% | 19.6% | 0.058 | 0.820 | 0.627 | 4.75 / 10.8 |
| pi3x image-only | 0.732 | 46.4% | 45.1% | 0.075 | 0.835 | 0.456 | 3.76 / 7.69 |
| pi3x_kvipe (own poses) | 0.652 | 67.9% | 19.6% | 0.095 | 0.799 | 0.670 | 4.49 / 8.21 |
| pi3x_kgeo (GeoCalib K) | 1.27 | 57.1% | 56.9% | 0.131 | 0.816 | 0.436 | 3.70 / 6.70 |
| da3_pkvipe (DA3 on ViPE cams) | 0.592 | 28.6% | 19.6% | 0.079 | 0.709 | 0.627 | 4.37 / 7.93 |

- B group (13, partly in the models' training data): pi3x_kvipe scale fail 15% (ViPE 69%, image-only Pi3X 54%),
  metric-depth fail 15% (ViPE 69%), ATE 0.25% (ViPE 0.35%); vKITTI2 all failure rates 0.
- Findings: (1) as a depth source inside the v0 hybrid, ViPE-K conditioning gives small gains that already match the
  oracle K; (2) as a stand-alone annotator, K conditioning fixes Pi3X's FOV and trajectory shape (traj-aligned cloud
  0.456 → 0.670, g20 EPE better than any hybrid) but its metric head becomes biased on real indoor/KITTI clips
  (sim3 scale 1.14 / 1.33: the wrong image-only FOV had been compensating scale); (3) a wrong K propagates
  (GeoCalib K → FOV fail 57%, ATE 1.27%), so the gate from E1a is mandatory; (4) DA3 conditioned on ViPE cameras
  gives depth consistent with ViPE poses (g20 7.9 px) but a worse depth shape than Pi3X (pc F 0.71).

## E1c — Pi3X as the main annotator, ViPE only for K and metric scale (2026-10-18 01:10–01:40, CPU)
- Code: `e1b_eval.py` (pseudo-methods `<m>~vipe`, `route_h`, `route_s`) → `E1b/`; `e3_scale_vote.py` + `e3_eval.py`
  → `E3/` (scale sources: ViPE trajectory `vipe_t`, ViPE depth median ratio `vipe_d`, DA3, votes).
- Label v1 = `pi3x_kgate@vipe_d`: Pi3X conditioned on the gated ViPE K, own poses and depth, one per-clip scale =
  median(ViPE depth) / median(Pi3X depth).

| group | label | ATE Sim3 % | SE3 ATE fail | scale fail | metric AbsRel | pc F | traj-aligned pc F | g5 / g20 EPE |
|---|---|---|---|---|---|---|---|---|
| A (51) | v0 | 0.592 | 10.7% | 28.6% | 0.061 | 0.809 | 0.618 | 4.82 / 11.1 |
| A (51) | **v1** | 0.652 | **0%** | 32.1% | 0.062 | 0.813 | **0.670** | **4.06 / 7.03** |
| ScanNet++ (28) | v0 | 0.223 | 3.6% | 3.6% | 0.042 | 0.965 | 0.767 | 0.81 / 1.91 |
| ScanNet++ (28) | v1 | 0.238 | 0% | 14.3% | 0.045 | 0.982 | **0.942** | 1.40 / 1.90 |
| ScanNet++ unseen-3 | v0 / v1 | 0.21 / 0.23 | 0 / 0 | 0 / 33% | 0.074 / 0.062 | 0.973 / 0.983 | 0.888 / 0.948 | 0.54 / 1.82 |
| B (13) | v0 / v1 | 0.345 / 0.251 | 31% / 46% | 69% / 77% | 0.158 / 0.165 | 0.049 / 0.064 | 0.041 / 0.070 | 7.05 / 4.05 |

  C group (72, no GT): g5 self-consistency 0.858 → 0.889, g20 0.654 → **0.792**, photometric warp error 14.6 → 11.5,
  deviation from the other metric methods 1.29× → 1.08×; embodied g5 0.672 → 0.889, driving 0.612 → 0.726.
  A-group mean rank (this experiment's methods): v1 4.75 vs v0 6.43.
- Scale votes (A / B scale-fail): own 68% / 15%, vipe_t 29% / 69%, vipe_d 32% / 77%, med3_d 36% / 31%,
  med5 32% / 46%. No vote fixes both: on unseen real data (A) ViPE is the best scale source, while Pi3X's own metric
  wins on B — but B (TartanAir, vKITTI2) is in Pi3X's training data, so that win is not trusted. Scale disagreement
  predicts scale failure only weakly (AUROC 0.66–0.78, n=41). **Metric scale remains the main open problem.**
- Trade-off seen on ScanNet++: v1 makes the cloud far more self-consistent (0.767 → 0.942) but its depth-ratio
  scale disagrees with ViPE's trajectory scale on 14% of clips (scale fail 3.6% → 14.3%) and g5 EPE rises 0.8 → 1.4 px
  (ViPE's poses are very accurate on ScanNet++). Candidate fix: geometric mean of ViPE depth and trajectory scale (E3b).
- Routing (`route_h`: hybrid unless the K gate fires) was dominated by `route_s` = v1 everywhere.

## E1d — QC rule on the new label (2026-10-18 01:30–01:50, CPU)
- Code: `e1_qc.py` (line-G rule unchanged) → `E1/qc_rule.csv`; `e1_qc2.py` (K-consistent partners) → `E1/qc_rule_v1b.csv`.
- Line-G rule on v0 / v1 labels (A+B, geometric failure = ATE-Sim3, FOV or metric depth failure): AUROC 0.77 / 0.75;
  keeps 63% / 66% with 12.5% / 11.9% failures among kept. On C it still rejects **60% for both labels**: the partners
  (unconditioned feed-forward models, SpatialVID's FOV) carry the FOV bias, so "disagreement" no longer means
  "label wrong". New K-consistent signals: depth vs K-conditioned partners (da3_pkvipe, pi3x_pkvipe) AUROC 0.76,
  ViPE-vs-others K disagreement 0.78, |label − SpatialVID FOV| 0.75, trajectory vs ViPE/SpatialVID 0.70.
  Rule v1b = max(xtraj_ba/3.0, xdepth_k/0.115, kdis/13.2) (thresholds = 90th pct of good A+B labels, in-sample):
  AUROC 0.80, keeps 66% with 11.9% failures, rejects 54% of C (indoor 10–20%, driving 70%, embodied 83%, hard 80%).
- Lesson: the earlier "80% of C rejected because of FOV" was mostly a property of the partner set; with a
  K-consistent label the residual rejections concentrate in driving / embodied / hard clips, where ViPE's
  trajectory itself disagrees with the others. Thresholds are in-sample (64 GT clips) — needs the user's review labels.

## Decision after E1 → pipeline v1
- Adopt: K gate (E1a) + Pi3X conditioned on the gated K as the main annotator (poses + depth), metric scale from
  ViPE depth median ratio; QC rule v1b plus the E1a K-disagreement flag. ViPE remains in the pipeline (K + scale).
- Not adopted: GeoCalib as K source; DA3 as main depth (shape worse); Pi3X's own metric scale; scale votes.

## E6b — controlled ViPE cost test (2026-10-18 00:50–01:55, GPU 3, ≈1.1 GPU·h)
- Code: `code/evolution/vipe_worker.py` (builds the ViPE pipeline once, loops over frame dirs), `e6b_run.sh`.
  Same 12 clips (6 A at 640×480 / 200 f, 6 C at 1920×1080 / 180 f), same GPU, sequential, other jobs on other GPUs.

| mode | total s (12 clips) | vs per-clip CLI |
|---|---|---|
| per-clip `vipe infer` (practice so far) | 920 | — |
| persistent worker | 711 | −23% |
| persistent worker, every 2nd frame | 458 | −50% |
| 2 persistent workers sharing one GPU (full frames) | 526 wall | −43% (throughput ×1.75) |

- The per-clip CLI pays ~15–20 s of process start / model loading per clip; ViPE leaves the GPU half idle, so two
  workers per H200 (≈10–27 GB each) almost double throughput. Accuracy of half frame rate: E6a (A unchanged, fast
  driving slightly worse).
- Decision: adopt persistent workers ×2 per GPU (no accuracy change). Half frame rate: adopt for static / slow
  categories only (indoor, walking), keep full rate for driving / drone — pending a larger check.

## E3b — compromise scale = geometric mean of ViPE depth-ratio and trajectory scale (2026-10-18 02:05, CPU)
- Why: E1c showed ViPE's dense depth and ViPE's trajectory disagree in scale by ~10% on many clips; v1 used the depth
  ratio (good metric depth, worse trajectory scale on ScanNet++), the trajectory scale does the opposite.
- Code: `e3_scale_vote.py` (new column k_vipe_td), `E3_KCOLS=vipe_d,vipe_t,vipe_td e3_eval.py` → `E3b/`.

| scale source for v1 | A scale fail | A depth fail | A SE3 fail | A AbsRel | B scale fail | ScanNet++ scale fail | ScanNet++ AbsRel |
|---|---|---|---|---|---|---|---|
| v0 hybrid (reference) | 28.6% | 7.1% | 10.7% | 0.061 | 69% | 3.6% | 0.042 |
| ViPE depth ratio (v1) | 32.1% | 7.1% | 0% | 0.062 | 77% | 14.3% | 0.045 |
| ViPE trajectory | 28.6% | 10.7% | 7.1% | 0.078 | 69% | 3.6% | 0.041 |
| **geometric mean (v1.1)** | **25.0%** | 10.7% | 7.1% | 0.069 | 62% | 7.1% | 0.041 |

- Counted over A+B+ScanNet++ metric clips: scale failures v0 18, depth-ratio 23, trajectory 18, geometric mean 17;
  metric-depth failures 7 / 7 / 9 / 8. Geometric mean = best scale, near-equal depth → adopted as v1.1 (small gain,
  small n; the metric-scale problem is NOT solved by re-weighting ViPE's own estimates).
- Ops note: `pkill -f snapshot_download` over ssh also matched the ssh session's own command line and killed it
  (same lesson as line E: never pkill -f with a pattern that appears in the calling command). HF downloads stall
  from this host; DA3METRIC-LARGE was fetched from ModelScope with code/filter/pdl.py (1.34 GB at 10 MB/s).

## E2 — split cut-rejected clips at shot boundaries instead of discarding them (2026-10-18 00:00–02:20, ≈2.3 GPU·h)
- Hypothesis (ideas.md #2): the cut rule rejects 31% of the pool; most of those clips contain usable single shots.
- Code: `code/evolution/e2_make_shots.py` (TransNetV2 boundaries from scores.json, ±3-frame guard, keep ≥ 60 frames
  = 4 s; frames are symlinks into the parent clip) → `data/filter_pool_shots/` (633 shots) + `E2/shots.csv`;
  `e2_run.sh` re-runs every filter stage + Qwen3-VL-32B + `decide.py` on the shots (45 min on one GPU) →
  `E2/decisions_shots.csv`; end-to-end check with the line-F protocol (`fc_setup.py`, 100 passed + 100 rejected
  shots, ViPE via the persistent worker + VGGT-Ω, `fc_eval.py`, `fc_analyze.py`) → `E2/check/`, `E2/fc/`.
- Side effect noted: stage logs of the shot run were appended to `outputs/logs/filter/<stage>.w0.jsonl` (shared with
  the pool run; rows are distinguishable by clip name `__s<k>`).
- Results:
  - 574 cut-rejected clips → 1764 shots, 633 kept (≥ 4 s) from 493 clips (66% of their frames).
  - Shot-level filter: 220 / 633 pass (35%), from 190 parent clips. Parents whose only reject reason was "cut":
    64% of their shots pass; parents also rejected by the VLM: 16%.
  - Yield: +220 clips / +23,530 frames on top of 668 passed clips (120,240 frames) → **+19.6% frames** (pool frame
    pass rate 36% → 44%). With a minimum shot length of 6 s: +147 shots / +15.2% frames; 8 s: +71 / +8.7%.
  - Quality (failure = ViPE crash, g5 self-consistency < 0.5, ViPE vs VGGT-Ω ATE > 10% or HFOV outside [20,130]):
    passed shots 26% (original passed clips 19.3%), rejected shots 40% (original 44%). By shot length:
    4–6 s 27%, 6–8 s 34%, 8–10 s 24%, 10–12 s 7%; ≥ 8 s → 15% (n=33, at the level of original clips).
- Decision: adopt splitting with a 6 s minimum (+15% frames) and let QC remove the extra failures; prefer ≥ 8 s
  when yield is not the bottleneck. Shorter shots carry less parallax — the main cause of the higher failure rate.

## E5a — swap ViPE's keyframe metric-depth prior for DA3METRIC-LARGE (2026-10-18 02:10–02:50, ≈1.3 GPU·h)
- Hypothesis: DA3 ranked first on metric depth in line E, so anchoring ViPE's BA with it should fix scale failures.
- Done: ViPE `pipeline.slam.keyframe_depth=dav3` (weights from ModelScope, `VIPE_DAV3_WEIGHTS`), dense post-alignment
  unchanged; persistent workers (3 on 2 GPUs) over A (51) + B (13) → `outputs/bench/eval3d_{A,B}/vipe_kfda3`;
  eval `code/evolution/e5_eval.py` → `E5/`. Median 52 s / clip.
- Results: A — ViPE scale fail 28.6% → 37.0%, FOV fail 19.6% → 26.0%, ATE Sim3 0.592 → 0.616%; v1.1 label with this
  ViPE as K/scale source: scale fail 25.0% → 33.3%. B — scale fail 69% → 85%. One crash (KITTI 07, "not enough known
  points"). A-group rank: v1.1 (default ViPE) 2.51 < v1.1 (DA3-prior ViPE) 2.89 < v0 3.13.
- Decision: rejected. A stronger single-image metric model does not translate into a better BA scale; ViPE's
  UniDepth-L prior stays. The planned "Pi3X depth as ViPE prior" (E5 proper) needs a multi-view adapter inside ViPE
  and is deprioritised given this negative result.

## E7 — cheap-first cascade: drop clips with feed-forward-only QC before running ViPE (2026-10-18 03:00, CPU)
- Hypothesis (ideas.md #3): image-only feed-forward disagreement predicts the final QC verdict, so ViPE can be skipped
  for hopeless clips.
- Code: `code/evolution/e7_cascade.py` → `E7/cascade.csv`. Pre-ViPE score = max(Pi3X-vs-FF trajectory, Pi3X-vs-FF
  depth, FF trajectory spread), each over the 90th pct of good A+B labels.
- Results: AUROC 0.71 for v1.1 geometric failure, 0.89 for the final v1b QC reject. At s = 2: skips 9% of A+B
  (no accepted clip lost) and 14% of C (3% of accepted C clips lost); at s = 1.5: skips 24% of C, loses 9%.
- Cost check: the cascade needs image-only Pi3X + VGGT-Ω/DA3 on every clip (≈ 27 GPU-s) to save ViPE (≈ 44 GPU-s)
  on 14–24% of clips (≈ 6–11 GPU-s) → **net more expensive**. Rejected for v1.2; becomes worth it only if the
  image-only passes are needed anyway (they are not, after E8).

## E8 — cheaper QC partners (2026-10-18 03:05, CPU)
- Why: rule v1b used SpatialVID (MegaSaM, minutes per clip) and pose-conditioned Pi3X as QC partners.
- Code: `code/evolution/e8_qc_cheap.py` → `E1/qc_rule_cheap.csv`. Partners: ViPE trajectory only, DA3 conditioned on
  ViPE cameras only, plus the K-disagreement flag.
- Results (A+B, v1.1-style label): AUROC 0.78 (v1b 0.80), keeps 64% with 12.2% failures (v1b 66% / 11.9%); C rejects
  54% (v1b 54%), same verdict as v1b on 97% of C clips. Thresholds: traj 3.27%, depth 0.194, K disagreement 13.2°.
- Decision: adopt (QC rule v1c). SpatialVID and pose-conditioned Pi3X leave the pipeline.

## E9 — literature hard thresholds as QC flags (ideas.md #6) (2026-10-18 03:15, CPU)
- Code: `code/evolution/e9_hard_rules.py` → `E9/flags.csv`, computed on the v1.3 label (K-gated Pi3X).
- Flags: HFOV ∉ [30°, 120°] (VGGT-Ω engine), adjacent-frame view change > 60° (Sekai) / > 20° (line G), ≥ 2 heading
  reversals > 150° within 10 s (Sekai), camera path < 2% of scene depth.
- Results: C hard vs normal flag rate — fov_out 10% vs 2%, jump60 20% vs 2% (same for jump20), reversal 50% vs 31%,
  tiny_move 0 vs 0. The hard "zoom" clips trigger none of them. A+B: jump60 fires on 1 clip (a GT failure), fov_out on
  none; reversal fires on 37.5% with no relation to failure (25% vs 30%).
- Decision: adopt `jump60` (precise, cheap) and `fov_out` as reject flags; reject `reversal` (noise from slow
  motion) and `tiny_move` (never fires with this scale); zoom stays with the filter's GeoCalib focal-trend rule.

## E6c — half-frame-rate ViPE in its new, reduced role (K + metric scale only) (2026-10-18 03:30, CPU)
- Code: `code/evolution/e6c_sub2_role.py` → `E6/sub2_role.csv` (re-uses outputs/qc/variants/*_sub2 ViPE runs).
- Results: HFOV difference sub2 vs full — median 0.19° / 0.09° / 0.61° (A / B / C), 90th pct 2.4° / 2.8° / 7.1°;
  v1.1 scale changes by > 10% on 18% / 46% / 25% of clips. A scale failures 25.0% → 28.6% (+1 clip of 28), B unchanged.
  In C the K difference exceeds 3° on 60% of driving and 50% of embodied clips, ≤ 20% elsewhere.
- Decision: half frame rate only for indoor / walking / slow categories (E6a agrees); full rate for driving, embodied,
  drone. Saves ≈ 35% of ViPE time on those categories (E6b).

## E10 — MapAnything with Apache-2.0 weights as a commercially usable annotator (ideas.md #9) (2026-10-18 03:40–05:10, ≈2.5 GPU·h)
- Setup: code via a GitHub proxy (github.com runs at ~10 KB/s from this host; sparse checkout of `mapanything/`),
  weights `facebook/map-anything-apache` from ModelScope (4.9 GB, 11.5 MB/s), venv `cache/ma-venv` (torch 2.7,
  aliyun mirror). Runner `code/evolution/ma_cond.py` (variants `img`, `kgate` = same gated K as v1.x; frames resized
  without crop to ≈255k px, `resize_mode=fixed_size`), eval `e10_eval.py` → `E10/`. 164 clips × 2 variants, 0 failures,
  ≈12–14 s / clip, peak 100 GB.
- MapAnything does not follow the conditioning K closely: output HFOV differs from the input by 3–7° on B.
- Results (v1.x label = pi3x_kgate~vipe for reference):

| group | metric | v1.x (Pi3X) | MA image-only | MA + gated K | MA + gated K + ViPE scale |
|---|---|---|---|---|---|
| A | ATE Sim3 % | 0.652 | 2.88 | 2.62 | 2.62 |
| A | FOV fail | 21.6% | 88% | 71% | 71% |
| A | pc F@5cm / traj-aligned | 0.813 / 0.670 | 0.61 / 0.20 | 0.60 / 0.38 | 0.60 / 0.38 |
| A | scale fail | 25.0% | 71% | 68% | 39% |
| ScanNet++ | ATE % / FOV err | 0.238 / 0.07° | 2.37 / 14.3° | 2.06 / 11.4° | — |
| C | g5 / g20 self-consistency | 0.889 / 0.792 | 0.742 / 0.569 | 0.745 / 0.562 | — |

- Decision: rejected as a replacement for Pi3X — far worse on every axis (A-group mean rank 4.3–6.0 vs 1.75).
  Caveat: integration not cross-checked against the official demo; the gap is large enough that a small integration
  error is unlikely to explain it, but a ScanNet++ FOV error of 11° even with K given is suspicious and worth one
  check before ruling the Apache model out. The commercial-licence question remains open (all strong depth / pose
  networks we use are non-commercial; ViPE itself is Apache-2.0).

## STATUS (2026-10-18 05:15) — every experiment so far
| id | what | status | decision |
|---|---|---|---|
| E0 | baseline v0 (lines E–H) | done | reference |
| E1a | K source selection, GeoCalib vs ViPE, gate rule | done | adopt gate; GeoCalib rejected |
| E1b | Pi3X / DA3 conditioned on K (kgt/kvipe/kgeo/pkvipe/kgate) | done | adopt K-conditioned Pi3X |
| E1c | Pi3X(K-gate) as main annotator, ViPE scale (v1) | done | adopted |
| E1d | QC rule on K-consistent label (v1b) | done | adopted, later replaced by v1c |
| E2 | split cut-rejected clips into shots, filter + end-to-end check | done | adopt, min 6 s |
| E3 | metric-scale votes | done | rejected (no vote fixes A and B) |
| E3b | geometric mean of ViPE depth / trajectory scale | done | adopted (v1.1) |
| E5a | ViPE with DA3-metric keyframe prior | done | rejected (scale worse) |
| E5 | Pi3X depth as ViPE prior (multi-view adapter) | not started | deprioritised after E5a |
| E6a/b/c | ViPE half frame rate, persistent workers, 2 per GPU, reduced role | done | workers ×2 adopted; half rate for slow categories |
| E7 | feed-forward-first cascade | done (offline) | rejected (net cost) |
| E8 | cheap QC partners (v1c) | done | adopted |
| E9 | literature hard thresholds | done | adopt view-jump > 60° and HFOV ∉ [30,120] flags |
| E10 | MapAnything Apache weights | done | rejected (much worse; integration caveat) |
| E11 | training validation with the pseudo labels | not started | needs days of GPU; next phase |
Current best: `pipeline_best.yaml` v1.3. Open: metric scale (A 25% fail), QC thresholds in-sample (need the user's
review labels), commercial-licence depth/pose model.

### v1.3 measured GPU cost per 12 s clip (180 frames, 1080p source), from cost.csv
- filter: rules ≈ 3.0 GPU-s (RAFT 2.0 + GeoCalib 1.0 + TransNet) + 2.9 CPU-s, Qwen3-VL-32B 0.57 GPU-s → ≈ 3.6 GPU-s
  per input clip; ≈ 42–44% of input frames survive with shot splitting.
- annotation per accepted clip: ViPE 43.8 GPU-s (2 workers / GPU; ≈ 30 with half rate on slow categories) +
  Pi3X K-conditioned 19.3 + DA3 on ViPE cameras 16 (QC partner) + DA3 image-only 8.8 and GeoCalib 1.1 (K gate /
  disagreement) ≈ 89 GPU-s (≈ 75 with half rate where allowed).
- QC v1c keeps ≈ 46% of wild-like accepted clips (C) / ≈ 64% of benchmark clips.
- ⇒ per 1M input clips: 1M × 3.6 s + 0.44M × ≈ 85 s ≈ 1.1k + 10.4k ≈ 11.5k H200-hours ≈ 60 days on 8 H200,
  yielding ≈ 0.2M QC-passed 12 s clips (≈ 36M frames) at the C-group acceptance rate.

## E13 — dual-gate QC: cross-model gate 1 AND model-independent self-check gate 2 (2026-10-20 17:15–17:50, ≈0.35 GPU·h on GPU 7)
- Hypothesis: the cross-model gate is blind to correlated errors (all models wrong the same way). A second gate that uses only
  the label itself plus the video should catch some of what gate 1 passes, and so lower the kept failure rate.
- Code: `code/evolution/E13_dualqc/` with these scripts:
  - `e13_cache.py`: RAFT/LightGlue cache for ScanNet++.
  - `e13_signals.py`: gate-2 signals for the v0/v13 composites; it reuses `qc_signals.signals`.
  - `e13_gate1_S.py`: v0 rule and v1c on ScanNet++.
  - `e13_analysis.py`: calibration, operating points, C group, plot.
  - `e13_fsweep_pi3x.py` / `e13_fsweep_eval.py`: focal sweep.
  - `e13_report.py`: `tables.md`.

  Outputs: `E13/{tables.md, raw/*.csv, retention_dual.png, fsweep_curves.png, summary.json}`.
- Gate-2 signals:
  - Sampson error of the LightGlue matches w.r.t. the label's F;
  - induced flow vs RAFT (relative EPE, <3 px share) on static, fb-consistent pixels;
  - multi-gap depth self-consistency (cons20, occlusion-aware cons5_occ);
  - free-space violation (fsv1/fsv5).

  Five rule families × six calibrations (per-signal Youden; per-signal 90/95/98th pct of good labels; common quantile
  for a union false-reject of 5% / 10%) were run as LODO on the 476 pool. The thresholds fitted on the whole pool were
  then applied unchanged to the composites: v0 (gate 1 = v0 rule) and v1.3 (gate 1 = v1c) on A, B and ScanNet++
  (S, newly scored, 28 clips), plus C.
- Pre-declared adoption criterion (defines "unreasonable"): dual vs gate 1 must lower the kept geo-failure rate by
  ≥ 20% relative AND ≥ 1.5 pp. It must also keep ≥ 90% of what gate 1 keeps and add ≤ 5 pp false rejects of good
  labels. Both the pool (LODO) and v1.3 A+B+S must pass.
- Chosen gate 2 (best pool variant within budget): reject if max(fsv1/0.122, (1−cons5_occ)/0.522) > 1
  (`fsv_cons`, fr10).
- Single-signal AUROCs on the pool:
  - geo: iflow5_rel 0.82, fsv5 0.74, cons20 0.74.
  - pose: iflow1_lt3 0.82, samp5_med 0.79.
  - fov: everything ≤ 0.77 except iflow5_rel.
  - depth: cons20 0.78.
- Results:

| set | gate | kept | geo fail kept | pose / fov / depth kept | false reject (good) |
|---|---|---|---|---|---|
| 476 pool (LODO) | gate 1 (v0 rule) | 35.9% | 8.8% | 1.2 / 5.8 / 1.8% | 24.3% |
| | gate 2 | 74.6% | 48.2% | 7.0 / 35.2 / 26.2% | 10.7% |
| | dual | 34.2% | 8.0% | 1.2 / 5.5 / 1.2% | 27.2% |
| v0 composite A+B+S (n=92, base 20.7%) | gate 1 | 73.9% | 2.9% | | 9.6% |
| | gate 2 | 94.6% | 17.2% | | 1.4% |
| | dual | 72.8% | 3.0% | | 11.0% |
| v1.3 composite A+B+S (n=92, base 19.6%) | gate 1 (v1c) | 75.0% | 5.8% | 0 / 1.4 / 4.3% | 12.2% |
| | gate 2 | 96.7% | 18.0% | | 1.4% |
| | dual | 73.9% | 5.9% | 0 / 1.5 / 4.4% | 13.5% |

  - On ScanNet++ both gates pass all 28 clips for both composites; the one failure (depth) is not caught.
  - Strictest variant (5 signals, per-signal Youden):
    - v1.3: dual keeps 55.4% with 2.0% failures (vs 75.0% / 5.8%), 32% false rejects.
    - Pool: dual keeps 22.9% with 5.5% failures.

    It cuts failures, but at a 26–36% relative retention loss.
  - Why gate 2 adds so little: 9 of the 15 failures that gate 1 keeps in the pool are FOV failures
    (vKITTI2 forward driving, 7scenes/TUM with Pi3X). Their self-consistency is excellent (cons5_occ 0.9–0.98,
    iflow1_lt3 0.94–0.98). A wrong focal length plus compensating depth and pose stays self-consistent, so gate 2
    cannot see the main error that gate 1 misses.
- C group (no GT):
  - v1.3: gate 1 rejects 54%, gate 2 rejects 1% (one DROID wrist clip, already rejected by gate 1), dual 54%.
    Gate 2 adds 0 rejects.
  - Pool (7 methods): gate 1 81%, gate 2 20%, dual 82% (1 extra label).
  - Hard-case AUROC of gate 2 is 0.45 (v1.3) and 0.37 (pool): hard clips look *more* self-consistent.
    Static camera, synthetic zoom and high-altitude aerial clips give cons5_occ 0.95–1.0 and fsv1 ≈ 0, because
    degenerate geometry is trivially consistent.
  - Hard cases caught by gate 2 but missed by gate 1: none. Missed by both: nNjYd1Qh1Ws (aerial) and uO99uLuqisc
    (zoom). None of the normal clips flagged by line G (suspected cut / zoom / static) is caught by gate 2.
- BTbkSEotmX0, single-label gate 2:
  - Rule scores: ViPE 0.45, v0 0.55, Pi3X 0.60, VGGT-Ω 0.68, v1.3 0.69, VGGT 0.91, SpatialVID 0.95, Voyager 1.06,
    DA3 1.40.
  - The individual signals split: BA labels are better on free space and flow, feed-forward labels on cons5/cons20.
  - Not conclusive per label, because the labels come from different models.
- BTbkSEotmX0, focal sweep (the decisive test): I re-solved v1.3 (Pi3X conditioned on K) with K × s, s = 0.45–1.6,
  and computed the self-check plus the photometric warp error.
  - Controls with GT (KITTI 00/05, GT 81.6/81.8°; Pi3X image-only says 69–71°): the self-checks are best at
    83–90° (photo5 83/90°, iflow5_epe 83/90°, cons20 83/84°, fsv1 83/84°). The sweep finds the true FOV
    within about +2 to +8° and overrides Pi3X's own prior.
  - vKITTI2 Scene01 (GT 81°): 81–87° on most signals. Scene06 (almost static) is flat, so the FOV cannot be observed.
  - BTbk: photo5 is best at 53°, photo1 at 61°, Sampson at 61°, fsv1 at 46°; iflow5_epe, cons5_occ and cons20 are
    best at the 67° sweep edge. Median argbest is 61°. Every signal gets worse towards ViPE's 33° (photo5 14.8 vs 13.7,
    cons20 0.53 vs 0.59–0.61).
  - ⇒ Model-independent evidence that the true HFOV is ≈ 50–65°: the feed-forward models are right and ViPE /
    SpatialVID (33–35°) are wrong. The shipped v1.3 label (ViPE K, which passes the [25,130] gate) therefore has a
    wrong K on this clip, as the gate-1 rejection (1.95) already suggested.
- Cost (production, per clip):
  - Cache: RAFT fw/bw gaps 1/5 + SuperPoint/LightGlue gaps 5/20 at 512 px = 0.58 + 0.81 ≈ 1.4 GPU-s, plus 0.5 CPU-s
    of frame reads.
  - Signals: 7.1 CPU-s median per label (4.2 s wall under 24-way load).
  - ⇒ ≈ 1.4 GPU-s + 7.6 CPU-s ≈ +1.6% of the v1.3 annotation GPU budget.
  - Focal sweep: 11 × 18.8 GPU-s ≈ 207 GPU-s per clip, offline/diagnostic only. A 3-point version (s = 0.6/1/1.6)
    would cost ≈ 57 GPU-s.
- Decision: **gate 2 not adopted as a reject gate; pipeline_best.yaml stays v1.3**.
  - Pool: −0.8 pp failures (−9% relative), 4.7% retention loss.
  - v1.3: −0.1 pp, i.e. no gain; 1.4% retention loss.
  - Both are far below the 20% / 1.5 pp bar. The only variants that do lower failures (Youden) cost > 25% retention.
  - Self-consistency is mostly redundant with the cross-model rule, and it is blind exactly where gate 1 is weak
    (FOV under forward motion, static/zoom/low parallax).
  - What does work, and is worth a follow-up (E14 idea): the **focal sweep** as an independent K check, triggered only
    when BA and feed-forward HFOV disagree (kdis > 13.2°, or driving / forward motion). It located the GT FOV on
    KITTI and resolves BTbk in favour of ≈ 55–60°. Next step: a 3-point sweep with a parabolic fit, validated on all
    KITTI/vKITTI2/TartanAir clips, then a decision on whether to override ViPE K.
- Failures / caveats:
  - The composite sets are small (92 labels, 18 failures), so one label is about 1.1 pp.
  - Gate-2 thresholds are fitted on the single-method pool and applied to the composites (domain shift, deliberate).
  - iflow5_rel has a heavy tail on good labels (98th pct ≈ 300), so quantile-calibrated rules containing it switch it
    off; Youden is unaffected.
  - The first photometric implementation crashed on `cv2.remap` (> SHRT_MAX points) and was replaced by numpy
    bilinear sampling.
  - The line-G cache is reused through symlinks in `E13/cache_links`; the ScanNet++ cache is new (`E13/cache/S`).
    Nothing under `outputs/qc` was modified.

## E12 — MoGe-2 / MoGe-3 (MIT weights) as metric-scale source, scale voter and depth source (2026-10-20 16:20–19:10, ≈3.2 GPU·h on GPUs 4–5 + CPU eval)
- Hypothesis: a single-image metric geometry model with a commercial-friendly licence (a) gives a per-clip metric scale
  that fixes part of v1.3's A-group scale failures (25%), alone or as a voter, and (b) can replace Pi3X depth in the
  label (licence-clean path), with K-conditioning (fov_x from the gated K) closing most of the gap.
- Setup: code `microsoft/MoGe` @74fbce0 (v3.0.0, GitHub direct, ~5 MB/s today) in `code/MoGe`; venv `cache/moge-venv`
  (torch 2.13.0+cu130, triton 3.7.1 — FlexGEMM fails to compile under torch 2.7 / triton 3.3); weights from HF
  `Ruicheng/moge-2-vitl-normal`, `moge-3-vitl`, `moge-3-vitg`. Runner `code/evolution/E12_moge/moge_run.py`
  (claim files, chain `run_chain.sh`, 1 worker per GPU): frames resized without crop to the pi3x_kgate depth grid
  (~250k px), official defaults (resolution_level 9, fp16, mask -> NaN, v3 refine_steps 3), batch 16; modes `free` and
  `kgate` (fov_x = HFOV of the gated K in pi3x_kgate/pred.npz). All frames of A (no ETH3D, 51 clips), B (13),
  ScanNet++ (28), C (72) = 27,134 frames × 6 runs, 0 failures. Outputs `<bench>/moge_<m2l|m3l|m3g>_<free|kgate>/`.
- Eval: `e12_scales.py` (per-clip scale sources; v1.3 rule copied from e10_eval.scale_to and reproduces E3b exactly:
  A 25.0%, vipe_d 32.1%), `e12_scale_report.py` (analytic: scaling Pi3X by k turns its GT Sim3 scale s into s/k),
  `e12_eval.py` (run_hybrid machinery, same metrics as eval3d_bench / E3b / E10; 30 methods), `e12_finalize.py`.
  MoGe scale source = pi3x_kgate × median over ≤40 frames of median(D_moge)/median(D_pi3x).
- Results — metric scale (fail = |scale−1| > 10%, metric clips A 28 / B 13 / ScanNet++ 28):

| scale source | A fail | B fail | ScanNet++ fail | A med err |
|---|---|---|---|---|
| v1.3 sqrt(ViPE d·t) | **25.0%** | 61.5% | 7.1% | 6.5% |
| ViPE own trajectory | 28.6% | 69.2% | 3.6% | 6.6% |
| Pi3X raw | 67.9% | 15.4% | 10.7% | 14.0% |
| DA3 depth ratio / own traj | 35.7% / 35.7% | 53.8% / 84.6% | 57.1% / 64.3% | 8.2% |
| MoGe-2 L kgate | 53.6% | 23.1% | 7.1% | 10.7% |
| MoGe-3 L kgate | 64.3% | 23.1% | 10.7% | 12.6% |
| MoGe-3 g free | 60.7% | 30.8% | 3.6% | 12.1% |
| **MoGe-3 g kgate (best MoGe)** | 46.4% | 30.8% | 3.6% | 8.6% |
| median(v1.3, MoGe-3g-k, DA3) | 35.7% | 30.8% | 7.1% | 7.4% |
| sqrt(v1.3 · MoGe-3g-k) | 32.1% | 38.5% | 3.6% | 6.9% |
| median(ViPE d, ViPE t, MoGe-3g-k) | 28.6% | 46.2% | 3.6% | 6.7% |
| agree gate MoGe-3g-k vs v1.3 within 10% (else flag) | 22.2% of kept, flags 35.7% | 25% kept, flags 69% | 3.8%, flags 7% | C flags 77.8% |

  - Free FOV hurts MoGe scale (MoGe-3 L free 71% on A); K-conditioning is essential.
  - Per dataset the picture is mixed: on 7-Scenes MoGe-3g-k fails 14% vs v1.3 43% (its depth scale is within 2% of GT
    where v1.3 is +10%), but on TUM it underestimates depth by ~11% (75% fail vs v1.3 0%) and KITTI stays at 60%.
    Frame-to-frame scale flicker of MoGe: median std(log ratio) 0.04–0.09.
  - B (vkitti2/tartanair, in Pi3X/DA3 training data): MoGe is much better than v1.3 (23–31% vs 61.5%), mainly
    vkitti2 (MoGe-3g-k 14% vs v1.3 86%) — same A-vs-B conflict as E3: nothing tested improves both.
  - C: MoGe and v1.3 disagree by a median 24–35% on wild clips (flag rate at 10%: 78%), so the agree gate is unusable.
- Results — single-image depth (median over clips; A): metric AbsRel MoGe-3g-k 0.075 (Pi3X raw 0.095, DA3 0.076,
  ViPE 0.114, MoGe-2 L-k 0.093), but seq-aligned AbsRel ≈0.10 for all MoGe vs Pi3X 0.046 (δ1 0.92 vs 0.97). Free FOV
  error: A 5.1° / 5.2° / 4.5° (MoGe-2 L / 3 L / 3 g; FOV fail 53 / 55 / 41%), B 6.1 / 3.9 / 3.2°, ScanNet++ 2.0 / 1.8 /
  1.2° (ViPE 1.2° on A) — MoGe is not a useful K source either.
- Results — MoGe as depth source (v1.3 cameras + gated K; D = one scale per clip, Df = per-frame scale; VD = ViPE
  cameras, the Pi3X-free path):

| A | metric AbsRel | pc F@5cm | pc_traj F | g5 / g20 EPE px | depth fail |
|---|---|---|---|---|---|
| v1.3 (Pi3X depth) | 0.069 | **0.813** | **0.670** | **4.06 / 7.03** | 10.7% |
| D: MoGe-3g-k | 0.094 | 0.629 | 0.643 | 4.71 / 9.50 | 14.3% |
| Df: MoGe-3g-k | 0.101 | 0.648 | 0.618 | 4.47 / 9.66 | 17.9% |
| Df: MoGe-2 L-k | 0.090 | 0.662 | 0.648 | 4.48 / 8.30 | 10.7% |
| ViPE alone | 0.114 | 0.493 | 0.544 | 5.62 / 11.2 | 14.3% |
| VD: ViPE cams + MoGe-3g-k | 0.081 | 0.629 | 0.602 | 5.78 / 11.8 | 10.7% |

  ScanNet++: v1.3 0.982 / 0.942 / 1.40 / 1.90 vs Df MoGe-3g-k 0.922 / 0.898 / 1.41 / 2.06 and VD 0.856 / 0.723 /
  0.84 / 2.23 (ViPE alone 0.500 / 0.494 / 1.32 / 3.63). C g5 / g20 self-consistency: v1.3 0.889 / 0.792, Df MoGe-3g-k
  0.731 / 0.495, VD 0.660 / 0.379, ViPE 0.772 / 0.369.
- Cost on H200 (batch 16, ~250k px, 180-frame clip): MoGe-2 L 4.2 s (23 ms/frame, 19 GB), MoGe-3 L 11.0 s
  (61 ms/frame, 19 GB), MoGe-3 g 17.3 s (96 ms/frame, 39 GB); plus a one-off FlexGEMM autotune per process (~15 s).
  Pi3X K-conditioned is 19.3 s per clip for comparison.
- Licence: MoGe code MIT (DINOv2 part Apache-2.0); HF model cards of all three weights say MIT; FlexGEMM,
  utils3d-moge and pipeline are MIT. This is the first strong depth model in the stack with permissive weights (still
  worth a legal check of the training-data terms).
- Decision: **reject** MoGe as scale source / voter in v1.3 (best A result 28.6–32.1% vs 25.0%; the agree gate only
  trades failures for flags and flags 78% of C). **Reject** MoGe depth in place of Pi3X depth (A pc F −0.15 to −0.18,
  seq AbsRel 2×, C self-consistency −0.16). **Pending** as the depth part of a licence-clean line (ViPE cameras/K +
  MoGe-3 g K-conditioned depth): better than ViPE alone on every depth metric (A pc F 0.49→0.63, metric AbsRel
  0.114→0.081; ScanNet++ pc F 0.50→0.86) but its scale is still ViPE's and it remains far below v1.3.
  pipeline_best.yaml unchanged (v1.3).
- Failures / lessons: (1) MoGe-3 needs torch ≥ 2.9-era Triton (FlexGEMM `tl.const` compile error under triton 3.3);
  (2) single-image metric scale has dataset-specific biases of ±10% (TUM −11%, KITTI −7–12%, 7-Scenes ≈0) that do not
  cancel those of ViPE, so voting does not help — the scale problem needs multi-view metric evidence, not another
  monocular voter; (3) the first CPU eval launch (24 workers without thread caps) pushed host load to ~540 — rerun with
  OMP/BLAS threads = 3 and 20 workers (`run_eval.sh`); always cap threads on this shared host.

## E15 — GLR track: GLRMap annotator behind the 3DData_anno filter and QC; sandbox evidence on easy web-style clips (2026-10-29, CPU only)
(status: running; the criteria below were written before the silver-reference and variant results were looked at)
- Question: can GLRMap (SfM + RoMa + PDA, non-metric) be put between the v1.3 front end (rules + VLM judge) and the v1.3 back end (cross-witness
  QC v1c + flags) so that a batch can be run without a human, with v1.3 as the fallback? What can be said without a GPU, without the GLRMap code
  and without the user's servers?
- Code: `pipeline/{run_batch_glr.sh, glr_prepare.py, glr_worker.sh, glr_stage1_gate.py, glr_collect.py, glr_merge.py, glr_common.py}`,
  `configs/pipeline_glr_v0.yaml`, `tests/glr/` (design and contracts: `docs/GLR_INTEGRATION.md`; server runbook for what the sandbox cannot do:
  `docs/GLR_PHASE0_RUNBOOK.md`). Sandbox-only helpers: `tools/stage1_sfm.py` (pycolmap replica of GLRMap stage 1), `tools/filter_stage_cpu.py`,
  `tools/fetch_ood_set.py`, `evolution/E15_glr/` (stand-ins for the dense stage and the witnesses, demo driver, sweep analysis).
- Clip set: 15 easy web-style clips (Mixkit / Pexels / Internet Archive; a static scene seen by a moving camera, 12 s) and 15 hard negative
  controls (static camera, pan only, driving, night, crowd, misc.), recipe and licences in `docs/experiments/E15_ood_manifest.json`.
- Pre-declared definitions for the sandbox sweep (replica of GLRMap stage 1 at reduced cost: 640 px long side, 60 frames = every 3rd frame of the
  15 fps clip, sequential matching overlap 12, unguided; the silver reference = every 6th frame, exhaustive matching, SIMPLE_RADIAL, incremental
  mapper):
  - a replica run is **silver-accurate** iff >= 90 % of its frames registered AND |f - f_ref| <= 10 % of f_ref AND Sim(3) ATE to the silver
    reference (common frames) <= 3.27 % of the reference path (the v1c threshold of E8). The silver reference is a second SfM reconstruction of the
    same frames, not ground truth: "silver-accurate" means "agrees with an independent exhaustive reconstruction";
  - a run that passes the stage-1 gate (verdict ok or suspect) but is not silver-accurate is a **silent error**; a run that the gate stops although
    it is silver-accurate is a **false stop**;
  - nothing in `configs/pipeline_glr_v0.yaml` is tuned on these clips; the gate thresholds were fixed before any of them was run.
- Known limitation of the replica (stated up front): it is harsher than the real stage 1 (full resolution, ~100 frames, guided matching), so signals
  that depend on resolution (points per image, planar-pair fraction) are NOT comparable with the evaluation sessions' numbers; clips that stop on
  them are rerun at 960 px (`v0a_hq960`) before anything is concluded.
