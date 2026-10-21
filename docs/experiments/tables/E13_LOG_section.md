
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
