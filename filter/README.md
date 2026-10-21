# Layer ① front-end video filter (rules + expert models + VLM)

Line F of the 3D-annotation pipeline: decide, before any geometric annotation, which 12 s clips of
CC-BY YouTube video are worth annotating. Every stage writes scores (not only pass / reject) into
`data/filter_pool/<clip>/scores.json`; `decide.py` turns the scores into `decisions.csv`.

All numbers below are from the run of 2026-10-17 (pool = 1835 clips from 436 CC-BY videos).

## Layout

| Path | Content |
|---|---|
| `data/filter_pool/pool.json` | index of the pool (one meta record per clip: URL with `&t=`, channel, license, query, category hint) |
| `data/filter_pool/<clip>/` | `clip.mp4` (≤1080p, 15 fps, 12 s), `frames/` (180 JPEG, 960 px wide), `sheet.jpg` (6 frames), `meta.json`, `scores.json` |
| `data/filter_pool/_calibC/` | group C (72 clips) mirrored in the same layout for calibration (frames are symlinks) |
| `data/filter_pool/candidates.json`, `picks.json` | search results (921 videos, 553 pass the metadata prefilter) and the selected 444 videos with start times |
| `data/filter_pool/decisions.csv` | final decision per clip: `decision`, `reason`, `soft_flags`, every score column |
| `outputs/filter/report/` | calibration and comparison tables (`calibC_pr.csv`, `vlm_agreement.csv`, `vlm_vs_rules.csv`, `summary.json`) |
| `outputs/filter/report/fc/` | end-to-end check (ViPE + VGGT-Ω no-GT QC vs. filter decision) |
| `outputs/filter/mosaic_pass.jpg`, `mosaic_reject.jpg` | 40 random passed / rejected contact sheets for the visual check |
| `outputs/bench/filter_check/` | ViPE and VGGT-Ω outputs for the end-to-end check |
| `outputs/logs/filter/` | per-stage logs, one JSON line per clip with its runtime (`<stage>.w*.jsonl`), VLM run stats (`vlm_<tag>.json`) |

## Scripts (`code/filter/`)

| Script | Env | What it does |
|---|---|---|
| `pool_search.py` | `cache/yt-venv` | YouTube search with the CC-BY filter, ~60 queries in 12 categories, metadata prefilter (CC license, ≥1080p, 2–60 min, not Shorts), random 2–5 s sleeps, back-off on rate limiting |
| `pool_clip.py select / cut / index` | `cache/yt-venv` | balanced selection (per-category quota, ≤3 videos per channel, dedup vs. group C and the first 12 YouTube clips); one yt-dlp call per video fetching all 3–5 sections (10–90 % of the video), hard timeout per video; frames + sheet + meta; `pool.json` |
| `s1_cuts.py` | `cache/filter-venv` | TransNetV2 (transnetv2-pytorch 1.0.5) + ffmpeg scene score |
| `s2_rules.py` | `cache/filter-venv` | luminance, Laplacian sharpness, LAION aesthetic (CLIP ViT-L/14 + the improved-aesthetic MLP SpatialVID uses) |
| `s3_ocr.py` | `cache/ocr-venv` | PaddleOCR PP-OCRv5 text detection on 3 frames: total / bottom-band / corner / static text area |
| `s4_flow.py` | `cache/filter-venv` | RAFT-large flow: motion magnitude, static ratio, spikes, jerk; homography vs. epipolar fit (parallax, pure-rotation share, dynamic-area fraction); flow divergence |
| `s5_geocalib.py` | `code/vipe/.venv` | GeoCalib (ViPE's copy, `distorted` weights, simple-radial) on 12 frames: focal stability (zoom), k1 (lens distortion) |
| `s6_track.py` | `cache/filter-venv` | CoTracker3 offline: track-loss rate, largest one-step visibility drop, track jumps |
| `vlm_run.py` | `cache/vllm-venv` (vLLM 0.30.0) | one shared prompt / JSON schema for every VLM, 12 frames (long side 640) |
| `decide.py` | `cache/filter-venv` | rules (+ VLM) → `decisions.csv` |
| `analyze.py` | `cache/filter-venv` | P / R on group C, VLM–VLM and VLM–rule agreement, rejection rates, stage runtimes |
| `fc_setup.py`, `fc_extend.py`, `fc_eval.py`, `fc_analyze.py` | `cache/eval-venv` | end-to-end check: bench construction, no-GT QC (copy of `code/bench/yt_eval.py`), failure prediction |
| `make_mosaic.py` | `cache/filter-venv` | the two overview mosaics |
| `run_stage_loop.sh`, `run_vlms_pool.sh`, `dl_vlms_pdl.sh`, `pdl.py`, `setup_venvs.sh`, `make_calibC.py` | | orchestration, downloads, venv setup |

## Results (2026-10-17)

### Pool
1835 clips (12 s, 15 fps) from 436 CC-BY videos (921 searched, 553 passed the metadata prefilter, 444 selected,
6 needed sign-in, a few sections failed). Categories (search hint): indoor_static 271, outdoor_static 247,
embodied 244, indoor_dynamic 235, outdoor_dynamic 223, driving 199, hard_edited 91, hard_cg_game 83,
hard_static_cam 83, hard_zoom 72, hard_lowlight 60, hard_far_aerial 27. Deviations: 436 videos instead of
~400 (≤3 videos per channel limited the hard categories), 1835 clips instead of ~2000.

### Expert stages (H200, one process; `outputs/logs/filter/<stage>.w*.jsonl`)
| Rule (reject) | Threshold | Pool reject rate | Stage runtime |
|---|---|---|---|
| cut | TransNetV2 max prob > 0.5 | 31.3 % | 1.7 s/clip (TransNet + ffmpeg) |
| dark / overexposed | mean luma < 20 / > 200 (SpatialVID 20–140) | 1.6 % / 0.7 % | rules 0.19 s/clip |
| blur | median Laplacian var < 25 (960 px) | 1.6 % | (rules) |
| text_overlay | OCR area > 0.3 (SpatialVID) or static text > 2 % | 6.1 % | OCR 1.0 s/clip (CPU) |
| static_camera | > 50 % of 0.2 s steps with flow < 0.25 % width (DynPose-style) | 16.7 % | flow 2.0 s/clip |
| too_dynamic | > 60 % grid points off the epipolar model | 6.6 % | (flow) |
| zoom | GeoCalib focal end/start > 1.4 or < 0.6 and \|Spearman\| > 0.8 | 2.1 % | GeoCalib 1.0 s/clip |
| fisheye | \|k1\| > 0.3 | 2.2 % | (GeoCalib) |
| any rule | | 49.8 % | ≈ 6 s/clip total |

Soft flags only (too many false positives on group C): `no_parallax` (homography explains flow; fires on planar highway
driving), `cut_track` (CoTracker visibility drop; fires on fast occluders), `low_aesthetic` (< 4; all DROID clips).
TransNetV2 at 0.5 is precise: visual checks of borderline (0.5–0.8) detections showed real dissolves, fades and hard cuts.

### VLMs (vLLM 0.30.0, 12 frames, one prompt, no guided decoding; 1×H200 each)
| Model | Weights GPU mem | Clips / GPU-h | JSON parse | C P/R (revised) alone | rules OR VLM P/R |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 17.0 GiB | ~17 000 | 100 % | 0.56 / 0.21 | 0.80 / 0.83 |
| **Qwen3-VL-32B** | 62.9 GiB | ~6 300 | 100 % | 0.67 / 0.58 | 0.72 / 0.88 |
| Qwen3.5-27B | 51.1 GiB | ~8 700 | 100 % | 0.61 / 0.58 | 0.67 / 0.83 |
| InternVL3.5-8B-HF | 15.9 GiB | ~6 700 | 100 % | 0.50 / 0.04 | 0.91 / 0.79 |
Rules alone on C: 0.95 / 0.79 (revised labels), 0.35 / 0.70 (original `hard_case` labels; most "false positives"
are unlabelled cuts / dissolves / static cameras, listed in `analyze.py: REVISED_EXTRA`).
Pool agreement on `reconstructable`: Qwen3-VL-32B ~ Qwen3.5-27B 74.6 % (κ 0.58), 8B ~ 32B 68.6 % (κ 0.43),
InternVL vs. others κ ≈ 0 (it answers "yes" almost always). Field agreement with rules (32B): has_cuts 85 %,
static 90 %, low light/blur 89 %, overlays 79 %, zoom 98 %. VLMs see cuts but still answer "yes"; they add
night / rain / extreme-dynamics rejections that rules miss and catch static robot cameras; they miss far aerial
shots and synthetic zoom.
InternVL3 replaced by InternVL3.5-8B-HF (InternVideo3-8B exists only as an unofficial upload). Qwen3.5-122B-A10B-FP8
and Qwen3-VL-235B-FP8 were not run: the host downloads at ~13 MB/s total (365 GB ≈ 8 h).

### End-to-end (ViPE + VGGT-Ω on 430 pool clips: 300 random + 130 to cover rare reasons)
Failure := ViPE crash, or ViPE g5 self-consistency < 0.5, or ATE(ViPE~VGGT-Ω) > 10 % of trajectory, or HFOV
outside 20–130°. Failure rate: passed 19.3 % vs rejected 44.0 % (random-300 subset: 12 % vs 41 %).
ATE ViPE~VGGT-Ω median 0.45 % (pass) vs 2.4 % (reject). Failure rate by reason: too_dynamic 95 %,
vlm:no_inaccurate 56 %, text_overlay 50 %, cut 47 %, blur 44 %, vlm:no_unreconstructable 32 %, static_camera 14 %
(no-GT QC cannot see that a static camera gives no usable geometry — reject it anyway). Best single predictors
(AUROC): dynamic-area fraction 0.78, flow magnitude 0.74, CoTracker visibility drop 0.72, GeoCalib focal
variation 0.70.

### Recommended pipeline
metadata prefilter → TransNetV2 cuts → luma / blur / OCR → RAFT flow (static, dynamic-area) → GeoCalib (zoom,
fisheye) → Qwen3-VL-32B `reconstructable` (reject = any rule OR VLM ≠ yes). Thresholds as in the table and `decide.py`.
Qwen3-VL-32B is the best single VLM on group C and its verdicts are the most predictive of ViPE failure among
the VLMs at similar precision; Qwen3-VL-8B is the cheap alternative (3× faster, lower recall); InternVL3.5-8B is not
useful. Pass rate on the pool: 668 / 1835 (36 %).

### Problems
* pypi.org ~60 KB/s from the host → venvs built from the aliyun mirror; HF ~4 MB/s → ModelScope with `pdl.py`.
* YouTube soft rate limit after ~300 metadata requests (search backs off); in-process yt-dlp section downloads hung
  for hours → yt-dlp CLI subprocess with a hard timeout.
* vLLM + InternVL3.5 hangs in pinned-memory allocation when the engine is forked after the parent's multimodal
  warm-up touched CUDA → `VLLM_WORKER_MULTIPROC_METHOD=spawn`; portrait clips need `VLM_MAXLEN=40960` for InternVL.
* `vipe_bench.py` loses workers to an `mp.Queue.get_nowait` race → `fc_vipe_shard.sh`.
* `_raw/` (18 GB of downloaded sections) is kept; it can be pruned.


### Large VLMs (added 2026-10-17 evening; `outputs/filter/report/big_compare.json`, `big_blindspots.csv`, `fc_235b/`, `fc_122b/`)
Downloads (ModelScope, `dl_big.sh` + `pdl.py`, ~12 MB/s): Qwen3-VL-235B-A22B-Instruct-FP8 5 h 26 min, Qwen3.5-122B-A10B-FP8 3 h 06 min.
FP8 MoE needs `VLLM_USE_DEEP_GEMM=0` (DeepGEMM JIT requires nvcc ≥ 12.9; set in `run_vlms_pool.sh`).

| | Qwen3-VL-32B | Qwen3-VL-235B-FP8 (TP4) | Qwen3.5-122B-A10B-FP8 (TP2) |
|---|---|---|---|
| Weights per GPU | 62.9 GiB ×1 | 56.0 GiB ×4 | 58.2 GiB ×2 |
| Clips / GPU-h (whole pool) | ~6 300 | ~3 300 (pool in 9 min) | ~7 750 |
| κ vs 32B on `reconstructable` (pool / filter_check) | – | 0.61 / 0.61 | 0.58 / 0.64 |
| C revised P/R alone | 0.67 / 0.58 | 0.46 / 0.63 | 0.55 / 0.50 |
| C revised P/R rules OR VLM | 0.72 / 0.88 | 0.51 / 0.83 | 0.65 / 0.83 |
| "has_cuts but yes" (pool) | 125 | 13 | 0 |
| far aerial ×2, synthetic zoom b06 | yes (missed) | yes (missed) | yes (missed) |
| Final pass rate (pool) | 36.4 % | 29.2 % | 35.9 % |
| filter_check ViPE failure, passed / rejected | 19.3 % / 44.0 % | 15.5 % / 41.6 % | 19.9 % / 42.4 % |

Conclusion: keep Qwen3-VL-32B as default. 235B is stricter (fewer passes, −4 pt failure among passed) and
fixes "sees the cut but says yes", but costs 4 GPUs, half the per-GPU speed and loses precision on group C.
Qwen3.5-122B is fast and internally consistent (never says yes after seeing a cut) but no better end to end.
None of the models catches far aerial shots or synthetic zoom; GeoCalib (zoom) and better aerial rules remain needed.
Cheap option: trust the rules' TransNetV2 cut check instead of the VLM's cut field (it already rejects those clips).
