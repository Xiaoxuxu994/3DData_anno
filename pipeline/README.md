# v1.3 pipeline: one command per clip

`pipeline/` turns the v1.3 recipe (`outputs/evolution/pipeline_best.yaml`, built up in experiments E1–E13) into
runnable entry points. Each step calls the experiment script that produced the recorded results, or a short copy of it
where that script had hard-coded inputs. The numbers match the recorded ones (see "Validation" below).

```
filter (layer 1)                annotate (layer 2)                         QC (layer 3)
cuts, rules, OCR, flow,   ->    ViPE (K, metric scale reference)     ->    scale k = sqrt(k_depth * k_traj)
GeoCalib zoom/fisheye,          GeoCalib + image-only Pi3X / DA3            label = Pi3X poses/depth x k
Qwen3-VL-32B judge              K gate [25, 130] deg                        v1c = max(traj/3.27%, depth/0.194, Kdis/13.2deg)
(+ optional shot split)         Pi3X conditioned on the gated K             flags jump60, fov_out
                                DA3 on ViPE cameras (QC witness)            accept = v1c <= 1 and no reject flag
```

## Usage

```bash
# one clip (video: first 12 s at 15 fps; or a directory of 000000.jpg ...)
code/pipeline/run_clip.sh /path/clip.mp4 outputs/my_run [--gpus=4,5] [--skip-vlm] [--split-cuts]

# many clips in one batch (models are loaded once per step; use this for throughput)
code/pipeline/run_batch.sh outputs/my_batch a.mp4 b.mp4 name=/path/frames_dir ... [--gpus=4,5]

# regression test (A-group GT clips; ~5 min on 2 idle H200)
code/tests/regression_A.sh [--gpus=4,5]
```

Options (`run_batch.sh`): `--gpus=a,b` (default: up to two idle GPUs, chosen automatically), `--skip-filter`,
`--skip-vlm`, `--split-cuts` (E2: shots >= 6 s from clips rejected for cuts, filtered again), `--force-annotate`
(also label clips the filter rejected), `--fps / --max-frames / --long` (video decoding), `--gt-bench=DIR` (tests).
Runs are resumable: every step skips clips whose output already exists.

`ROOT` defaults to `/data/home/z00533370`; set `ANNO_ROOT` to move everything. GPU steps run inside the container
`ANNO_CONTAINER` (default `z00533370_dev`, the one ViPE was built and timed in); `ANNO_CONTAINER=` runs them on the
host (all envs also import on the host). CPU threads per process: `ANNO_THREADS` (default 4).

## Steps and environments

| Step (log name) | Script | Environment | GPU |
|---|---|---|---|
| prep | `pipeline/prep.py` | `cache/qc-venv` | – |
| f_cuts / f_rules / f_flow | `filter/s1_cuts.py`, `s2_rules.py`, `s4_flow.py` via `pipeline/filter_stage.py` | `cache/filter-venv` | 1 |
| f_ocr | `filter/s3_ocr.py` | `cache/ocr-venv` | CPU |
| f_calib | `filter/s5_geocalib.py` | `code/vipe/.venv` (+`PYTHONPATH=code/vipe:code/filter`) | 1 |
| f_vlm | `filter/vlm_run.py` (Qwen3-VL-32B, `cache/modelscope/Qwen/Qwen3-VL-32B-Instruct`, TP 1) | `cache/vllm-venv` | 1 (~63 GB) |
| f_decide | `filter/decide.py --vlm qwen3vl_32b` | `cache/filter-venv` | – |
| (split / select) | `pipeline/filter_select.py` | `cache/qc-venv` | – |
| a_vipe | `evolution/vipe_worker.py` (persistent worker, default profile) | `code/vipe/.venv` | 1 |
| a_geocalib | `pipeline/geocalib_k.py` (= `evolution/geocalib_bench.py`, own output dir) | `code/vipe/.venv` | 1 |
| a_pi3x_img | `bench/pi3x_bench.py` | `code/Pi3/.venv` | 1 (~82 GB) |
| a_da3_img | `bench/da3_bench.py` | `code/depth-anything-3/.venv` | 1 |
| a_pi3x_kgate | `pipeline/pi3x_kgate.py` (K gate + `evolution/pi3x_cond.py` forward) | `code/Pi3/.venv` | 1 |
| q_da3_pkvipe | `evolution/da3_cond.py` | `code/depth-anything-3/.venv` | 1 |
| q_label_qc | `pipeline/v13_label_qc.py` (scale from `evolution/e3_scale_vote.py`, signals from `qc/qc_cross.py`, flags = `evolution/e9_hard_rules.py`) | `cache/qc-venv` | – |

The filter stages write their runtime logs through `fcommon.LOGS`; `filter_stage.py` redirects that to
`OUT/logs/filter`, so a pipeline run never touches the experiment logs.

## Output (`OUT/`)

| Path | Content |
|---|---|
| `labels/<clip>.npz` | **the label**: `c2w` (N,4,4 OpenCV camera-to-world, metres), `K` (N,3,3 at the depth resolution), `depth` (N,h,w metres), `conf` (Pi3X confidence 0–1), `scale` (k), `k_depth`, `k_traj`, `hfov` (deg), `k_source` (ViPE or median of GeoCalib / Pi3X / DA3), `frame_hw` |
| `qc/<clip>.json` | `v1c`, its parts (`xtraj_ba` %, `xdepth_k`, `kdis` deg, each divided by its threshold in `v1c_parts`), `flags` (E9 values), `reject_flags`, `info_flags` (`k_gate_fired`, `suspect_cut`, `focal_unobservable`, `suspect_zoom`), `accept`, scale details, K-gate details |
| `results.csv` | one row per annotated clip: accept, v1c, signals, flags, scale, HFOV |
| `filter/decisions.csv`, `filter/selection.json` | filter verdict per clip (reasons, soft flags, every stage score) and which clips were annotated |
| `pool/<clip>/scores.json` | raw filter stage scores |
| `bench/` | intermediate predictions in the experiments' bench layout (`vipe/`, `pi3x/`, `da3/`, `pi3x_kgate/`, `da3_pkvipe/`, `gate/`, `geocalib/`, `v13/`) |
| `timing.json`, `timing.jsonl`, `logs/` | wall time per step (including model loading), per-clip model seconds, one log per step |

`accept = false` means the label should not be used for training; the reasons are in `qc/<clip>.json`.

## Timing (H200, 2 GPUs, measured 2026-10-21)

| Run | filter (incl. VLM) | annotate + QC | total wall |
|---|---|---|---|
| TUM fr1_room, 200 f 640×480, `--skip-filter` | – | 2.1 min | 2.1 min |
| Taipei indoor walk, 180 f 1080p, full | 1.8 min (VLM load ≈ 70 s) | 3.4 min | 5.2 min |
| BTbk driving 1080p, `--split-cuts --force-annotate` (clip + 1 shot) | 3.5 min (filter twice) | 4.5 min | 8 min |
| regression: 4 A clips in one batch, `--skip-filter` | – | 6.3 min | 6.3 min |

Single-clip wall time is dominated by model loading (each step loads its model once); batch runs amortise it.
Per-clip model time (`timing.json` → `per_clip_model_sec`): ViPE ≈ 58 s (640×480, 200 f) / ≈ 90 s (1080p, 180 f),
Pi3X ≈ 19 s × 2, DA3 ≈ 8–17 s × 2, GeoCalib ≈ 1 s, label + QC ≈ 5–10 s CPU. Production cost numbers (2 ViPE workers per
GPU, half frame rate for slow categories) are in `outputs/evolution/cost.csv`; this entry point runs one ViPE worker.

## Validation (2026-10-21)

- `tests/regression_A.sh` (TUM fr1_room, 7-Scenes fire, KITTI 07, Bonn balloon): 70 PASS, 2 WARN, REGRESSION PASS.
  Label metrics (ATE, Sim3 scale, FOV error, metric AbsRel, F@5cm, g20 EPE), scale k, failure flags, jump60 / fov_out
  and the accept decision match `E3b/raw/A.csv`, `E3/scales_A.csv`, `E1/qc_rule_cheap.csv`, `E9/flags.csv`
  (scale within 0.6 %, ATE within 2 %). The 2 WARN: KITTI 07 xdepth_k 0.27 vs 0.19 and v1c 4.28 vs 4.43 because the
  ViPE re-run differs from the recorded ViPE run by 2.9 % of the trajectory (DA3 witness is conditioned on ViPE's
  cameras); the label itself is identical (depth AbsRel 0.0006 between runs) and both runs reject.
- `tests/compare_C.py` on the Taipei clip: all filter scores identical except GeoCalib (focal ratio 0.879 vs 0.881),
  same VLM verdict and decision; scale 1.3819 vs 1.3823, v1c 0.156 vs 0.156, flags identical.
- BTbk driving clip: filter scores and decision identical (reject: cut + VLM); ViPE re-run differs by 3.6 % of the
  trajectory, so v1c is 3.03 instead of 1.94 (both reject, HFOV 31.7° vs 33.1°); the Pi3X label differs by 0.1 %.

## Known limitations

- ViPE (and to a lesser degree GeoCalib / DA3) is not bit-reproducible. On indoor clips re-runs agree closely (scale
  k within 0.6 %, K disagreement within ~0.5°); on forward-driving clips the ViPE trajectory can differ by 3–4 %, which
  moves the QC scores that use ViPE as witness (v1c 1.94 → 3.03 on BTbk) while the label barely changes. Clips with
  v1c near 1 can flip between runs. The regression test uses tolerances and reports this case as WARN.
- One ViPE worker per run; the 2-workers-per-GPU and half-frame-rate settings of the cost model (E6b/E6c) are not
  automated here (start two `run_batch.sh` on disjoint clip lists, or add `vipe_worker.py ... 0 2` / `1 2`).
- Shot splitting (`--split-cuts`) relies on TransNetV2 boundaries; undetected soft cuts end up in the QC flags
  (jump60) instead.
- The QC thresholds (3.27 %, 0.194, 13.2°) were fitted in-sample on 64 GT clips; the human review should recalibrate them.
- Clip-level flags of line G (`static`, `zoom` majority over 7 annotators) do not exist in v1.3; `suspect_zoom` uses
  ViPE's per-frame focal variation instead (> 7 %), `focal_unobservable` the camera path / depth ratio (< 2 %). Both
  are informational only; zoom rejection stays with the filter's GeoCalib rule.
- The VLM step needs ~63 GB on one GPU and 1–1.5 min of start-up; `--skip-vlm` falls back to the rules
  (C-group recall 0.79 instead of 0.88, see `code/filter/README.md`).
- Inputs longer than 12 s are cut to the first `--max-frames` frames (180 = 12 s at 15 fps) when decoded from video;
  frame directories are used in full.

## GLR track (GLRMap annotator behind the same filter and QC)

`run_batch_glr.sh` runs `run_batch.sh` unchanged (filter, ViPE, K sources, Pi3X label, v1c QC), then routes the clips that passed the filter
(`glr_prepare.py`: static scene + real parallax -> GLRMap, otherwise v1.3), hands GLRMap a job package per clip (`glr_worker.sh`, with a stage-1 exit
gate between SfM and the dense stages), converts the result to the label contract and checks it with the same witnesses (`glr_collect.py`: v1c with the
GLRMap label, tiers T1/T2/T3) and finally picks one label per clip with a fallback to the v1.3 label (`glr_merge.py`). Outputs sit next to the v1.3
ones (`labels_glr/`, `qc_glr/`, `results_glr.csv`, `final/`). Status, contracts, rules compliance and the first-run checklist: `docs/GLR_INTEGRATION.md`.

```bash
code/pipeline/run_batch_glr.sh outputs/my_glr_batch a.mp4 b.mp4 ... --phase=a      # filter + v1.3 + route + job packages (H200 host)
GLRMAP_HOME=... PY_GLR=... GLR_TASKSET=0-7 code/pipeline/glr_worker.sh outputs/my_glr_batch/glr_jobs/<clip>   # GLRMap host
code/pipeline/run_batch_glr.sh outputs/my_glr_batch --phase=c                      # collect, QC, final selection
```

