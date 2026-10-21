#!/bin/bash
# v1.3 pipeline on a batch of clips: (1) filter -> (2) annotate -> (3) QC.  Models are loaded once per step.
#
# Usage: pipeline/run_batch.sh OUT INPUT [INPUT ...] [options]
#   INPUT  video file or frame directory (000000.jpg ...); "NAME=INPUT" sets the clip name
#   --gpus=4,5        GPUs to use (default: up to 2 idle GPUs picked automatically; 1 is enough)
#   --skip-filter     no filter layer (e.g. GT benchmark clips); every clip is annotated
#   --skip-vlm        filter rules only (no Qwen3-VL-32B judge; vLLM start-up costs ~2-3 min)
#   --split-cuts      E2: split clips rejected for cuts into shots >= 6 s and filter them again
#   --force-annotate  annotate even the clips the filter rejected (their filter verdict is still reported)
#   --gt-bench=DIR    link GT from a bench (outputs/bench/eval3d_A ...) for evaluation (tests only)
#   --fps=15 --max-frames=180 --long=1920   video decoding (12 s at 15 fps, long side <= 1920)
# Output: OUT/labels/<clip>.npz, OUT/qc/<clip>.json, OUT/results.csv, OUT/filter/decisions.csv, OUT/timing.json
# Environments: see pipeline/common.sh and pipeline/README.md.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/common.sh"

OUT=$(realpath -m "$1"); shift
INPUTS=(); GPUS=""; SKIP_FILTER=0; SKIP_VLM=0; SPLIT=0; FORCE=0; PREP_OPTS=()
for a in "$@"; do
  case $a in
    --gpus=*) GPUS=${a#*=} ;;
    --skip-filter) SKIP_FILTER=1 ;;
    --skip-vlm) SKIP_VLM=1 ;;
    --split-cuts) SPLIT=1 ;;
    --force-annotate) FORCE=1 ;;
    --gt-bench=*) PREP_OPTS+=("--gt_bench=${a#*=}") ;;
    --fps=*) PREP_OPTS+=("--fps=${a#*=}") ;;
    --max-frames=*) PREP_OPTS+=("--max_frames=${a#*=}") ;;
    --long=*) PREP_OPTS+=("--long=${a#*=}") ;;
    --*) echo "unknown option $a"; exit 2 ;;
    *) INPUTS+=("$a") ;;
  esac
done
[ ${#INPUTS[@]} -ge 1 ] || { sed -n 2,15p "$0"; exit 2; }
mkdir -p "$OUT/logs" "$OUT/filter"
B=$OUT/bench
[ -n "$GPUS" ] || GPUS=$(pick_gpus 2)
[ -n "$GPUS" ] || { echo "no idle GPU found; pass --gpus="; exit 3; }
G1=${GPUS%%,*}; G2=${GPUS##*,}
echo "OUT=$OUT GPUS=$GPUS container=${ANNO_CONTAINER:-<host>}" | tee -a "$OUT/logs/pipeline.log"
cd "$PIPE"
T_START=$(date +%s)

# ---------------- 0. inputs -> bench layout
step prep $PY_QC prep.py "$OUT" "${INPUTS[@]}" "${PREP_OPTS[@]}" || exit 1

# ---------------- 1. filter (rules + VLM), each stage = the unchanged code/filter script
export FILTER_POOL=$OUT/pool ANNO_LOGS=$OUT/logs/filter
run_filter() {
  step f_cuts  run_py $G1 $PY_FILTER filter_stage.py s1_cuts.py 0 1 "$FILTER_POOL" || return 1
  step f_rules run_py $G1 $PY_FILTER filter_stage.py s2_rules.py 0 1 "$FILTER_POOL" || return 1
  step f_ocr   run_py "" $PY_OCR filter_stage.py s3_ocr.py 0 1 "$FILTER_POOL" || return 1
  step f_flow  run_py $G1 $PY_FILTER filter_stage.py s4_flow.py 0 1 "$FILTER_POOL" || return 1
  PYTHONPATH=$CODE/vipe:$CODE/filter step f_calib run_py $G1 $PY_VIPE filter_stage.py s5_geocalib.py 0 1 "$FILTER_POOL" || return 1
  if [ $SKIP_VLM = 0 ]; then
    step f_vlm run_py $G1 $PY_VLLM filter_stage.py vlm_run.py $VLM_TAG $VLM_PATH 1 "$FILTER_POOL" || return 1
    step f_decide $PY_FILTER $CODE/filter/decide.py "$FILTER_POOL" "$OUT/filter/decisions.csv" --vlm $VLM_TAG || return 1
  else
    step f_decide $PY_FILTER $CODE/filter/decide.py "$FILTER_POOL" "$OUT/filter/decisions.csv" || return 1
  fi
}
if [ $SKIP_FILTER = 0 ]; then
  run_filter || exit 1
  if [ $SPLIT = 1 ]; then
    n_new=$($PY_QC filter_select.py "$OUT" split | tail -1)
    echo "shots added: $n_new" | tee -a "$OUT/logs/pipeline.log"
    [ "${n_new:-0}" -gt 0 ] && { run_filter || exit 1; }
  fi
  sel=$([ $FORCE = 1 ] && echo --force)
  n_ann=$($PY_QC filter_select.py "$OUT" select $sel | tail -1)
  echo "clips to annotate: $n_ann" | tee -a "$OUT/logs/pipeline.log"
  [ "$n_ann" -gt 0 ] || { echo "nothing passed the filter"; $PY_QC "$PIPE/summarize.py" "$OUT" $(( $(date +%s) - T_START )); exit 0; }
fi

# ---------------- 2. annotate
# ViPE (persistent worker, default profile) -> K + metric scale reference; GeoCalib / image-only Pi3X / DA3 -> K gate
$PY_QC -c "import json; [print('$B/frames/' + c['clip']) for c in json.load(open('$B/clips.json'))]" > "$OUT/vipe_list.txt"
step a_vipe run_py $G1 $PY_VIPE $CODE/evolution/vipe_worker.py v13 "$B/vipe" "$OUT/vipe_list.txt" &
P_VIPE=$!
if [ "$G2" != "$G1" ]; then  # second GPU: the K-gate sources run next to ViPE
  ( PYTHONPATH=$CODE/vipe step a_geocalib run_py $G2 $PY_VIPE geocalib_k.py "$B" &&
    step a_pi3x_img run_py $G2 $PY_PI3 $CODE/bench/pi3x_bench.py w0 "$B" &&
    step a_da3_img run_py $G2 $PY_DA3 $CODE/bench/da3_bench.py w0 "$B" ) || exit 1
  wait $P_VIPE || exit 1
else
  wait $P_VIPE || exit 1
  PYTHONPATH=$CODE/vipe step a_geocalib run_py $G1 $PY_VIPE geocalib_k.py "$B" || exit 1
  step a_pi3x_img run_py $G1 $PY_PI3 $CODE/bench/pi3x_bench.py w0 "$B" || exit 1
  step a_da3_img run_py $G1 $PY_DA3 $CODE/bench/da3_bench.py w0 "$B" || exit 1
fi
# K gate + Pi3X conditioned on the gated K (the label) | DA3 on ViPE cameras (QC witness)
step a_pi3x_kgate run_py $G1 $PY_PI3 pi3x_kgate.py "$B" &
P1=$!
step q_da3_pkvipe run_py $G2 $PY_DA3 $CODE/evolution/da3_cond.py w0 "$B" || exit 1
wait $P1 || exit 1

# ---------------- 3. metric scale, label, QC v1c + flags
step q_label_qc run_py "" $PY_QC v13_label_qc.py "$OUT" || exit 1
$PY_QC "$PIPE/summarize.py" "$OUT" $(( $(date +%s) - T_START ))
