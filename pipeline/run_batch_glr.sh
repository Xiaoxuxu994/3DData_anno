#!/bin/bash
# GLR track on a batch of clips: v1.3 front end + witnesses (run_batch.sh) -> route -> GLRMap jobs -> collect/QC -> final selection.
#
# Usage: pipeline/run_batch_glr.sh OUT INPUT [INPUT ...] [options]
#   INPUT / options of run_batch.sh are passed through: --gpus= --skip-filter --skip-vlm --split-cuts --force-annotate --fps= --max-frames= --long=
#   --phase=a|b|c|all   a: run_batch.sh (filter, ViPE, K sources, v1.3 label + QC) + route + job packages     [H200 host]
#                       b: glr_worker.sh on every job without a result (GLRMap host; needs GLRMAP_HOME, PY_GLR)
#                       c: glr_collect.py (labels + QC + tier) + glr_merge.py (final verdict per clip)        [H200 host]
#                       all (default) = a, b, c on this host. With two hosts: run a here, rsync -aL OUT/glr_jobs to the GLRMap host,
#                       run `pipeline/glr_worker.sh JOB` there (or --phase=b with the same OUT path), rsync the jobs back, run c here.
#   --glr-all           send every clip that passed the filter to GLRMap (default: route by the filter's flow statistics, see glr_prepare.py)
#   --worker-mode=mono|split   passed to glr_worker.sh (default mono = GLRMap's own run_all.sh)
#   --glr-force         continue after a stage-1 gate stop (the verdict still rejects the label in phase c)
#   --dry-run           print what would run (phase a and c are only listed)
# Output: everything of run_batch.sh (labels/, qc/, results.csv, filter/ ...) plus filter/glr_route.csv, glr_jobs/<clip>/, labels_glr/, qc_glr/,
#         results_glr.csv and final/{results.csv, labels/, summary.json}. Existing results are never overwritten (resume = run again).
# CPU limits (AGENTS.md section 3): ANNO_THREADS per process; GLRMap's SfM spawns more threads by itself -> set GLR_TASKSET on shared hosts.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/common.sh"
PY_GLR=${PY_GLR:-$ROOT/cache/glrmap-venv/bin/python}

OUT=$(realpath -m "$1"); shift
PASS=(); INPUTS=(); PHASE=all; GLRALL=0; WMODE=mono; GFORCE=0; DRY=0
for a in "$@"; do
  case $a in
    --phase=*) PHASE=${a#*=} ;;
    --glr-all) GLRALL=1 ;;
    --worker-mode=*) WMODE=${a#*=} ;;
    --glr-force) GFORCE=1 ;;
    --dry-run) DRY=1 ;;
    --gpus=*|--skip-filter|--skip-vlm|--split-cuts|--force-annotate|--fps=*|--max-frames=*|--long=*|--gt-bench=*) PASS+=("$a") ;;
    --*) echo "unknown option $a"; exit 2 ;;
    *) INPUTS+=("$a") ;;
  esac
done
case $PHASE in a|b|c|all) ;; *) echo "--phase must be a, b, c or all"; exit 2 ;; esac
[ $PHASE = b ] || [ ${#INPUTS[@]} -ge 1 ] || { sed -n 2,19p "$0"; exit 2; }
mkdir -p "$OUT/logs"
cd "$PIPE"
run() { if [ $DRY = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }

if [ $PHASE = a ] || [ $PHASE = all ]; then
  if [ ${#INPUTS[@]} -ge 1 ]; then
    run "$HERE/run_batch.sh" "$OUT" "${INPUTS[@]}" "${PASS[@]}" || exit 1
  fi
  RO=(); [ $GLRALL = 1 ] && RO+=(--glr-all)
  run step g_route run_py "" $PY_QC glr_prepare.py "$OUT" route "${RO[@]}" || exit 1
  run step g_jobs  run_py "" $PY_QC glr_prepare.py "$OUT" jobs || exit 1
fi

if [ $PHASE = b ] || [ $PHASE = all ]; then
  WO=(--mode=$WMODE); [ $GFORCE = 1 ] && WO+=(--force); [ $DRY = 1 ] && WO+=(--dry-run)
  for job in "$OUT"/glr_jobs/*/; do
    [ -f "$job/job.json" ] || continue
    [ -f "$job/RESULT_DIR" ] && { echo "has a result, skipped: $(basename "$job")"; continue; }
    echo "[$(date +%T)] GLRMap: $(basename "$job")" | tee -a "$OUT/logs/pipeline.log"
    PY_GLR=$PY_GLR "$HERE/glr_worker.sh" "$job" "${WO[@]}" >> "$OUT/logs/glr_$(basename "$job").log" 2>&1
    rc=$?; echo "  rc=$rc" | tee -a "$OUT/logs/pipeline.log"   # 3 = stopped at the gate is a normal outcome, not an error
  done
fi

if [ $PHASE = c ] || [ $PHASE = all ]; then
  run step g_collect run_py "" $PY_QC glr_collect.py "$OUT" || exit 1
  run step g_merge   run_py "" $PY_QC glr_merge.py "$OUT" || exit 1
  [ $DRY = 1 ] || { [ -f "$OUT/final/results.csv" ] && cat "$OUT/final/results.csv"; }
fi
