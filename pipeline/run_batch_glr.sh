#!/bin/bash
# GLR track on a batch of clips: v1.3 front end + witnesses (run_batch.sh) -> route -> GLRMap jobs -> collect/QC -> final selection.
#
# Usage: pipeline/run_batch_glr.sh OUT [INPUT ...] [options]            (INPUT is required for --phase=all, the default)
#   INPUT / options of run_batch.sh are passed through: --gpus= --skip-filter --skip-vlm --split-cuts --force-annotate --fps= --max-frames= --long=
#   --phase=a|b|c|all   a: run_batch.sh (filter, ViPE, K sources, v1.3 label + QC; only when INPUTs are given) + route + job packages  [H200 host]
#                       b: glr_worker.sh on every job without a finished result (GLRMap host; needs GLRMAP_HOME, PY_GLR, GLR_TASKSET)
#                       c: glr_collect.py (labels + QC + tier) + glr_merge.py (final verdict per clip)        [H200 host]
#                       all (default) = a, b, c on this host. With two hosts: run a here, rsync -aL OUT/glr_jobs to the GLRMap host,
#                       run `--phase=b` there with the same OUT path (or `pipeline/glr_worker.sh JOB` per job), rsync OUT/glr_jobs back
#                       (the whole directory, including RESULT_DIR), run `--phase=c` here.
#   --glr-all           send every clip that passed the filter to GLRMap (default: route by the filter's flow statistics, see glr_prepare.py)
#   --worker-mode=mono|split   passed to glr_worker.sh (default mono = GLRMap's own run_all.sh; split needs GLR_DENSE_CMD, or GLR_STAGE1_ONLY=1)
#   --glr-force         continue after a stage-1 gate stop (the verdict still rejects the label in phase c)
#   --glr-retry         also run again the jobs whose last result is `failed` / `gate_error` (interrupted or unfinished jobs always run again)
#   --gpus=LIST         GPUs for run_batch.sh AND for the GLRMap worker (CUDA_VISIBLE_DEVICES). Without it phase b takes one idle GPU (pick_gpus)
#   --dry-run           print what would run (phase a and c are only listed); nothing is created or written
# Output: everything of run_batch.sh (labels/, qc/, results.csv, filter/ ...) plus filter/glr_route.csv, glr_jobs/<clip>/, labels_glr/, qc_glr/,
#         results_glr.csv and final/{results.csv, labels/, summary.json}. The v1.3 outputs and every GLRMap result directory are never
#         overwritten (result, result_1, ...). results_glr.csv, qc_glr/, labels_glr/ and final/ are derived from them and are rebuilt by
#         phase c, so "resume = run again": finished jobs are skipped, unfinished ones run again, phase c reflects the current state.
# Exit: 0 | 1 a step failed, or some GLRMap job failed (rc 4 / 5; 3 = stopped at the gate is a normal outcome) | 2 bad usage / environment
# CPU limits (AGENTS.md section 3): ANNO_THREADS per process; GLRMap's SfM ignores it and spawns hundreds of threads -> GLR_TASKSET=<cpu list>
#         is required for phase b (or GLR_ALLOW_UNCAPPED=1 on a host that is yours alone).
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/common.sh"
PY_GLR=${PY_GLR:-$ROOT/cache/glrmap-venv/bin/python}
usage() { sed -n 2,24p "$0"; }

[ $# -ge 1 ] && [ -n "$1" ] || { usage; exit 2; }
OUT=$(realpath -m -- "$1"); shift
[ -n "$OUT" ] && [ "$OUT" != / ] || { usage; exit 2; }
PASS=(); INPUTS=(); PHASE=all; GLRALL=0; WMODE=mono; GFORCE=0; GRETRY=0; DRY=0; GPUS_OPT=""
for a in "$@"; do
  case $a in
    --phase=*) PHASE=${a#*=} ;;
    --glr-all) GLRALL=1 ;;
    --worker-mode=*) WMODE=${a#*=} ;;
    --glr-force) GFORCE=1 ;;
    --glr-retry) GRETRY=1 ;;
    --dry-run) DRY=1 ;;
    --gpus=*) GPUS_OPT=${a#*=}; PASS+=("$a") ;;
    --skip-filter|--skip-vlm|--split-cuts|--force-annotate|--fps=*|--max-frames=*|--long=*|--gt-bench=*) PASS+=("$a") ;;
    --*) echo "unknown option $a"; exit 2 ;;
    *) INPUTS+=("$a") ;;
  esac
done
case $PHASE in a|b|c|all) ;; *) echo "--phase must be a, b, c or all"; exit 2 ;; esac
case $WMODE in mono|split) ;; *) echo "--worker-mode must be mono or split"; exit 2 ;; esac
[ $PHASE != all ] || [ ${#INPUTS[@]} -ge 1 ] || { usage; exit 2; }
[ $DRY = 1 ] || mkdir -p "$OUT/logs"
cd "$PIPE"
run() { if [ $DRY = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
NFAIL=0; FAILED=()

if [ $PHASE = a ] || [ $PHASE = all ]; then
  if [ ${#INPUTS[@]} -ge 1 ]; then
    run "$HERE/run_batch.sh" "$OUT" "${INPUTS[@]}" "${PASS[@]}" || exit 1
  fi
  RO=(); [ $GLRALL = 1 ] && RO+=(--glr-all)
  run step g_route run_py "" $PY_QC glr_prepare.py "$OUT" route "${RO[@]}" || exit 1
  # a clip whose package cannot be written is reported and skipped (the merge falls back to v1.3 for it); the others are still prepared
  run step g_jobs  run_py "" $PY_QC glr_prepare.py "$OUT" jobs || { echo "some job packages could not be written, see $OUT/logs/g_jobs.log"; NFAIL=$((NFAIL + 1)); }
fi

if [ $PHASE = b ] || [ $PHASE = all ]; then
  if [ $DRY = 0 ]; then   # fail once, early, with a clear message instead of once per job
    [ -n "${GLRMAP_HOME:-}" ] && [ -d "$GLRMAP_HOME" ] || { echo "phase b needs GLRMAP_HOME (the GLRMap checkout)"; exit 2; }
    command -v "$PY_GLR" > /dev/null 2>&1 || { echo "PY_GLR=$PY_GLR not found (python of the GLRMap environment)"; exit 2; }
    if [ -z "${GLR_TASKSET:-}" ] && [ "${GLR_ALLOW_UNCAPPED:-0}" != 1 ]; then
      echo "phase b needs GLR_TASKSET=<cpu list> (GLOMAP ignores thread caps; AGENTS.md rule 2) or GLR_ALLOW_UNCAPPED=1 on a host that is yours alone"; exit 2
    fi
    if [ $WMODE = split ] && [ -z "${GLR_DENSE_CMD:-}" ] && [ "${GLR_STAGE1_ONLY:-0}" != 1 ]; then
      echo "--worker-mode=split needs GLR_DENSE_CMD (a stage-1-only result has no depth and is rejected by the QC); GLR_STAGE1_ONLY=1 runs stage 1 + gate only"; exit 2
    fi
    if [ -n "$GPUS_OPT" ]; then
      export CUDA_VISIBLE_DEVICES=$GPUS_OPT
    elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi > /dev/null 2>&1; then
      G=$(pick_gpus 1)
      [ -n "$G" ] || { echo "no idle GPU for GLRMap (AGENTS.md rule 1): free one or pass --gpus=N"; exit 2; }
      export CUDA_VISIBLE_DEVICES=$G
    fi
    [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || echo "GLRMap runs on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi
  finished() {  # a job is finished when RESULT_DIR names a directory with result.json that is not `interrupted` (and, unless --glr-retry, not failed)
    local rd
    [ -f "$1/RESULT_DIR" ] || return 1
    rd=$1/$(basename "$(cat "$1/RESULT_DIR")")
    [ -f "$rd/result.json" ] || return 1
    grep -q '"status": "interrupted"' "$rd/result.json" && return 1
    [ $GRETRY = 1 ] && grep -q -E '"status": "(failed|gate_error)"' "$rd/result.json" && return 1
    return 0
  }
  WO=(--mode=$WMODE); [ $GFORCE = 1 ] && WO+=(--force); [ $DRY = 1 ] && WO+=(--dry-run)
  for job in "$OUT"/glr_jobs/*/; do
    [ -f "$job/job.json" ] || continue
    name=$(basename "$job")
    if finished "$job"; then echo "has a result, skipped: $name"; continue; fi
    if [ $DRY = 1 ]; then
      echo "[dry-run] GLRMap: $name"
      PY_GLR=$PY_GLR "$HERE/glr_worker.sh" "$job" "${WO[@]}"
      continue
    fi
    echo "[$(date +%T)] GLRMap: $name" | tee -a "$OUT/logs/pipeline.log"
    PY_GLR=$PY_GLR "$HERE/glr_worker.sh" "$job" "${WO[@]}" >> "$OUT/logs/glr_$name.log" 2>&1
    rc=$?
    echo "  rc=$rc" | tee -a "$OUT/logs/pipeline.log"
    case $rc in 0|3) ;; *) FAILED+=("$name(rc=$rc)") ;; esac   # 3 = stopped at the gate is a normal outcome, not an error
  done
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "GLRMap jobs that failed (details in $OUT/logs/glr_<clip>.log and glr_jobs/<clip>/result*/logs/): ${FAILED[*]}" | tee -a "$OUT/logs/pipeline.log"
    NFAIL=$((NFAIL + ${#FAILED[@]}))
  fi
fi

if [ $PHASE = c ] || [ $PHASE = all ]; then
  run step g_collect run_py "" $PY_QC glr_collect.py "$OUT" || exit 1
  run step g_merge   run_py "" $PY_QC glr_merge.py "$OUT" || exit 1
  [ $DRY = 1 ] || { [ -f "$OUT/final/results.csv" ] && cat "$OUT/final/results.csv"; }
fi
[ $NFAIL -eq 0 ] || exit 1
exit 0
