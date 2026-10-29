#!/bin/bash
# GLR track, phase B (GLRMap host): run GLRMap on ONE job package from glr_prepare.py, with the stage-1 exit gate.
#
# Usage: glr_worker.sh JOB_DIR [--mode=mono|split] [--force] [--dry-run]
#   JOB_DIR   OUT/glr_jobs/<clip> (images/, job.json) - copy it to the GLRMap host with `rsync -aL`
#   --mode=mono   (default) run GLR_RUN_CMD once (GLRMap's own run_all.sh: stage 1 + RoMa + PDA), then read the stage-1 model and
#                 apply the gate afterwards. If your GLRMap copy has the NPU "camera_sanity" patch the run stops itself
#                 (exit 3) and the worker records `native_gate_stop`.
#   --mode=split  run GLR_STAGE1_CMD, gate, and only continue with GLR_DENSE_CMD when the gate does not say stop (saves the
#                 hours of RoMa/PDA on a wrong camera). Needs both templates (see below); without GLR_DENSE_CMD it stops after the
#                 gate (status stage1_only) - useful to replay Phase 0 on stage 1 alone.
#   --force       (or GLR_FORCE=1) continue after a gate stop; the gate verdict stays in result.json and is used by the QC
#   --dry-run     print the commands, run nothing
# Environment:
#   GLRMAP_HOME   GLRMap checkout (directory with reconstruction_flow/ and data_generate_demo/)   [required unless --dry-run]
#   PY_GLR        python of the GLRMap environment (needs numpy + pycolmap)       [default python3]
#   ANNO_THREADS  CPU threads per process (default 4); GLOMAP spawns more by itself -> set GLR_TASKSET on shared hosts
#   GLR_TASKSET   CPU list for `taskset -c` (e.g. 8-15); hard cap on cores regardless of library thread pools
#   GLR_CAMERA_MODEL  PINHOLE (GLRMap default) | SIMPLE_PINHOLE (fx = fy)
#   GLR_RUN_CMD   mono: default `bash data_generate_demo/run_all.sh`; it reads IMAGES, OUTPUT (+ MASK_DIR) from the environment
#   GLR_STAGE1_CMD / GLR_DENSE_CMD   split templates, run from GLRMAP_HOME; placeholders {images} {out} {s1} {threads} {py}
#                 default stage 1: {py} -m reconstruction_flow.glomap --images {images} --output {s1} --camera-model $GLR_CAMERA_MODEL
#                 dense: no default - derive it from data_generate_demo/run_roma_pcd.sh and run_pda_depth.sh (docs/GLR_INTEGRATION.md)
#   GLR_TIMEOUT   seconds per GLRMap command (default none)
# Output (never overwritten: result, result_1, ...; the name is written to JOB_DIR/RESULT_DIR):
#   result*/{stage1|run, dense, logs/, timing.jsonl, gate.json, result.json}
# Exit: 0 complete | 3 stopped at the gate | 4 bad arguments / would overwrite | 5 a GLRMap command failed
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
JOB=${1:-}; shift
MODE=mono; FORCE=${GLR_FORCE:-0}; DRY=0
for a in "$@"; do
  case $a in
    --mode=*) MODE=${a#*=} ;;
    --force) FORCE=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown option $a"; exit 4 ;;
  esac
done
[ -f "$JOB/job.json" ] || { sed -n 2,28p "$0"; exit 4; }
[ "$MODE" = mono ] || [ "$MODE" = split ] || { echo "--mode must be mono or split"; exit 4; }
JOB=$(realpath "$JOB")
PY=${PY_GLR:-python3}
THREADS=${ANNO_THREADS:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
TS=""; [ -n "${GLR_TASKSET:-}" ] && TS="taskset -c $GLR_TASKSET"
TO=""; [ -n "${GLR_TIMEOUT:-}" ] && TO="timeout $GLR_TIMEOUT"
CAM=${GLR_CAMERA_MODEL:-PINHOLE}
[ $DRY = 1 ] || [ -n "${GLRMAP_HOME:-}" ] || { echo "GLRMAP_HOME is not set"; exit 4; }

RES=$JOB/result; n=1
while [ -e "$RES" ]; do RES=$JOB/result_$n; n=$((n + 1)); done   # rule: never overwrite an existing result
if [ $DRY = 0 ]; then mkdir -p "$RES/logs" && basename "$RES" > "$JOB/RESULT_DIR"; fi
IMAGES=$JOB/images
NFR=$(ls "$IMAGES" | wc -l)
CLIP=$(basename "$JOB")
STATUS=failed; GATE=""; DENSE_RAN=0; RC=0
CMDS=()

fill() {  # replace the placeholders of a command template
  local t=$1
  t=${t//\{images\}/$IMAGES}; t=${t//\{out\}/$RES}; t=${t//\{s1\}/$RES/stage1}; t=${t//\{threads\}/$THREADS}; t=${t//\{py\}/$PY}
  echo "$t"
}
step() {  # step NAME COMMAND-STRING : run in GLRMAP_HOME with taskset/timeout, log, timing
  local name=$1 cmd=$2 t0 t1 rc
  CMDS+=("$name: $cmd")
  echo "[$(date +%T)] >>> $name: $cmd"
  [ $DRY = 1 ] && return 0
  t0=$(date +%s%N)
  ( cd "$GLRMAP_HOME" && eval "$TS $TO $cmd" ) >> "$RES/logs/$name.log" 2>&1
  rc=$?
  t1=$(date +%s%N)
  printf '{"step": "%s", "sec": %.2f, "rc": %d}\n' "$name" "$(( (t1 - t0) / 1000000 ))e-3" "$rc" >> "$RES/timing.jsonl"
  [ $rc -eq 0 ] || echo "step $name failed (rc=$rc), see $RES/logs/$name.log"
  return $rc
}
find_model() {  # shallowest sparse/<k>/cameras.{bin,txt} below $1 -> model dir (stdout)
  find "$1" -path '*/sparse/*/cameras.*' 2>/dev/null | awk '{print length($0) " " $0}' | sort -n | head -1 | cut -d' ' -f2- | xargs -r dirname
}
run_gate() {  # run_gate MODEL_DIR -> $RES/gate.json ; sets GATE = ok|suspect|stop|missing
  local model=$1 db grc
  [ -n "$model" ] || { GATE=missing; echo '{"verdict": "stop", "stop": ["no_model"], "suspect": [], "signals": {}}' > "$RES/gate.json"; return; }
  db=$(dirname "$(dirname "$model")")/database.db
  [ -f "$db" ] || db=$(find "$(dirname "$(dirname "$model")")" -maxdepth 2 -name database.db | head -1)
  $PY "$HERE/glr_stage1_gate.py" "$model" "$db" "$NFR" --job "$JOB/job.json" > "$RES/gate_full.json" 2> "$RES/logs/gate.log"
  local grc=$?
  $PY -c "import json,sys; d=json.load(open('$RES/gate_full.json')); json.dump(dict(d['gate'], model=d['model'], db=d['db']), open('$RES/gate.json','w'), indent=1)" 2>>"$RES/logs/gate.log"
  case $grc in 0) GATE=ok ;; 1) GATE=suspect ;; 3) GATE=stop ;; *) GATE=error ;; esac
  echo "stage-1 gate: $GATE (model $model)"
}

if [ "$MODE" = mono ]; then
  RUN_CMD=${GLR_RUN_CMD:-bash data_generate_demo/run_all.sh}
  export IMAGES OUTPUT=$RES/run
  [ -d "$JOB/masks" ] && export MASK_DIR=$JOB/masks
  step run "$RUN_CMD"; RUN_RC=$?
  if [ $DRY = 0 ]; then
    MODEL=$(find_model "$RES/run"); run_gate "$MODEL"
    if [ $RUN_RC -eq 3 ]; then STATUS=native_gate_stop
    elif [ $RUN_RC -ne 0 ]; then STATUS=failed; RC=5
    elif [ "$GATE" = stop ]; then STATUS=complete_gate_stop; RC=3
    else STATUS=complete; DENSE_RAN=1; fi
  fi
else
  S1_CMD=${GLR_STAGE1_CMD:-"{py} -m reconstruction_flow.glomap --images {images} --output {s1} --camera-model $CAM"}
  step stage1 "$(fill "$S1_CMD")"; rc=$?
  if [ $DRY = 0 ]; then
    if [ $rc -ne 0 ]; then STATUS=failed; RC=5
    else
      MODEL=$(find_model "$RES/stage1"); run_gate "$MODEL"
      if [ "$GATE" = stop ] && [ "$FORCE" != 1 ]; then STATUS=stopped_at_gate; RC=3
      elif [ -z "${GLR_DENSE_CMD:-}" ]; then STATUS=stage1_only
      else
        step dense "$(fill "$GLR_DENSE_CMD")"; rc=$?
        if [ $rc -eq 0 ]; then STATUS=complete; DENSE_RAN=1; else STATUS=failed; RC=5; fi
      fi
    fi
  else
    [ -n "${GLR_DENSE_CMD:-}" ] && step dense "$(fill "$GLR_DENSE_CMD")"
  fi
fi

if [ $DRY = 0 ]; then
  export RES JOB CLIP MODE STATUS GATE DENSE_RAN NFR THREADS FORCE GLR_TASKSET GLRMAP_HOME
  CMDS_JSON=$(printf '%s\n' "${CMDS[@]}" | $PY -c "import sys,json; print(json.dumps([l.rstrip('\n') for l in sys.stdin if l.strip()]))")
  CMDS_JSON=$CMDS_JSON $PY - <<'EOF'
import json, os, subprocess, socket
from pathlib import Path
res = Path(os.environ["RES"])
git = ""
try:
    git = subprocess.run(["git", "-C", os.environ["GLRMAP_HOME"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    pass
find = lambda pat: sorted(str(p.relative_to(res)) for p in res.glob(pat))[:3]
timing = [json.loads(l) for l in (res / "timing.jsonl").read_text().splitlines()] if (res / "timing.jsonl").exists() else []
r = {"clip": os.environ["CLIP"], "mode": os.environ["MODE"], "status": os.environ["STATUS"], "gate": os.environ["GATE"],
     "dense_ran": os.environ["DENSE_RAN"] == "1", "force": os.environ["FORCE"] == "1", "n_frames": int(os.environ["NFR"]),
     "threads": int(os.environ["THREADS"]), "taskset": os.environ.get("GLR_TASKSET", ""), "host": socket.gethostname(),
     "glrmap_commit": git, "commands": json.loads(os.environ["CMDS_JSON"]), "timing": timing,
     "found": {"sparse_models": find("**/sparse/*/cameras.*"), "databases": find("**/database.db"),
               "pda_fused": find("**/fused/*.npz"), "pi3x_prior": find("**/prior/**/*.npz"),
               "roma_points": find("**/new_points.npz")}}
(res / "result.json").write_text(json.dumps(r, indent=1))
print(json.dumps({k: r[k] for k in ("clip", "status", "gate", "dense_ran")}))
EOF
fi
exit $RC
