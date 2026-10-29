#!/bin/bash
# GLR track, phase B (GLRMap host): run GLRMap on ONE job package from glr_prepare.py, with the stage-1 exit gate.
#
# Usage: glr_worker.sh JOB_DIR [--mode=mono|split] [--force] [--dry-run]
#   JOB_DIR   OUT/glr_jobs/<clip> (images/, job.json) - copy it to the GLRMap host with `rsync -aL` (images are symlinks)
#   --mode=mono   (default) run GLR_RUN_CMD once (GLRMap's own run_all.sh: stage 1 + RoMa + PDA), then read the stage-1 model and
#                 apply the gate afterwards. If your GLRMap copy has the NPU "camera_sanity" patch the run stops itself
#                 (exit 3) and the worker records `native_gate_stop`.
#   --mode=split  run GLR_STAGE1_CMD, gate, and only continue with GLR_DENSE_CMD when the gate does not say stop (saves the
#                 hours of RoMa/PDA on a wrong camera). Needs both templates (see below); without GLR_DENSE_CMD it stops after the
#                 gate (status stage1_only) - useful to replay Phase 0 on stage 1 alone; glr_collect rejects such a result (no depth).
#   --force       (or GLR_FORCE=1) continue after a gate STOP (not after a gate error / missing model); the verdict stays in
#                 result.json and is used by the QC
#   --dry-run     print the commands, run nothing
# Environment:
#   GLRMAP_HOME   GLRMap checkout (directory with reconstruction_flow/ and data_generate_demo/)   [required unless --dry-run]
#   PY_GLR        python of the GLRMap environment (needs numpy + pycolmap; checked before anything runs)  [default python3]
#   ANNO_THREADS  CPU threads per process (default 4). GLOMAP ignores OMP/MKL caps and spawns hundreds of threads ->
#   GLR_TASKSET   CPU list for `taskset -c` (e.g. 8-15): hard cap on cores for the WHOLE command template. Required unless
#                 GLR_ALLOW_UNCAPPED=1 (AGENTS.md rule 2: a shared host once reached load 540)
#   GLR_TIMEOUT   seconds per GLRMap command (default none); also wraps the whole template
#   GLR_CAMERA_MODEL  PINHOLE (GLRMap default) | SIMPLE_PINHOLE (fx = fy)
#   GLR_RUN_CMD   mono: default `bash data_generate_demo/run_all.sh`; it reads IMAGES, OUTPUT (+ MASK_DIR) from the environment
#   GLR_STAGE1_CMD / GLR_DENSE_CMD   split templates (bash command lines, run by `bash -c` from GLRMAP_HOME); placeholders
#                 {images} {out} {s1} {threads} {py} {focal_w} {hfov} become references to the variables GLR_IMAGES GLR_OUT GLR_S1
#                 GLR_THREADS GLR_PY GLR_FOCAL_W GLR_HFOV (quote-aware: write them bare or inside "..."; values with blanks, quotes,
#                 ';' or '&' stay data, never code). {hfov} / {focal_w} = the image-only K prior of the job (median HFOV in degrees of
#                 GeoCalib / Pi3X / DA3, and the same as focal / image width) - empty when the job has none. GLRMap's own CLI decides
#                 whether it can take an initial focal; the pycolmap replica (tools/stage1_sfm.py --focal-w) can.
#                 default stage 1: {py} -m reconstruction_flow.glomap --images {images} --output {s1} --camera-model $GLR_CAMERA_MODEL
#                 dense: no default - derive it from data_generate_demo/run_roma_pcd.sh and run_pda_depth.sh (docs/GLR_INTEGRATION.md)
# Output (never overwritten: result, result_1, ...). JOB_DIR/RESULT_DIR (the name of the result directory) is written LAST, after
# result.json: a job without it was interrupted or is still running and may be run again.
#   result*/{stage1|run, dense, logs/, timing.jsonl, gate.json, result.json}
# gate.json always exists after a stage-1 model was looked for: verdict ok | suspect | stop | error (the gate itself failed - never "ok").
# Exit: 0 complete / stage1_only | 3 stopped at the gate | 4 bad arguments / would overwrite | 5 a GLRMap command or the gate failed
#       130 / 143 interrupted (INT / TERM: the GLRMap command group is stopped, result.json says `interrupted`)
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
[ -f "$JOB/job.json" ] || { sed -n 2,40p "$0"; exit 4; }
[ "$MODE" = mono ] || [ "$MODE" = split ] || { echo "--mode must be mono or split"; exit 4; }
JOB=$(realpath "$JOB")
PY=${PY_GLR:-python3}
THREADS=${ANNO_THREADS:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
CAM=${GLR_CAMERA_MODEL:-PINHOLE}
[[ $CAM =~ ^[A-Z_]+$ ]] || { echo "GLR_CAMERA_MODEL must look like PINHOLE or SIMPLE_PINHOLE"; exit 4; }
[ $DRY = 1 ] || [ -n "${GLRMAP_HOME:-}" ] || { echo "GLRMAP_HOME is not set"; exit 4; }

# whole-template wrapper: [setsid] [taskset -c LIST] [timeout N] bash -c TEMPLATE  (the cap applies to every command of the template)
WRAP=()
if [ -n "${GLR_TASKSET:-}" ]; then
  WRAP+=(taskset -c "$GLR_TASKSET")
elif [ $DRY = 0 ] && [ "${GLR_ALLOW_UNCAPPED:-0}" != 1 ]; then
  echo "GLR_TASKSET is not set. GLOMAP ignores OMP/MKL thread caps and takes every core (AGENTS.md rule 2)."
  echo "Set GLR_TASKSET=<cpu list> (e.g. 8-15), or GLR_ALLOW_UNCAPPED=1 on a host that is yours alone."
  exit 4
fi
[ -n "${GLR_TIMEOUT:-}" ] && WRAP+=(timeout --kill-after=60 "$GLR_TIMEOUT")
command -v setsid > /dev/null 2>&1 && WRAP=(setsid "${WRAP[@]}")   # own process group: the whole GLRMap tree can be stopped

CLIP=$(basename "$JOB")
IMAGES=$JOB/images
read -r NFR FIRST_IMG HFOV FOCAL_W < <("$PY" -c 'import json,math,sys
j = json.load(open(sys.argv[1])); h = (j.get("k_prior") or {}).get("hfov_deg")
print(len(j["image_names"]), j["image_names"][0], "%.4f" % h if h else "-", "%.5f" % (0.5 / math.tan(math.radians(h) / 2)) if h else "-")' "$JOB/job.json" 2> /dev/null)
[ "$HFOV" != - ] || HFOV=""; [ "$FOCAL_W" != - ] || FOCAL_W=""
if [ $DRY = 0 ]; then
  [ -n "$NFR" ] || { echo "cannot read image_names from $JOB/job.json with PY_GLR=$PY"; exit 4; }
  "$PY" -c 'import numpy, pycolmap' 2> /dev/null || { echo "PY_GLR=$PY cannot import numpy and pycolmap (the stage-1 gate needs them)"; exit 4; }
  [ -f "$IMAGES/$FIRST_IMG" ] || { echo "$IMAGES/$FIRST_IMG is missing or a broken symlink - copy the job with rsync -aL"; exit 4; }
  [ -z "${GLR_TASKSET:-}" ] || taskset -c "$GLR_TASKSET" true || { echo "GLR_TASKSET=$GLR_TASKSET is not a usable CPU list"; exit 4; }
fi
NFR=${NFR:-0}

RES=$JOB/result; n=0   # never overwrite an existing result; mkdir is atomic, so two workers cannot share a directory
if [ $DRY = 0 ]; then
  [ -w "$JOB" ] || { echo "$JOB is not writable"; exit 4; }
  until mkdir "$RES" 2> /dev/null; do
    n=$((n + 1)); RES=$JOB/result_$n
    [ $n -lt 10000 ] || { echo "cannot create a result directory in $JOB"; exit 4; }
  done
  mkdir -p "$RES/logs"
  printf '%s %s %s\n' "$$" "$(hostname)" "$(date -u +%FT%TZ)" > "$RES/RUNNING"
else
  while [ -e "$RES" ]; do n=$((n + 1)); RES=$JOB/result_$n; done
fi
STATUS=failed; GATE=""; DENSE_RAN=0; RC=0; CHILD=""; FINISHED=0; MODEL=""
CMDS=()
export GLR_IMAGES=$IMAGES GLR_OUT=$RES GLR_S1=$RES/stage1 GLR_THREADS=$THREADS GLR_PY=$PY GLR_FOCAL_W=$FOCAL_W GLR_HFOV=$HFOV

fill() {  # fill TEMPLATE : {images} {out} {s1} {threads} {py} {focal_w} {hfov} -> "${GLR_IMAGES}" ... (bash expands them when the template runs)
  local t=$1 o="" q="" c name hit i=0 n=${#1}
  while [ $i -lt $n ]; do
    c=${t:$i:1}; hit=""
    if [ "$c" = "{" ]; then
      for name in images out s1 threads py focal_w hfov; do
        [ "${t:$i:$((${#name} + 2))}" = "{$name}" ] && { hit=$name; break; }
      done
    fi
    if [ -n "$hit" ] && [ "$q" != "'" ]; then
      if [ "$q" = '"' ]; then o+="\${GLR_${hit^^}}"; else o+="\"\${GLR_${hit^^}}\""; fi
      i=$((i + ${#hit} + 2)); continue
    fi
    if [ "$c" = "\\" ] && [ "$q" != "'" ]; then o+="${t:$i:2}"; i=$((i + 2)); continue; fi
    case "$c" in
      "'") [ "$q" = '"' ] || { [ "$q" = "'" ] && q="" || q="'"; } ;;
      '"') [ "$q" = "'" ] || { [ "$q" = '"' ] && q="" || q='"'; } ;;
    esac
    o+=$c; i=$((i + 1))
  done
  printf '%s' "$o"
}
step() {  # step NAME COMMAND-STRING : `bash -c COMMAND-STRING` in GLRMAP_HOME under the wrapper, logged and timed
  local name=$1 cmd=$2 t0 t1 rc
  CMDS+=("$name: $cmd")
  echo "[$(date +%T)] >>> $name: $cmd"
  [ $DRY = 1 ] && return 0
  t0=$(date +%s%N)
  ( cd "$GLRMAP_HOME" && exec "${WRAP[@]}" bash -c "$cmd" ) >> "$RES/logs/$name.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"; rc=$?
  CHILD=""
  t1=$(date +%s%N)
  printf '{"step": "%s", "sec": %.2f, "rc": %d}\n' "$name" "$(( (t1 - t0) / 1000000 ))e-3" "$rc" >> "$RES/timing.jsonl"
  [ $rc -eq 0 ] || echo "step $name failed (rc=$rc), see $RES/logs/$name.log"
  return $rc
}
find_model() {  # shallowest sparse/<k>/cameras.{bin,txt} below $1 -> model dir (stdout); NUL-safe, ties broken by name
  local best="" f
  while IFS= read -r -d '' f; do
    if [ -z "$best" ] || [ ${#f} -lt ${#best} ] || { [ ${#f} -eq ${#best} ] && [[ $f < $best ]]; }; then best=$f; fi
  done < <(find "$1" -path '*/sparse/*/cameras.*' -print0 2> /dev/null)
  [ -n "$best" ] && dirname -- "$best"
}
run_gate() {  # run_gate MODEL_DIR -> $RES/gate.json ; sets GATE = ok | suspect | stop | error (never anything else)
  local model=$1 base db grc v
  GATE=error
  if [ -z "$model" ]; then
    printf '%s\n' '{"verdict": "stop", "stop": ["no_model"], "suspect": [], "signals": {}}' > "$RES/gate.json"
    GATE=stop; echo "stage-1 gate: stop (no sparse model found)"; return
  fi
  base=$(dirname "$(dirname "$model")")
  db=$base/database.db
  [ -f "$db" ] || db=$(find "$base" -maxdepth 2 -name database.db -print -quit 2> /dev/null)
  "$PY" "$HERE/glr_stage1_gate.py" "$model" "$db" "$NFR" --job "$JOB/job.json" --require-db --write "$RES/gate.json" \
    > "$RES/gate_full.json" 2> "$RES/logs/gate.log"
  grc=$?
  v=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$RES/gate.json" 2> /dev/null | tr -cd 'a-z_' | head -c 20)
  case "$v:$grc" in   # the verdict and the exit code must agree, otherwise the gate did not run properly
    ok:0 | suspect:1 | stop:3 | error:4) GATE=$v ;;
    *) GATE=error
       printf '{"verdict": "error", "stop": [], "suspect": [], "signals": {}, "error": "gate gave no consistent verdict (exit %s, verdict %s), see logs/gate.log"}\n' \
         "$grc" "${v:-none}" > "$RES/gate.json" ;;
  esac
  echo "stage-1 gate: $GATE (model $model)"
}
finish() {  # result.json first, RESULT_DIR last (a job that was killed hard has neither and will be run again)
  [ $DRY = 0 ] || return 0
  [ $FINISHED = 0 ] || return 0
  FINISHED=1
  export RES JOB CLIP MODE STATUS GATE DENSE_RAN NFR THREADS FORCE GLR_TASKSET GLRMAP_HOME MODEL
  CMDS_JSON=$(printf '%s\0' "${CMDS[@]}" | "$PY" -c 'import sys,json; print(json.dumps([l for l in sys.stdin.read().split("\0") if l.strip()]))' 2> /dev/null)
  export CMDS_JSON=${CMDS_JSON:-[]}
  "$PY" - << 'EOF' || printf '{"clip": "%s", "status": "%s", "gate": "%s", "dense_ran": false}\n' "$CLIP" "$STATUS" "${GATE:-error}" > "$RES/result.json"
import json, os, subprocess, socket
from pathlib import Path
res = Path(os.environ["RES"])
git = ""
try:
    git = subprocess.run(["git", "-C", os.environ["GLRMAP_HOME"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    pass
find = lambda pat: sorted(str(p.relative_to(res)) for p in res.glob(pat))[:3]
try:  # the sparse model the gate looked at: glr_collect builds the label from this very model
    gate_model = str(Path(os.environ["MODEL"]).relative_to(res)) if os.environ.get("MODEL") else ""
except ValueError:
    gate_model = ""
timing = [json.loads(l) for l in (res / "timing.jsonl").read_text().splitlines()] if (res / "timing.jsonl").exists() else []
r = {"clip": os.environ["CLIP"], "mode": os.environ["MODE"], "status": os.environ["STATUS"], "gate": os.environ["GATE"],
     "dense_ran": os.environ["DENSE_RAN"] == "1", "force": os.environ["FORCE"] == "1", "n_frames": int(os.environ["NFR"]),
     "threads": int(os.environ["THREADS"]), "taskset": os.environ.get("GLR_TASKSET", ""), "host": socket.gethostname(),
     "glrmap_commit": git, "gate_model": gate_model, "commands": json.loads(os.environ["CMDS_JSON"]), "timing": timing,
     "found": {"sparse_models": find("**/sparse/*/cameras.*"), "databases": find("**/database.db"),
               "pda_fused": find("**/fused/*.npz"), "pi3x_prior": find("**/prior/**/*.npz"),
               "roma_points": find("**/new_points.npz")}}
tmp = res / "result.json.tmp"
tmp.write_text(json.dumps(r, indent=1))
tmp.rename(res / "result.json")
print(json.dumps({k: r[k] for k in ("clip", "status", "gate", "dense_ran")}))
EOF
  rm -f "$RES/RUNNING"
  printf '%s\n' "$(basename "$RES")" > "$JOB/RESULT_DIR.tmp" && mv -f "$JOB/RESULT_DIR.tmp" "$JOB/RESULT_DIR"
}
on_signal() {  # on_signal EXIT_CODE : stop the GLRMap process group, record `interrupted`
  trap '' INT TERM HUP
  if [ -n "$CHILD" ]; then
    kill -TERM -- "-$CHILD" 2> /dev/null || kill -TERM "$CHILD" 2> /dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$CHILD" 2> /dev/null || break; sleep 1; done
    kill -KILL -- "-$CHILD" 2> /dev/null || kill -KILL "$CHILD" 2> /dev/null
  fi
  STATUS=interrupted; RC=$1; GATE=${GATE:-}
  finish
  exit "$RC"
}
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

if [ "$MODE" = mono ]; then
  RUN_CMD=${GLR_RUN_CMD:-bash data_generate_demo/run_all.sh}
  export IMAGES OUTPUT=$RES/run
  [ -d "$JOB/masks" ] && export MASK_DIR=$JOB/masks
  step run "$RUN_CMD"; RUN_RC=$?
  if [ $DRY = 0 ]; then
    MODEL=$(find_model "$RES/run"); run_gate "$MODEL"
    if [ $RUN_RC -eq 3 ]; then STATUS=native_gate_stop; RC=3
    elif [ $RUN_RC -ne 0 ]; then STATUS=failed; RC=5
    elif [ "$GATE" = stop ]; then STATUS=complete_gate_stop; DENSE_RAN=1; RC=3
    elif [ "$GATE" = error ]; then STATUS=complete_gate_error; DENSE_RAN=1; RC=5
    else STATUS=complete; DENSE_RAN=1; fi
  fi
else
  S1_CMD=${GLR_STAGE1_CMD:-"{py} -m reconstruction_flow.glomap --images {images} --output {s1} --camera-model $CAM"}
  step stage1 "$(fill "$S1_CMD")"; rc=$?
  if [ $DRY = 0 ]; then
    if [ $rc -ne 0 ]; then STATUS=failed; RC=5
    else
      MODEL=$(find_model "$RES/stage1"); run_gate "$MODEL"
      if [ "$GATE" = error ]; then STATUS=gate_error; RC=5
      elif [ "$GATE" = stop ] && { [ "$FORCE" != 1 ] || [ -z "$MODEL" ]; }; then STATUS=stopped_at_gate; RC=3
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

finish
exit $RC
