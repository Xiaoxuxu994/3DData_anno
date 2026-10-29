#!/bin/bash
# E15 sandbox demo: the GLR track (pipeline/run_batch_glr.sh) on real clips on a CPU-only machine, with stand-ins where the GPU / GLRMap parts are.
#
# What is real:     the filter scores (tools/filter_stage_cpu.py runs filter/s1..s5 unchanged), filter/decide.py (rules + a VLM verdict), the
#                   router and job packages (glr_prepare.py), the worker with its stage-1 exit gate (glr_worker.sh), collect + QC + merge.
# What is a stand-in: stage 1 = the pycolmap replica tools/stage1_sfm.py (not GLRMap), the dense stage = evolution/E15_glr/dense_standin.py
#                   (sparse points interpolated, not RoMa / Pi3X / PDA), the witnesses = evolution/E15_glr/witness_standin.py (a second SfM
#                   reconstruction, not ViPE / DA3). The v1.3 side (labels/, results.csv) does not exist, so there is no fallback label.
# The numbers therefore show that the pieces fit together on real clips and how the gate / QC rules react - not how accurate GLRMap is.
#
# Usage: run_demo.sh RUN S1_ROOT OUT CLIP [CLIP ...]
#   RUN      run directory: bench/{clips.json, frames/<clip>/}, pool/<clip>/scores.json (prep.py + filter stages + a `vlm_claude` verdict)
#   S1_ROOT  tools/stage1_sfm.py outputs; S1_ROOT/<clip>/ref is the silver reference used by the witness stand-in
#   OUT      new run directory (must not exist)
# Environment: PY (python with numpy scipy opencv pycolmap pandas, default python3), PY_FILTER (python with pandas for decide.py, default $PY),
#              GLR_TASKSET (default 0-3), VLM_TAG (default claude), STAGE1_ARGS (default "--long 640 --overlap 12 --guided 0 --seed 1 --threads 4")
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 4 ] || { sed -n 2,19p "$0"; exit 2; }
RUN=$(realpath "$1") S1_ROOT=$(realpath "$2") OUT=$(realpath -m "$3"); shift 3
CLIPS=("$@")
PY=${PY:-python3}; PY_FILTER=${PY_FILTER:-$PY}
[ ! -e "$OUT" ] || { echo "$OUT exists - the demo writes a new directory"; exit 2; }
mkdir -p "$OUT/bench" "$OUT/filter" "$OUT/pool" "$OUT/logs"
ROOT_TMP=$OUT/.root                       # throw-away ANNO_ROOT so common.sh finds code/ and a python for the QC steps
mkdir -p "$ROOT_TMP/cache/qc-venv/bin"
ln -s "$REPO" "$ROOT_TMP/code"
printf '#!/bin/bash\nexec %s "$@"\n' "$(command -v "$PY")" > "$ROOT_TMP/cache/qc-venv/bin/python"
chmod +x "$ROOT_TMP/cache/qc-venv/bin/python"
export ANNO_ROOT=$ROOT_TMP ANNO_CONTAINER= ANNO_THREADS=${ANNO_THREADS:-4}

# the demo's view of the run: only the chosen clips
ln -s "$RUN/bench/frames" "$OUT/bench/frames"
for c in "${CLIPS[@]}"; do ln -s "$RUN/pool/$c" "$OUT/pool/$c"; done
"$PY" - "$RUN/bench/clips.json" "$OUT/bench" "${CLIPS[@]}" << 'EOF'
import json, sys
from pathlib import Path
allc = {c["clip"]: c for c in json.loads(Path(sys.argv[1]).read_text())}
sel = [allc[c] for c in sys.argv[3:]]
for name in ("clips.json", "clips_all.json"):
    (Path(sys.argv[2]) / name).write_text(json.dumps(sel))
EOF
# the K prior of the jobs (and the gate's K witness) = GeoCalib's median HFOV, written where the v1.3 run writes bench/geocalib/<clip>.json
"$PY" - "$OUT" "${CLIPS[@]}" << 'EOF'
import json, math, sys
from pathlib import Path
out = Path(sys.argv[1])
(out / "bench" / "geocalib").mkdir()
meta = {c["clip"]: c for c in json.loads((out / "bench" / "clips.json").read_text())}
for clip in sys.argv[2:]:
    s = json.loads((out / "pool" / clip / "scores.json").read_text())
    if "calib" not in s:
        continue
    H, W = meta[clip]["hw"]
    v = sorted(s["calib"]["vfov_deg"])
    vf = float(v[len(v) // 2])
    (out / "bench" / "geocalib" / f"{clip}.json").write_text(json.dumps(
        {"vfov": vf, "hfov": math.degrees(2 * math.atan(math.tan(math.radians(vf) / 2) * W / H)), "W": W, "H": H}))
EOF
"$PY_FILTER" "$REPO/filter/decide.py" "$OUT/pool" "$OUT/filter/decisions.csv" --vlm "${VLM_TAG:-claude}" > "$OUT/logs/decide.log" 2>&1
echo "== filter decisions"; "$PY" -c "
import csv
for r in csv.DictReader(open('$OUT/filter/decisions.csv')): print(f\"  {r['clip']:16s} {r['decision']:7s} {r.get('reason','')}\")"

echo "== phase a: route + job packages"
"$REPO/pipeline/run_batch_glr.sh" "$OUT" --phase=a
cat "$OUT/filter/glr_route.csv" | "$PY" -c "
import csv, sys
for r in csv.DictReader(sys.stdin): print(f\"  {r['clip']:16s} route={r['route']:6s} {r['reasons']}\")"

echo "== phase b: GLRMap worker (split mode: replica stage 1 -> gate -> dense stand-in)"
export GLRMAP_HOME=$REPO PY_GLR=$(command -v "$PY") GLR_TASKSET=${GLR_TASKSET:-0-3}
export GLR_STAGE1_CMD="{py} $REPO/tools/stage1_sfm.py --images {images} --out {s1} ${STAGE1_ARGS:---long 640 --overlap 12 --guided 0 --seed 1 --threads 4}"
export GLR_DENSE_CMD="{py} $REPO/evolution/E15_glr/dense_standin.py {s1} {out}/dense"
"$REPO/pipeline/run_batch_glr.sh" "$OUT" --phase=b --worker-mode=split || echo "(some GLRMap jobs failed - see the summary above)"

echo "== witnesses (stand-ins) and phase c"
JOBS=()
for d in "$OUT"/glr_jobs/*/; do [ -f "$d/job.json" ] && JOBS+=("$(basename "$d")"); done
if [ ${#JOBS[@]} -gt 0 ]; then
  "$PY" "$REPO/evolution/E15_glr/witness_standin.py" "$S1_ROOT" "$OUT/pool" "$OUT/glr_jobs" "$OUT/witness" "${JOBS[@]}"
fi
"$REPO/pipeline/run_batch_glr.sh" "$OUT" --phase=c --witness-dir="$OUT/witness"
