#!/bin/bash
# E15 stage-1 sweep for ONE clip, CPU only: the pycolmap replica of GLRMap's first stage (tools/stage1_sfm.py), its camera-model / prior variants and
# the silver reference - exactly the recipe that produced the tables of docs/experiments/LOG.md (E15). Not GLRMap itself.
#
# Usage: run_sweep.sh FRAMES_DIR OUT_DIR [--focal-w F] [--hq960] [--no-ref] [--dry-run]
#   FRAMES_DIR   the clip's frames (15 fps, 000000.jpg ... 000179.jpg, as pipeline/prep.py writes them)
#   OUT_DIR      new directory (must not exist); writes OUT_DIR/{v0a,v0b,v1sp,v2prior,v3fixed,ref[,v0a_hq960]}/ for evolution/E15_glr/summarize_sweep.py
#   --focal-w F  GeoCalib / Pi3X / DA3 median focal / image width: the prior for v2prior and v3fixed (without it those two are skipped)
#   --hq960      also run v0a again at 960 px (resolution control: only for clips whose reduced run stopped on resolution-dependent signals)
#   --no-ref     skip the silver reference
# Runs (all but v0a and ref reuse v0a's matching database, so only the mapper runs again):
#   v0a        PINHOLE, initial focal 1.2 x max(W, H), focal refined, principal point fixed; 640 px long side, every 3rd frame (60), sequential
#              matching with overlap 12, no guided matching, global mapper    <- the reduced-cost replica, harsher than the real stage 1
#   v0b        v0a with another mapper seed (stability)
#   v1sp       SIMPLE_PINHOLE (fx = fy)
#   v2prior    PINHOLE with the prior focal as the start value, still refined
#   v3fixed    PINHOLE with the prior focal fixed
#   ref        every 6th frame, exhaustive matching, SIMPLE_RADIAL, incremental mapper (the silver reference: a second SfM reconstruction, not truth)
# Environment: PY (python with numpy scipy pycolmap, default python3), THREADS (default ${ANNO_THREADS:-4}); the commands are printed before they run.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S1=$HERE/../../tools/stage1_sfm.py
DRY=0 HQ=0 REF=1 FW=""
POS=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --hq960) HQ=1 ;;
    --no-ref) REF=0 ;;
    --focal-w) shift; [ $# -gt 0 ] || { echo "--focal-w needs a value"; exit 2; }; FW=$1 ;;
    --focal-w=*) FW=${1#*=} ;;
    -h | --help) sed -n 2,23p "$0"; exit 0 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) POS+=("$1") ;;
  esac
  shift
done
[ ${#POS[@]} -eq 2 ] || { sed -n 2,23p "$0"; exit 2; }
FRAMES=${POS[0]} OUT=${POS[1]}
PY=${PY:-python3}
T=${THREADS:-${ANNO_THREADS:-4}}
[ -d "$FRAMES" ] || { echo "no frames directory $FRAMES"; exit 2; }
[ ! -e "$OUT" ] || { echo "$OUT exists - the sweep writes a new directory (never overwrites)"; exit 2; }
if [ -n "$FW" ]; then
  "$PY" -c 'import sys; v = float(sys.argv[1]); assert 0.2 <= v <= 5, v' "$FW" 2> /dev/null || { echo "--focal-w $FW is not a plausible focal / width (0.2 .. 5)"; exit 2; }
fi

run() {
  echo "+ $*"
  [ $DRY = 1 ] || "$@"
}
[ $DRY = 1 ] || mkdir -p "$OUT"
BASE=(--stride 3 --long 640 --overlap 12 --guided 0 --seed 1 --threads "$T")
run "$PY" "$S1" --images "$FRAMES" --out "$OUT/v0a" "${BASE[@]}"
REUSE=("$PY" "$S1" --threads "$T" --reuse-db "$OUT/v0a")
run "${REUSE[@]}" --seed 2 --out "$OUT/v0b"
run "${REUSE[@]}" --seed 1 --db-camera SIMPLE_PINHOLE --out "$OUT/v1sp"
if [ -n "$FW" ]; then
  run "${REUSE[@]}" --seed 1 --focal-w "$FW" --out "$OUT/v2prior"
  run "${REUSE[@]}" --seed 1 --focal-w "$FW" --fixed-focal 1 --out "$OUT/v3fixed"
else
  echo "(no --focal-w: v2prior and v3fixed skipped)"
fi
[ $REF = 0 ] || run "$PY" "$S1" --images "$FRAMES" --out "$OUT/ref" --stride 6 --long 640 --camera-model SIMPLE_RADIAL --matcher exhaustive --mapper incremental --guided 0 --seed 1 --threads "$T"
[ $HQ = 0 ] || run "$PY" "$S1" --images "$FRAMES" --out "$OUT/v0a_hq960" --stride 3 --long 960 --overlap 12 --guided 0 --seed 1 --threads "$T"
echo "done: $OUT  (next: python evolution/E15_glr/summarize_sweep.py <parent of the clip directories> <summary dir> [--pool POOL_DIR])"
