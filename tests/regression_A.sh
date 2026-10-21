#!/bin/bash
# Regression test of the v1.3 annotate + QC path on A-group GT clips (no filter layer).
# Usage: tests/regression_A.sh [OUT] [--gpus=4,5] [--strict] [CLIP ...]
#   OUT    default ROOT/outputs/pipeline_test/regression_A (resumable: finished steps are skipped)
#   CLIP   clip names from outputs/bench/eval3d_A/clips.json; default: one TUM, 7-Scenes, KITTI and Bonn clip
# Runs pipeline/run_batch.sh --skip-filter on the clips, then tests/regression_A.py (tolerances and references are
# documented there). ~5 min on 2 idle H200 for the 4 default clips. Exit code 0 = PASS.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../pipeline/common.sh"
OUT=$ROOT/outputs/pipeline_test/regression_A; OPTS=(); STRICT=""; CLIPS=()
for a in "$@"; do
  case $a in
    --gpus=*) OPTS+=("$a") ;;
    --strict) STRICT=--strict ;;
    /*|./*) OUT=$a ;;
    *) CLIPS+=("$a") ;;
  esac
done
[ ${#CLIPS[@]} -gt 0 ] || CLIPS=(tum__freiburg1_room 7scenes__fire_seq03 kitti__kitti_07 bonn__balloon)
A=$ROOT/outputs/bench/eval3d_A
INPUTS=(); for c in "${CLIPS[@]}"; do INPUTS+=("$A/frames/$c"); done
"$HERE/../pipeline/run_batch.sh" "$OUT" "${INPUTS[@]}" --skip-filter --gt-bench=$A "${OPTS[@]}" || exit 1
$PY_QC "$HERE/regression_A.py" "$OUT" $STRICT
