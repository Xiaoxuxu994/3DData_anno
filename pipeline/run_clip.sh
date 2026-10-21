#!/bin/bash
# v1.3 pipeline for one clip: filter -> annotate -> QC.
# Usage: pipeline/run_clip.sh INPUT OUT [options of run_batch.sh]
#   INPUT  video (decoded at 15 fps, first 12 s) or frame directory; OUT  output directory
# Prints the filter verdict, QC v1c score / flags and the label path. See pipeline/README.md.
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { sed -n 2,5p "$0"; exit 2; }
IN=$1; OUT=$2; shift 2
"$HERE/run_batch.sh" "$OUT" "$IN" "$@"
