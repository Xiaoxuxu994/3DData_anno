#!/bin/bash
# Sharded ViPE over filter_check clips (vipe_bench.py lost workers to an mp.Queue get_nowait race).
# Usage: fc_vipe_shard.sh GPU WID NW
H=/data/home/z00533370; B=$H/outputs/bench/filter_check; O=$B/vipe
cd $H/code/vipe
$H/cache/filter-venv/bin/python -c "import json;[print(c[\"clip\"]) for c in json.load(open(\"$B/clips.json\"))]" | awk -v w=$2 -v n=$3 "NR%n==w" | while read c; do
  [ -f $O/depth/$c.zip ] && continue
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=12 $H/code/vipe/.venv/bin/vipe infer --image-dir $B/frames/$c -o $O -p default > $O/logs/$c.log 2>&1
  echo "{\"clip\": \"$c\", \"n_frames\": 180, \"gpu\": \"$1\", \"rc\": $?, \"sec\": $(( $(date +%s)-t0 ))}" >> $O/logs/status.jsonl
done
echo SHARD_DONE $2 >> $H/outputs/logs/filter/fc_vipe_shard.log
