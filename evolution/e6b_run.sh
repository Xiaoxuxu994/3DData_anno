#!/bin/bash
# E6b: controlled ViPE cost test on one GPU (run inside the container).
# (a) per-clip `vipe infer` (current practice), (b) persistent worker (models loaded once),
# (c) persistent worker on half-frame-rate clips, (d) two persistent workers sharing one GPU.
# Usage: e6b_run.sh GPU
H=/data/home/z00533370; E=$H/outputs/evolution/E6; G=$1
PY=$H/code/vipe/.venv/bin/python
cd $H/code/evolution
export CUDA_VISIBLE_DEVICES=$G
mkdir -p $E/cli/logs $E/logs
for d in $(cat $E/lists/full.txt); do
  c=$(basename $d)
  t=$(date +%s.%N)
  $H/code/vipe/.venv/bin/vipe infer --image-dir $d -o $E/cli -p default > $E/logs/cli_$c.log 2>&1
  rc=$?
  t2=$(date +%s.%N)
  $PY -c "import json,sys; print(json.dumps({'clip': '$c', 'rc': $rc, 'sec': round($t2 - $t, 2)}))" >> $E/cli/logs/status.jsonl
done
$PY vipe_worker.py worker_full $E/worker_full $E/lists/full.txt > $E/logs/worker_full.log 2>&1
$PY vipe_worker.py worker_sub2 $E/worker_sub2 $E/lists/sub2.txt > $E/logs/worker_sub2.log 2>&1
t=$(date +%s)
$PY vipe_worker.py conc2 $E/conc2 $E/lists/full.txt 0 2 > $E/logs/conc2_0.log 2>&1 &
$PY vipe_worker.py conc2 $E/conc2 $E/lists/full.txt 1 2 > $E/logs/conc2_1.log 2>&1
wait
echo "conc2_wall $(( $(date +%s) - t ))" > $E/logs/conc2_wall.txt
echo E6B_DONE > $E/logs/done.txt
