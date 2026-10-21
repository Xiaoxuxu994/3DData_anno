#!/bin/bash
# E12 CPU evaluation chain (thread-capped for the shared host). usage: run_eval.sh KIND [KIND ...]
cd /data/home/z00533370/code/evolution/E12_moge
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
P=../../SpatialVID/.venv/bin/python; L=/data/home/z00533370/outputs/evolution/E12/logs
for k in "$@"; do
  $P e12_scales.py $k >> $L/scales.log 2>&1
  $P e12_eval.py $k --workers=20 > $L/eval_$k.log 2>&1
done
echo EVAL_DONE "$@" >> $L/eval_chain.log
