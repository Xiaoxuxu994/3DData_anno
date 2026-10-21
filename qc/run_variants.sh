#!/bin/bash
# Stability re-runs (line G): GPU 0 = ViPE; GPU 1 = VGGT-Omega first, then ViPE. Only starts on a GPU with < 2 GB used.
H=/data/home/z00533370
V=$H/outputs/qc/variants
L=$H/outputs/qc/logs; mkdir -p $L
BS="$V/A_rev $V/B_rev $V/C_rev $V/A_sub2 $V/B_sub2 $V/C_sub2"
wait_free() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 60; done; }
mark() { echo "$(date +%F_%T) $*" >> $L/chains.log; }
case $1 in
0) wait_free 0; mark "g0 vipe start"
   (cd $H/code/vipe && .venv/bin/python $H/code/qc/vipe_variant.py 0 3 $BS > $L/vipe_g0.log 2>&1); mark "g0 vipe done" ;;
1) wait_free 1; mark "g1 vggt_omega start"
   (cd $H/code/vggt-omega && CUDA_VISIBLE_DEVICES=1 .venv/bin/python $H/code/bench/vggt_omega_bench.py q1 $BS > $L/vggt_omega_g1.log 2>&1)
   mark "g1 vggt_omega done"; wait_free 1; mark "g1 vipe start"
   (cd $H/code/vipe && .venv/bin/python $H/code/qc/vipe_variant.py 1 3 $BS > $L/vipe_g1.log 2>&1); mark "g1 vipe done" ;;
esac
