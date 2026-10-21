#!/bin/bash
# CPU chain: signals after the flow/match cache, cross-model + stability after the variant re-runs, then calibration.
H=/data/home/z00533370; L=$H/outputs/qc/logs; PY=$H/cache/qc-venv/bin/python
cd $H/code/qc
until grep -q CACHE_DONE $L/cache.log; do sleep 60; done
OMP_NUM_THREADS=2 $PY qc_signals.py A B C > $L/signals.log 2>&1
until grep -q "g0 vipe done" $L/chains.log && grep -q "g1 vipe done" $L/chains.log; do sleep 60; done
OMP_NUM_THREADS=2 $PY qc_cross.py A B C > $L/cross.log 2>&1
$PY qc_calib.py > $L/calib.log 2>&1
echo QC_CHAIN_DONE >> $L/calib.log
