#!/bin/bash
# Replaces run_svid_chain.sh after svid A started: B on GPU 3 after A, wild on GPUs 0-2 as soon as Voyager is done.
H=/data/home/z00533370
O=$H/outputs/bench; L=$O/eval3d_logs
mark() { echo "$(date +%F_%T) $*" >> $L/chains.log; }
case $1 in
B)  while pgrep -f "svid_bench.sh" > /dev/null && [ ! -f $O/eval3d_A/spatialvid/.svid_done ]; do
      grep -q SVID_BENCH_DONE $L/svid_eval3d_A.log && break; sleep 60; done
    while pgrep -f "cvd_opt/inference_batch.py --csv_path $O/eval3d_A" > /dev/null; do sleep 60; done
    mark "svid eval3d_A done"; mark "svid eval3d_B start (GPU 3)"
    B=$O/eval3d_B GPUS=3 bash $H/code/bench/svid_bench.sh > $L/svid_eval3d_B.log 2>&1; mark "svid eval3d_B done" ;;
W)  while pgrep -f voyager_bench.py > /dev/null; do sleep 60; done
    mark "svid wild start (GPU 0-2)"
    B=$O/wild GPUS=0,1,2 bash $H/code/bench/svid_bench.sh > $L/svid_wild.log 2>&1; mark "svid wild done" ;;
esac
