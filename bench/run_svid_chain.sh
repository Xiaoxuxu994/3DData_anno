#!/bin/bash
# SpatialVID chain (replaces chain 2's svid part, which was blocked by the extra Voyager worker on GPU 2):
# A and B on GPU 3 right away, wild on GPUs 0-3 once all Voyager workers are done.
H=/data/home/z00533370
O=$H/outputs/bench; L=$O/eval3d_logs
mark() { echo "$(date +%F_%T) $*" >> $L/chains.log; }
for b in eval3d_A eval3d_B; do mark "svid $b start (GPU 3)"
  B=$O/$b GPUS=3 bash $H/code/bench/svid_bench.sh > $L/svid_$b.log 2>&1; mark "svid $b done"; done
while pgrep -f voyager_bench.py > /dev/null; do sleep 60; done
mark "svid wild start (GPU 0-3)"
B=$O/wild GPUS=0,1,2,3 bash $H/code/bench/svid_bench.sh > $L/svid_wild.log 2>&1; mark "svid wild done"
