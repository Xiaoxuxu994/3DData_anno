#!/bin/bash
# Supplement for the A group after TUM (8) / Bonn (8) were added: feed-forward methods + ViPE on new clips.
# Voyager / SpatialVID read clips.json when they reach A, so they pick the new clips up by themselves.
H=/data/home/z00533370
O=$H/outputs/bench; A=$O/eval3d_A; L=$O/eval3d_logs
export OMP_NUM_THREADS=16
mark() { echo "$(date +%F_%T) $*" >> $L/chains.log; }
case $1 in
0)  mark "supp g0 vggt_omega A start"
    (cd $H/code/vggt-omega && CUDA_VISIBLE_DEVICES=0 .venv/bin/python $H/code/bench/vggt_omega_bench.py s0 $A > $L/vggt_omega_supp.log 2>&1)
    mark "supp g0 pi3x A start"
    (cd $H/code/Pi3 && CUDA_VISIBLE_DEVICES=0 .venv/bin/python $H/code/bench/pi3x_bench.py s0 $A > $L/pi3x_supp.log 2>&1)
    mark "supp g0 vipe A start"
    (cd $H/code/vipe && .venv/bin/python $H/code/bench/vipe_bench.py $A 0 2 > $L/vipe_eval3d_A_supp.log 2>&1)
    mark "supp g0 done" ;;
1)  mark "supp g1 da3 A start"
    (cd $H/code/depth-anything-3 && CUDA_VISIBLE_DEVICES=1 .venv/bin/python $H/code/bench/da3_bench.py s1 $A > $L/da3_supp.log 2>&1)
    mark "supp g1 done" ;;
esac
