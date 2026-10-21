#!/bin/bash
# Per-GPU job chains for the eval3d A/B + wild runs (GPUs 0-3 only; 4-7 belong to line F).
# Usage: run_eval3d_chains.sh <chain-id>   (0..3) ; logs -> outputs/bench/eval3d_logs/
H=/data/home/z00533370
O=$H/outputs/bench
A=$O/eval3d_A; B=$O/eval3d_B; W=$O/wild
L=$O/eval3d_logs; mkdir -p $L
export OMP_NUM_THREADS=16
wait_free() {  # block until GPU $1 has < 2 GB used
  until [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 60; done
}
mark() { echo "$(date +%F_%T) $*" >> $L/chains.log; }
case $1 in
0)  wait_free 0; mark "g0 vggt_omega start"
    (cd $H/code/vggt-omega && CUDA_VISIBLE_DEVICES=0 .venv/bin/python $H/code/bench/vggt_omega_bench.py g0 $A $B $W > $L/vggt_omega_g0.log 2>&1)
    mark "g0 vggt_omega done"; wait_free 0; mark "g0 voyager start"
    (cd $H/code/HunyuanWorld-Voyager/data_engine && CUDA_VISIBLE_DEVICES=0 .venv/bin/python voyager_bench.py g0 $A $B $W > $L/voyager_g0.log 2>&1)
    mark "g0 voyager done" ;;
1)  wait_free 1; mark "g1 da3 start"
    (cd $H/code/depth-anything-3 && CUDA_VISIBLE_DEVICES=1 .venv/bin/python $H/code/bench/da3_bench.py g1 $A $B $W > $L/da3_g1.log 2>&1)
    mark "g1 da3 done"; wait_free 1; mark "g1 voyager start"
    (cd $H/code/HunyuanWorld-Voyager/data_engine && CUDA_VISIBLE_DEVICES=1 .venv/bin/python voyager_bench.py g1 $A $B $W > $L/voyager_g1.log 2>&1)
    mark "g1 voyager done" ;;
2)  wait_free 2; mark "g2 pi3x start"
    (cd $H/code/Pi3 && CUDA_VISIBLE_DEVICES=2 .venv/bin/python $H/code/bench/pi3x_bench.py g2 $A $B $W > $L/pi3x_g2.log 2>&1)
    mark "g2 pi3x done"
    until [ -f $L/vipe.done ]; do sleep 60; done
    wait_free 2; wait_free 3
    for b in $A $B $W; do mark "svid $(basename $b) start"
      B=$b GPUS=2,3 bash $H/code/bench/svid_bench.sh > $L/svid_$(basename $b).log 2>&1
      mark "svid $(basename $b) done"; done ;;
3)  wait_free 3; mark "g3 vipe start"
    cd $H/code/vipe
    for b in $A $B $W; do .venv/bin/python $H/code/bench/vipe_bench.py $b 3 2 > $L/vipe_$(basename $b).log 2>&1; mark "vipe $(basename $b) done"; done
    touch $L/vipe.done ;;
esac
