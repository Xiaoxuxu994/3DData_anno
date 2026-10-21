#!/bin/bash
# E12 chain: one worker processes every (MODEL, MODE) queue over all benches; claim files share work between workers.
# usage: run_chain.sh GPU WID
H=/data/home/z00533370
export HF_HOME=$H/cache/huggingface CUDA_VISIBLE_DEVICES=$1
O=$H/outputs/bench
for m in m2l m3l m3g; do for mode in free kgate; do
  $H/cache/moge-venv/bin/python $H/code/evolution/E12_moge/moge_run.py $2 $m $mode $O/eval3d_A $O/eval3d_B $O/scannetpp $O/wild \
    >> $H/outputs/evolution/E12/logs/run_${m}_${mode}_$2.log 2>&1
done; done
echo CHAIN_DONE >> $H/outputs/evolution/E12/logs/chain_$2.log
