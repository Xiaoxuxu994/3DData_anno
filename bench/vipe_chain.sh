#!/bin/bash
# ViPE: full ScanNet++ (87 segs) -> ScanNet++ bench -> YouTube bench, GPUs 0-3 (4-7 belong to others)
cd /data/home/z00533370/code/vipe
.venv/bin/python /data/home/z00533370/code/vipe_eval/run_multi_gpu.py --gpus 0,1,2,3 > /data/home/z00533370/outputs/vipe/scannetpp_sample30/logs/launcher.log 2>&1
.venv/bin/python /data/home/z00533370/code/bench/vipe_bench.py /data/home/z00533370/outputs/bench/scannetpp 0,1,2,3 1 > /data/home/z00533370/outputs/logs/vipe_bench_scannetpp.log 2>&1
until grep -q YT_EXIT /data/home/z00533370/outputs/logs/yt_clip.log; do sleep 30; done
/data/home/z00533370/cache/eval-venv/bin/python /data/home/z00533370/code/bench/make_yt_bench.py > /data/home/z00533370/outputs/logs/make_yt_bench.log 2>&1
.venv/bin/python /data/home/z00533370/code/bench/vipe_bench.py /data/home/z00533370/outputs/bench/youtube 0,1,2,3 1 > /data/home/z00533370/outputs/logs/vipe_bench_youtube.log 2>&1
echo VIPE_CHAIN_DONE > /data/home/z00533370/outputs/logs/vipe_chain.done
