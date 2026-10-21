#!/bin/bash
# Run the downloaded VLMs over the pool (+ group C) on one GPU; each skips clips it already judged.
# Usage: run_vlms_pool.sh GPU TAG:PATH:TP [TAG:PATH:TP ...]
H=/data/home/z00533370; G=$1; shift
cd $H/code/filter
export VLLM_USE_DEEP_GEMM=0
export HF_HOME=$H/cache/huggingface VLLM_CACHE_ROOT=$H/cache/vllm VLLM_WORKER_MULTIPROC_METHOD=spawn
for spec in "$@"; do
  IFS=: read TAG MP TP <<< "$spec"
  CUDA_VISIBLE_DEVICES=$G $H/cache/vllm-venv/bin/python vlm_run.py $TAG $MP $TP $H/data/filter_pool $H/data/filter_pool/_calibC > $H/outputs/logs/filter/vlm_${TAG}_pool_$(date +%H%M).log 2>&1
  echo "DONE $TAG $(date +%T)"
done
