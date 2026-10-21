#!/bin/bash
# Large VLMs for the comparison (approved 2026-10-17): 235B FP8 first, then Qwen3.5-122B FP8.
for r in Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 Qwen/Qwen3.5-122B-A10B-FP8; do
  echo "== $r start $(date +%F_%T)"
  for k in 1 2 3; do
    /data/home/z00533370/cache/ms-venv/bin/python /data/home/z00533370/code/filter/pdl.py $r /data/home/z00533370/cache/modelscope/$r 12 >> /data/home/z00533370/outputs/logs/pdl_$(basename $r).log 2>&1 && { echo "OK $r $(date +%F_%T)"; break; } || echo "retry $k $r"
  done
done
echo DL_BIG_DONE $(date +%F_%T)
