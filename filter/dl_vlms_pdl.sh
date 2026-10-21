#!/bin/bash
# Sequential VLM downloads with pdl.py (chunked parallel ranges from ModelScope).
for r in OpenGVLab/InternVL3_5-8B-HF Qwen/Qwen3.5-27B Qwen/Qwen3.5-122B-A10B-FP8 Qwen/Qwen3-VL-235B-A22B-Instruct-FP8; do
  echo "== $r $(date +%T)"
  for k in 1 2 3; do
    /data/home/z00533370/cache/ms-venv/bin/python /data/home/z00533370/code/filter/pdl.py $r /data/home/z00533370/cache/modelscope/$r 6 >> /data/home/z00533370/outputs/logs/pdl_$(basename $r).log 2>&1 && { echo "OK $r $(date +%T)"; break; } || echo "retry $k $r"
  done
done
echo DL_ALL_DONE
