#!/bin/bash
# VLM weights from ModelScope (~2-3x faster than HF from this host); Qwen3-VL-8B keeps downloading via HF.
for r in Qwen/Qwen3-VL-32B-Instruct OpenGVLab/InternVL3_5-8B-HF Qwen/Qwen3.5-27B Qwen/Qwen3.5-122B-A10B-FP8 Qwen/Qwen3-VL-235B-A22B-Instruct-FP8; do
  echo "== $r $(date +%T)"
  for k in 1 2 3; do
    /data/home/z00533370/cache/ms-venv/bin/modelscope download --model $r --local_dir /data/home/z00533370/cache/modelscope/$r --max-workers 8 > /data/home/z00533370/outputs/logs/dl_ms_$(basename $r).log 2>&1 && { echo "OK $r $(date +%T)"; break; } || echo "retry $k $r"
  done
done
echo DL_ALL_DONE
