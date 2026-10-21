#!/bin/bash
export HF_HOME=/data/home/z00533370/cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0
for r in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3.5-27B Qwen/Qwen3-VL-32B-Instruct OpenGVLab/InternVL3_5-8B-HF Qwen/Qwen3.5-122B-A10B-FP8 Qwen/Qwen3-VL-235B-A22B-Instruct-FP8; do
  echo "== $r $(date +%T)"; /data/home/z00533370/cache/dl-venv/bin/hf download $r > /dev/null 2>&1 && echo "OK $r $(date +%T)" || echo "FAIL $r"
done
echo DL_ALL_DONE
