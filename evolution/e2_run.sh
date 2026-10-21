#!/bin/bash
# E2: run every filter stage + Qwen3-VL-32B on the single-shot sub-clips, then decide.py (same rules as the pool).
# Usage: e2_run.sh GPU
H=/data/home/z00533370; G=$1; P=$H/data/filter_pool_shots; L=$H/outputs/evolution/E2/logs
mkdir -p $L; cd $H/code/filter
export TORCH_HOME=$H/cache/torch HF_HOME=$H/cache/huggingface FFMPEG=$H/cache/bin/ffmpeg PADDLE_PDX_CACHE_HOME=$H/cache/paddlex OMP_NUM_THREADS=8
FV=$H/cache/filter-venv/bin/python
T0=$(date +%s)
($H/cache/ocr-venv/bin/python s3_ocr.py 0 1 $P > $L/ocr.log 2>&1; echo "OCR_END $(date +%s)" >> $L/times.log) &
for s in cuts rules flow track; do
  t=$(date +%s)
  case $s in
    cuts)  CUDA_VISIBLE_DEVICES=$G $FV s1_cuts.py 0 1 $P ;;
    rules) CUDA_VISIBLE_DEVICES=$G $FV s2_rules.py 0 1 $P ;;
    flow)  CUDA_VISIBLE_DEVICES=$G $FV s4_flow.py 0 1 $P ;;
    track) CUDA_VISIBLE_DEVICES=$G $FV s6_track.py 0 1 $P ;;
  esac > $L/$s.log 2>&1
  echo "$s $(( $(date +%s) - t ))" >> $L/times.log
done
t=$(date +%s); CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$H/code/vipe:$H/code/filter $H/code/vipe/.venv/bin/python s5_geocalib.py 0 1 $P > $L/calib.log 2>&1; echo "calib $(( $(date +%s) - t ))" >> $L/times.log
export VLLM_USE_DEEP_GEMM=0 VLLM_CACHE_ROOT=$H/cache/vllm VLLM_WORKER_MULTIPROC_METHOD=spawn
t=$(date +%s); CUDA_VISIBLE_DEVICES=$G $H/cache/vllm-venv/bin/python vlm_run.py qwen3vl_32b $H/cache/modelscope/Qwen/Qwen3-VL-32B-Instruct 1 $P > $L/vlm32b.log 2>&1; echo "vlm32b $(( $(date +%s) - t ))" >> $L/times.log
wait
$FV decide.py $P $H/outputs/evolution/E2/decisions_shots.csv --vlm qwen3vl_32b > $L/decide.log 2>&1
echo "E2_DONE total $(( $(date +%s) - T0 ))" >> $L/times.log
