#!/bin/bash
# Re-run one filter stage over the growing pool every 10 min until data/filter_pool/STOP_STAGES exists (stages skip done clips).
# Usage: run_stage_loop.sh STAGE GPU     STAGE in cuts|rules|ocr|flow|track|calib
H=/data/home/z00533370; S=$1; G=$2; P=$H/data/filter_pool; L=$H/outputs/logs/filter
cd $H/code/filter
export TORCH_HOME=$H/cache/torch HF_HOME=$H/cache/huggingface FFMPEG=$H/cache/bin/ffmpeg PADDLE_PDX_CACHE_HOME=$H/cache/paddlex OMP_NUM_THREADS=8
FV=$H/cache/filter-venv/bin/python
while true; do
  last=0; [ -f $P/STOP_STAGES ] && last=1
  case $S in
    cuts)  CUDA_VISIBLE_DEVICES=$G $FV s1_cuts.py 0 1 ;;
    rules) CUDA_VISIBLE_DEVICES=$G $FV s2_rules.py 0 1 ;;
    ocr)   $H/cache/ocr-venv/bin/python s3_ocr.py 0 1 ;;
    flow)  CUDA_VISIBLE_DEVICES=$G $FV s4_flow.py 0 1 ;;
    track) CUDA_VISIBLE_DEVICES=$G $FV s6_track.py 0 1 ;;
    calib) CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$H/code/vipe:$H/code/filter $H/code/vipe/.venv/bin/python s5_geocalib.py 0 1 ;;
  esac >> $L/pool_$S.log 2>&1
  [ $last = 1 ] && { echo "LOOP_DONE $S" >> $L/pool_$S.log; break; }
  sleep 600
done
