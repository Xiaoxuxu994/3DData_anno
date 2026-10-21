# Shared settings for the v1.3 pipeline scripts (sourced, not executed).
# Everything lives under ROOT = ${ANNO_ROOT:-/data/home/z00533370}: ROOT/code (this repo + third-party repos),
# ROOT/cache (Python envs, model weights), ROOT/outputs, ROOT/data.

ROOT=${ANNO_ROOT:-/data/home/z00533370}
CODE=$ROOT/code
PIPE=$CODE/pipeline

# Python environments (same ones the experiment scripts used)
PY_VIPE=$CODE/vipe/.venv/bin/python             # ViPE + GeoCalib (ViPE's copy)
PY_PI3=$CODE/Pi3/.venv/bin/python               # Pi3X
PY_DA3=$CODE/depth-anything-3/.venv/bin/python  # Depth Anything 3
PY_FILTER=$ROOT/cache/filter-venv/bin/python    # filter stages: cuts, rules, flow; decide.py
PY_OCR=$ROOT/cache/ocr-venv/bin/python          # filter stage: OCR (CPU)
PY_VLLM=$ROOT/cache/vllm-venv/bin/python        # filter stage: VLM judge (vLLM)
PY_QC=$ROOT/cache/qc-venv/bin/python            # scale, QC, evaluation (numpy / scipy / OpenEXR)

# GPU jobs run inside this container when it is set (ViPE was built and timed there); empty = run on the host.
ANNO_CONTAINER=${ANNO_CONTAINER-z00533370_dev}

# Model locations / caches
VLM_TAG=${VLM_TAG:-qwen3vl_32b}
VLM_PATH=${VLM_PATH:-$ROOT/cache/modelscope/Qwen/Qwen3-VL-32B-Instruct}

# Shared server: cap CPU threads per process
THREADS=${ANNO_THREADS:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export HF_HOME=$ROOT/cache/huggingface TORCH_HOME=$ROOT/cache/torch XDG_CACHE_HOME=$ROOT/cache HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export FFMPEG=${FFMPEG:-$ROOT/cache/bin/ffmpeg}
export PADDLE_PDX_CACHE_HOME=$ROOT/cache/paddlex

# run_py GPU PYTHON SCRIPT ARGS... : run a Python script on GPU (empty = CPU only), in the container if configured.
run_py() {
  local gpu=$1; shift
  local envs=(-e OMP_NUM_THREADS=$THREADS -e MKL_NUM_THREADS=$THREADS -e OPENBLAS_NUM_THREADS=$THREADS
              -e HF_HOME=$HF_HOME -e TORCH_HOME=$TORCH_HOME -e XDG_CACHE_HOME=$XDG_CACHE_HOME
              -e HF_HUB_OFFLINE=$HF_HUB_OFFLINE -e FFMPEG=$FFMPEG -e PADDLE_PDX_CACHE_HOME=$PADDLE_PDX_CACHE_HOME
              -e CUDA_VISIBLE_DEVICES=$gpu -e PYTHONPATH=${PYTHONPATH:-} -e ANNO_ROOT=$ROOT
              -e FILTER_POOL=${FILTER_POOL:-} -e ANNO_LOGS=${ANNO_LOGS:-}
              -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_DEEP_GEMM=0)
  if [ -n "$ANNO_CONTAINER" ]; then
    docker exec -w "$PWD" "${envs[@]}" "$ANNO_CONTAINER" "$@"
  else
    CUDA_VISIBLE_DEVICES=$gpu VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_DEEP_GEMM=0 "$@"
  fi
}

# pick_gpus N : print up to N idle GPU ids (no compute process, < 1 GB used), comma-separated
pick_gpus() {
  local n=${1:-1}
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
    while IFS=', ' read -r idx uuid mem; do
      if [ "$mem" -lt 1024 ] && ! grep -q "$uuid" <<< "$busy"; then echo "$idx"; fi
    done | head -n "$n" | paste -sd, -
}

# step NAME CMD... : run a pipeline step, append its wall time to $OUT/timing.jsonl, abort on failure
step() {
  local name=$1; shift
  local t0 t1 rc
  t0=$(date +%s%N)
  echo "[$(date +%T)] >>> $name" | tee -a "$OUT/logs/pipeline.log"
  "$@" >> "$OUT/logs/$name.log" 2>&1
  rc=$?
  t1=$(date +%s%N)
  printf '{"step": "%s", "sec": %.2f, "rc": %d}\n' "$name" "$(( (t1 - t0) / 1000000 ))e-3" "$rc" >> "$OUT/timing.jsonl"
  if [ $rc -ne 0 ]; then echo "step $name failed (rc=$rc), see $OUT/logs/$name.log" | tee -a "$OUT/logs/pipeline.log"; return $rc; fi
}
