#!/bin/bash
# (Re)create the dev container. Runs as z00533370 (uid 1003) so every file it writes is owned by you.
H=/data/home/z00533370
docker run -d --name z00533370_dev --gpus all --ipc=host --network=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 -u 1003:1004 \
  -e HOME=$H -e XDG_CACHE_HOME=$H/cache -e UV_CACHE_DIR=$H/cache/uv -e UV_PYTHON_INSTALL_DIR=$H/cache/uv/python -e UV_LINK_MODE=copy \
  -e HF_HOME=$H/cache/huggingface -e TORCH_HOME=$H/cache/torch -e TRITON_CACHE_DIR=$H/cache/triton \
  -e TORCH_EXTENSIONS_DIR=$H/cache/torch_extensions -e TORCH_CUDA_ARCH_LIST=9.0 -e MAX_JOBS=32 -e OMP_NUM_THREADS=16 \
  -v $H:$H -w $H --restart unless-stopped z00533370/vipe:cu128 sleep infinity
# GPUs 4-7 are used by other people: always set CUDA_VISIBLE_DEVICES to free cards.
