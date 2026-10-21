#!/bin/bash
# New venvs for line F; PyPI mirror (aliyun) because pypi.org runs at ~60 KB/s from this host.
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple UV_HTTP_TIMEOUT=300
C=/data/home/z00533370/cache; L=/data/home/z00533370/outputs/logs
cd $C
(uv pip install --python filter-venv/bin/python torch==2.7.0 torchvision==0.22.0 opencv-python-headless numpy pandas pillow scipy einops imageio open_clip_torch transnetv2-pytorch==1.0.5 ftfy regex > $L/filter_venv.log 2>&1; filter-venv/bin/python -c "import torch, torchvision, open_clip, cv2; print(torch.__version__, torchvision.__version__, torch.version.cuda)" >> $L/filter_venv.log 2>&1; echo FILTER_VENV_DONE >> $L/filter_venv.log) &
(uv pip install --python ocr-venv/bin/python paddlepaddle paddleocr opencv-python-headless > $L/ocr_venv.log 2>&1; ocr-venv/bin/python -c "import paddle, paddleocr; print(paddle.__version__, paddleocr.__version__)" >> $L/ocr_venv.log 2>&1; echo OCR_VENV_DONE >> $L/ocr_venv.log) &
(uv pip install --python vllm-venv/bin/python vllm==0.30.0 qwen-vl-utils pillow opencv-python-headless > $L/vllm_setup.log 2>&1; vllm-venv/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)" >> $L/vllm_setup.log 2>&1; echo VLLM_VENV_DONE >> $L/vllm_setup.log) &
(uv venv -q -p 3.12 ms-venv; uv pip install --python ms-venv/bin/python modelscope > $L/ms_venv.log 2>&1; echo MS_VENV_DONE >> $L/ms_venv.log) &
wait
