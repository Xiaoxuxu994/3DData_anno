# Python 环境

每个 `*.txt` 是对应环境的完整包列表（`name==version`，从 site-packages 导出，2026-10-21）。所有环境都是 Python 3.12，用 [uv](https://github.com/astral-sh/uv) 创建。

| 环境（原路径） | 文件 | 用在哪一层 | v1.3 必需 |
|---|---|---|---|
| `code/vipe/.venv` | `vipe-dotvenv.txt` | ② 内参 K + 米制尺度（ViPE，torch 2.9 cu128） | 是 |
| `code/Pi3/.venv` | `Pi3-dotvenv.txt` | ② 主打标器 Pi3X（以 K 为条件） | 是 |
| `code/depth-anything-3/.venv` | `depth-anything-3-dotvenv.txt` | ③ 质检证人 DA3；② 内参门控备选 | 是 |
| `cache/filter-venv` | `filter-venv.txt` | ① 规则过滤：TransNetV2、RAFT、GeoCalib | 是 |
| `cache/ocr-venv` | `ocr-venv.txt` | ① 字幕 / 文字检测（PaddleOCR） | 是 |
| `cache/vllm-venv` | `vllm-venv.txt` | ① VLM 过滤（Qwen3-VL-32B，vLLM 0.30） | 是 |
| `cache/qc-venv` | `qc-venv.txt` | ③ 质检信号、hybrid 合成 | 是 |
| `cache/eval-venv` | `eval-venv.txt` | 评测与报表（A/B/C 组指标） | 评测用 |
| `cache/yt-venv` | `yt-venv.txt` | YouTube 检索与下载（yt-dlp） | 采集用 |
| `cache/ms-venv`、`dl-venv`、`dl3d-venv` | 同名 | ModelScope / 数据集下载 | 下载用 |
| `cache/moge-venv`、`ma-venv`、`droid-venv`，`code/vggt-omega/.venv`、`code/SpatialVID/.venv` | 同名 | 对比实验（E10、E12、benchmarks） | 否 |

ViPE 也可以在容器里跑：镜像 `z00533370/vipe:cu128`（容器 `z00533370_dev`，把 `/data/home/z00533370` 原路径挂进去）。`code/hybrid` 的部分脚本通过 `docker exec` 调用它。

## 重建一个环境

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple   # 国内镜像；pypi.org 在服务器上很慢
uv venv -p 3.12 $ANNO_ROOT/cache/qc-venv
grep -v '^#' envs/qc-venv.txt | sed 's/+cu12[0-9]//' > /tmp/req.txt
uv pip install --python $ANNO_ROOT/cache/qc-venv/bin/python -r /tmp/req.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

- 带 `+cu126` / `+cu128` 的 torch 需要对应的 PyTorch index（vipe 用 cu128）。
- `pi3`、`vipe`、`depth_anything_3` 这类是对应仓库的 editable 安装：先 `bash third_party/clone.sh`，再在仓库里 `uv pip install -e .`。
- 原始的创建命令见 `filter/setup_venvs.sh`。
