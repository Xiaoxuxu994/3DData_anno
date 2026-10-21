# 模型权重与数据

权重和数据都不在 git 里。原服务器上的位置如下；换机器时按“来源”下载，或从原目录拷贝。

## 模型权重（`$ANNO_ROOT/cache/`）

| 模型 | 用途 | 来源 | 原位置 |
|---|---|---|---|
| ViPE（含其依赖：UniDepth、Metric3D、SAM2、GroundingDINO 等） | ② K + 尺度 | ViPE 仓库首次运行自动下载 | `cache/huggingface`、`cache/torch/hub/checkpoints` |
| Pi3X | ② 主打标器 | HF `yyfz233/Pi3X` | `cache/huggingface/hub/models--yyfz233--Pi3X` |
| DA3（Nested Giant-Large 1.1、Giant） | ③ 证人、② 门控 | HF `depth-anything/DA3NESTED-GIANT-LARGE-1.1` 等 | `cache/huggingface/hub/` |
| GeoCalib | ① 鱼眼 / 变焦、② 门控 | pip 包 `geocalib`，首次运行自动下载 | `cache/torch` |
| RAFT large | ① 运动规则 | torchvision | `cache/torch/hub/checkpoints/raft_large_*.pth` |
| TransNetV2 | ① 切镜 | pip 包 `transnetv2-pytorch` | filter-venv 内 |
| PP-OCRv5 mobile det | ① 文字 | HF `PaddlePaddle/PP-OCRv5_mobile_det` | `cache/huggingface` |
| Qwen3-VL-32B-Instruct | ① VLM | ModelScope `Qwen/Qwen3-VL-32B-Instruct` | `cache/modelscope/Qwen/` |
| VGGT-Omega 416（复现版） | 基准对比 | ModelScope | `cache/modelscope/VGGT-Omega/` |
| MoGe-2 / MoGe-3、MapAnything、Voyager、PURE | 对比实验 | HF / ModelScope | `cache/huggingface`、`cache/modelscope` |

服务器访问 HF 和 PyPI 较慢：PyPI 用阿里云镜像；大文件优先走 ModelScope（下载脚本 `filter/pdl.py`、`filter/dl_vlms_*.sh`）。

## 数据（`$ANNO_ROOT/data/`）

| 目录 | 内容 | 规模 |
|---|---|---|
| `eval3d/A_rank/` | A 组：有真值、不在 Pi3 训练集里，用于排名：7-Scenes、TUM、Bonn、Sintel、KITTI（ETH3D 已移出） | 51 段 |
| `eval3d/B_diag/` | B 组：有真值，用于诊断：ScanNet++、TartanAir、vKITTI2（ScanNet++ / TartanAir 在 Pi3 训练集里，结论只作参考） | 48 段 |
| `eval3d/C_wild/` | C 组：野外视频，无真值，按类别（室内静态 / 动态、户外静态 / 动态、驾驶、具身、难例） | 72 段 |
| `filter_pool/` | ① 筛选的片段池（YouTube CC-BY，436 个视频），每段的帧、分数与 `decisions.csv` | 1835 段，81 GB |
| `eval3d_raw/`、`scannetpp_sample30/`、`tartanair_v2/` 等 | 原始数据集（转换脚本在 `eval3d/`） | |

## 结果（`$ANNO_ROOT/outputs/`）

| 目录 | 内容 |
|---|---|
| `outputs/evolution/` | 迭代实验 E1–E13 的原始结果；`LOG.md`、`experiments.csv`、`pipeline_best.yaml` 的副本已放在本仓库 `docs/experiments/`、`configs/` |
| `outputs/bench/` | 各方法在 A/B/C 组上的预测与评测 |
| `outputs/filter/` | 筛选报告与拼图 |
| `outputs/qc/` | v0 质检信号与校准 |
| `outputs/viz/` | 演示视频与 PLY |
| `outputs/pipeline_test/` | `pipeline/` 入口和回归测试的输出 |

**回归测试依赖**：`tests/regression_A.py`、`tests/compare_C.py` 读取 `outputs/evolution/`（E3、E3b、E8、E9 的逐段结果 CSV）和 `outputs/bench/eval3d_A/`（GT 与 ViPE 结果）作为参考值；换机器时这两个目录要一起拷贝。
