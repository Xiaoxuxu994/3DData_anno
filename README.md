# 3DData_anno：视频 3D 伪标签管线

把野外视频（12 s 片段）自动标注成 3D 基模的训练数据：每帧相机内参、位姿、米制深度和置信度，并自动质检。当前版本 **v1.3**。

```
视频 ─▶ ① 筛选 ─▶ ② 打标 ─▶ ③ 自校验 ─▶ ④ 人工抽检
       规则 + VLM    ViPE 内参 → Pi3X     跨模型核对      按 QC 分数抽看
```

| 层 | 做什么 | 关键结果 |
|---|---|---|
| ① 筛选 | 规则（切镜、亮度、文字、光流、鱼眼 / 变焦）+ Qwen3-VL-32B；切镜处切开、保留 ≥ 6 s 镜头 | 被拒片段打标失败率是通过片段的 2.3 倍 |
| ② 打标 | ViPE 估内参（带门控）→ Pi3X 以内参为条件出位姿和深度 → 尺度对齐 ViPE | A 组点云 F@5cm 0.49 → 0.81（对比 ViPE 单独） |
| ③ 自校验 | 与 ViPE 轨迹、DA3 深度、其他内参来源核对，QC = 三项相对误差最大值，> 1 拒绝；另有旗标 | 保留 75%，留下的失败率 5.8% |
| ④ 人工抽检 | 按 QC 分数排序抽看，用来校准阈值 | 进行中 |

## 快速开始

```bash
export ANNO_ROOT=/data/home/<you>                 # 布局：$ANNO_ROOT/{code,cache,data,outputs}
git clone git@github.com:Xiaoxuxu994/3DData_anno.git $ANNO_ROOT/code
cd $ANNO_ROOT/code
python3 tools/relocate.py $ANNO_ROOT --apply      # 旧脚本里的根路径换成你的（原机器上不需要）
bash third_party/clone.sh                         # ViPE、Pi3、DA3，固定 commit
# 按 envs/README.md 建环境；按 docs/WEIGHTS_DATA.md 准备权重
bash pipeline/run_clip.sh <clip.mp4> <out_dir>    # 跑一段：筛选 → 打标 → 质检
bash tests/regression_A.sh                        # 改代码后：A 组回归，对比参考指标
```

`pipeline/README.md` 有入口的参数、输出格式和耗时。

## 文档

| 文件 | 内容 |
|---|---|
| `docs/ARCHITECTURE.md` | 四层管线的细节、阈值、结果和成本 |
| `docs/CODE_MAP.md` | 每个目录做什么、哪些是生产代码、路径约定 |
| `docs/WEIGHTS_DATA.md` | 模型权重、数据集和结果的位置与来源 |
| `docs/KNOWN_ISSUES.md` | 已知问题与下一步（尺度、内参 E14、阈值校准…） |
| `docs/experiments/` | 全部实验记录：`LOG.md`（成功和失败都记）、`experiments.csv`、成本模型、各实验的结果表 |
| `configs/pipeline_v13.yaml` | 当前最佳配置与参考指标 |

## 开发约定

- 新实验编号接着 E14，写进 `docs/experiments/LOG.md` 和 `experiments.csv`，失败的也记。
- 采纳的改动：更新 `configs/pipeline_v13.yaml`（版本号、参考指标），跑 `tests/regression_A.sh`。
- 结果写到 `$ANNO_ROOT/outputs/`，不要提交到 git；新代码从 `ANNO_ROOT` 读根路径。
- 共享服务器：只用空闲 GPU，限制 CPU 线程（`OMP_NUM_THREADS` 等）。

第三方模型代码（ViPE、Pi3、Depth Anything 3 等）遵循各自的许可证，不包含在本仓库中。
