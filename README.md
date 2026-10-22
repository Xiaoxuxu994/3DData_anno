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

- 一批片段：`bash pipeline/run_batch.sh <out_dir> a.mp4 b.mp4 ... [--gpus=4,5] [--skip-filter] [--skip-vlm] [--split-cuts]`，模型只加载一次。
- 参数、输出格式（标签 npz、`qc.json`、`timing.json`）、每步用哪个环境：见 `pipeline/README.md`。

### 实测（2026-10-21，H200）

| 测试 | 结果 |
|---|---|
| TUM fr1_room 单段 | 与原实验逐项一致（尺度 1.0904 / 1.0902，QC 0.326 / 0.326），2.1 min |
| C 组台北室内，完整流程含 VLM | 筛选结论、尺度、QC 与原实验一致，5.2 min |
| `tests/regression_A.sh`（TUM、7-Scenes、KITTI 07、Bonn） | **PASS**：70 PASS / 2 WARN / 0 FAIL，6.3 min |

2 项 WARN 来自 ViPE 重跑的随机性（KITTI 07 的 QC 4.28 / 4.43，标签本身一致，两次都拒），见 `docs/KNOWN_ISSUES.md` 第 7 条。

## 文档

| 文件 | 内容 |
|---|---|
| `AGENTS.md` | **执行手册**（给 AI agent 和新同事）：环境、硬性规则、命令、输入输出、完成标准 |
| `docs/ARCHITECTURE.md` | 四层管线的细节、阈值、结果和成本 |
| `docs/CODE_MAP.md` | 每个目录做什么、哪些是生产代码、路径约定 |
| `docs/WEIGHTS_DATA.md` | 模型权重、数据集和结果的位置与来源 |
| `docs/KNOWN_ISSUES.md` | 已知问题与下一步（尺度、内参 E14、阈值校准…） |
| `docs/experiments/` | 全部实验记录：`LOG.md`（成功和失败都记）、`experiments.csv`、成本模型、各实验的结果表 |
| `configs/pipeline_v13.yaml` | 当前最佳配置与参考指标 |

## 交接须知

- **权限**：仓库是私有的，在 GitHub 的 Settings → Collaborators 添加协作者。
- **不在 git 里的东西**：模型权重、数据、结果都在 `$ANNO_ROOT/{cache,data,outputs}`，清单见 `docs/WEIGHTS_DATA.md`。回归测试还需要 `outputs/evolution/`（E3、E3b、E8、E9 逐段结果）和 `outputs/bench/eval3d_A/`（GT 与 ViPE 结果）作为参考值，换机器时一起拷贝。
- **路径**：旧脚本写死了 `/data/home/z00533370`，clone 到别处后先跑 `tools/relocate.py`。
- **运行位置**：默认在容器 `z00533370_dev`（镜像 `z00533370/vipe:cu128`）里跑，`ANNO_CONTAINER=` 设为空则在宿主机跑；GPU 自动挑空闲卡，每进程限 4 线程（`pipeline/common.sh`）。
- **已知限制**：ViPE 重跑不完全可复现，QC 接近 1 的片段结论可能翻转；入口每卡只开 1 个 ViPE，成本模型里的“每卡 2 并发”和慢速类别隔帧尚未自动化。完整列表见 `docs/KNOWN_ISSUES.md`。
- **推送**：原服务器没有 GitHub 凭据；要在服务器上直接 push，给仓库加一个 deploy key（Settings → Deploy keys，勾选写权限）。

## 开发约定

- 新实验编号接着 E14，写进 `docs/experiments/LOG.md` 和 `experiments.csv`，失败的也记。
- 采纳的改动：更新 `configs/pipeline_v13.yaml`（版本号、参考指标），跑 `tests/regression_A.sh`。
- 结果写到 `$ANNO_ROOT/outputs/`，不要提交到 git；新代码从 `ANNO_ROOT` 读根路径。
- 共享服务器：只用空闲 GPU，限制 CPU 线程（`OMP_NUM_THREADS` 等）。

第三方模型代码（ViPE、Pi3、Depth Anything 3 等）遵循各自的许可证，不包含在本仓库中。
