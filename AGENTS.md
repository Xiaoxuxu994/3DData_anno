# AGENTS.md：给 AI agent（和新同事）的执行手册

读完这一页就能在这个仓库里安全地跑管线、做实验、提交改动。更细的背景见 `README.md` 和 `docs/`。

## 1. 这是什么

视频 3D 伪标签管线 **v1.3**：把 12 s 视频片段标注成每帧的内参 K、位姿 c2w、米制深度和置信度，并自动质检。

```
① 筛选 filter/  →  ② 打标 ViPE K(+门控) → Pi3X 以 K 为条件 → 尺度对齐 ViPE  →  ③ 质检 v1c + flags  →  ④ 人工抽检
```

- 生产入口：`pipeline/run_clip.sh`（一段）、`pipeline/run_batch.sh`（一批）。
- 当前最佳配置与参考指标：`configs/pipeline_v13.yaml`。
- 待办（按优先级）：`docs/KNOWN_ISSUES.md`。

## 2. 运行环境（事实）

| 项 | 值 |
|---|---|
| 根目录 | `ROOT=${ANNO_ROOT:-/data/home/z00533370}`，布局 `ROOT/{code,cache,data,outputs}`；**本仓库就是 `ROOT/code`** |
| 第三方代码 | `ROOT/code/{vipe,Pi3,depth-anything-3,...}`，被 gitignore；版本见 `third_party/repos.lock`，缺失时 `bash third_party/clone.sh` |
| Python 环境 | 每一步用固定的环境，定义在 `pipeline/common.sh`（`PY_VIPE`、`PY_PI3`、`PY_DA3`、`PY_FILTER`、`PY_OCR`、`PY_VLLM`、`PY_QC`）；包列表在 `envs/` |
| 容器 | GPU 步骤默认在容器 `z00533370_dev`（镜像 `z00533370/vipe:cu128`）里通过 `docker exec` 跑，`ROOT` 以相同路径挂载；`ANNO_CONTAINER=` 设为空则在宿主机跑。重建容器：`docker/run_dev.sh`，镜像：`docker/Dockerfile.vipe` |
| GPU | 8 × H200，**多人共享**；`pick_gpus N` 自动挑空闲卡（无进程且显存 < 1 GB） |
| 网络 | 慢：PyPI 用阿里云镜像，HF 慢、ModelScope 快；运行时 `HF_HUB_OFFLINE=1`，权重全部已在 `ROOT/cache` |
| 模型显存 | Pi3X 约 82 GB、Qwen3-VL-32B 约 63 GB，一张卡同时只放一个大模型 |

## 3. 硬性规则（共享服务器，必须遵守）

1. **只用空闲 GPU**：先 `nvidia-smi`，用 `--gpus=` 或 `CUDA_VISIBLE_DEVICES` 指定；同一任务最多占 2 张卡，除非用户明确同意。
2. **限制 CPU 线程**：`ANNO_THREADS`（默认 4）；并行进程数 × 线程数不超过约 32。曾经因为没限线程把负载打到 540。
3. **不删除、不覆盖**已有的数据和结果（`ROOT/data`、`ROOT/outputs`、`ROOT/cache`），新结果写到新的子目录，例如 `outputs/<你的实验名>/`。
4. **不碰别人的进程和容器**；不要 `pkill -f <模式>`，按 PID 结束自己的进程。
5. **不改系统**（apt、全局 pip、docker daemon、权限）；需要时先问用户。
6. **不要修改实验脚本去适配新需求**（`evolution/`、`bench/`、`qc/`、`hybrid/` 里的脚本是已记录结果的出处）；需要改就复制一份到 `pipeline/` 或新目录再改，和 `pipeline/` 现在的做法一致。
7. **结果不进 git**：npz / ply / mp4 / 权重已在 `.gitignore`；只提交代码、配置和文档。
8. **不要写死路径**：新代码从 `ANNO_ROOT` 读根路径（shell 里 `source pipeline/common.sh`，Python 里 `os.environ.get("ANNO_ROOT", "/data/home/z00533370")`）。

## 4. 常用任务与命令

所有命令在 `ROOT` 下执行（`cd $ANNO_ROOT`）。长任务用 `nohup ... &` 后台跑，再看日志。

### 4.1 标注一段 / 一批视频

```bash
code/pipeline/run_clip.sh /path/clip.mp4 outputs/<run_name> [--gpus=4,5] [--skip-vlm] [--split-cuts]
code/pipeline/run_batch.sh outputs/<run_name> a.mp4 b.mp4 name=/path/frames_dir ... [--gpus=4,5]
```

- 输入：视频（按 15 fps 解码前 12 s）或帧目录（`000000.jpg ...`）。
- 可续跑：已完成的步骤自动跳过；失败时看 `OUT/logs/<step>.log`。
- 参数：`--skip-filter`（只打标 + 质检）、`--skip-vlm`（不起 vLLM，召回从 0.88 降到 0.79）、`--split-cuts`（切镜切分，镜头 ≥ 6 s）、`--force-annotate`（筛选拒掉的也打标）。

### 4.2 改了代码之后：回归测试（必跑）

```bash
code/tests/regression_A.sh [--gpus=4,5]          # A 组 4 段有真值片段，约 6 min，退出码 0 = PASS
```

- 当前基线：70 PASS / 2 WARN / 0 FAIL。2 个 WARN 是 KITTI 07 上 ViPE 重跑的随机性（见第 7 节），可以接受。
- **出现 FAIL 不能提交**；WARN 数量增加要在提交说明里解释。

### 4.3 做一个新实验

1. 在 `docs/experiments/experiments.csv` 新增一行，编号接着现有最大编号（下一个是 **E14**），状态 `running`。
2. 代码放 `evolution/E14_<短名>/`，结果写 `outputs/evolution/E14/`。
3. 评测：有真值用 A 组（`outputs/bench/eval3d_A/`，51 段）作主结论，B 组（ScanNet++、TartanAir、vKITTI2）只作诊断（Pi3 训练集包含 ScanNet++ 和 TartanAir），C 组（72 段野外，无真值）看质检通过率和自一致性。指标定义见 `docs/experiments/LOG.md` 和 `eval3d/eval3d_common.py`。
4. 在 `docs/experiments/LOG.md` 追加一节：动机、做法、结果表、**是否采纳及理由**。失败的实验也要记。
5. 若采纳：改 `pipeline/` 对应步骤，更新 `configs/pipeline_v13.yaml`（`version` 加一、写明 `changes`、更新参考指标），跑 4.2 的回归，必要时同步更新 `tests/regression_A.py` 的参考值。

### 4.4 可视化

```bash
OMP_NUM_THREADS=1 $ROOT/cache/qc-venv/bin/python code/viz/render_demo.py [--workers=5] [--only=id1,id2]   # → outputs/viz/videos/
OMP_NUM_THREADS=1 $ROOT/cache/qc-venv/bin/python code/viz/export_ply.py  [--workers=4] [--only=id1,id2]   # 点云 + 相机 PLY
```

- 两个脚本渲染的是脚本内置的示例片段表（读实验目录 `outputs/bench/<组>/pi3x_kgate` 等），四宫格：RGB / 深度 / 俯视轨迹 / 点云。
- 要可视化 `pipeline/` 的新结果，需要在脚本的片段表里加条目，或按 `OUT/labels/<clip>.npz` 的格式（第 5 节）改写读取部分；改写时复制一份，不要改原脚本（规则 6）。

## 5. 输入输出契约

`OUT/labels/<clip>.npz`（**标签本体**）：

| 键 | 形状 / 含义 |
|---|---|
| `c2w` | (N,4,4)，OpenCV 约定（x 右、y 下、z 前）camera-to-world，单位米 |
| `K` | (N,3,3)，对应深度图分辨率 |
| `depth` | (N,h,w)，米 |
| `conf` | (N,h,w)，Pi3X 置信度 0–1 |
| `scale`、`k_depth`、`k_traj` | 米制尺度 k = √(k_depth × k_traj) 及其两个分量 |
| `hfov`、`k_source` | 水平视场角（度）；内参来源（ViPE 或门控后的中位数） |

`OUT/qc/<clip>.json`：`v1c`（> 1 拒绝）及三个分量 `xtraj_ba`（%）、`xdepth_k`、`kdis`（度），`flags`、`reject_flags`（`jump60`、`fov_out`）、`info_flags`（`k_gate_fired`、`suspect_cut`、`focal_unobservable`、`suspect_zoom`），**`accept`**（false = 不能用于训练）。

`OUT/results.csv` 每段一行汇总；`OUT/filter/decisions.csv` 是筛选结论；`OUT/timing.json` 是各步耗时。

## 6. 代码导航：入口 → 组件

| 步骤 | 文件 | 出处 |
|---|---|---|
| 筛选各 stage | `filter/s1_cuts.py`、`s2_rules.py`、`s3_ocr.py`、`s4_flow.py`、`s5_geocalib.py`、`vlm_run.py`、`decide.py`（由 `pipeline/filter_stage.py` 调用） | `filter/README.md` |
| 切镜切分 | `pipeline/filter_select.py` | E2 |
| ViPE | `evolution/vipe_worker.py` | E6 |
| 内参门控 | `pipeline/pi3x_kgate.py`（门控 + `evolution/pi3x_cond.py` 的前向）、`pipeline/geocalib_k.py`、`bench/pi3x_bench.py`、`bench/da3_bench.py` | E1a、E1b |
| 尺度 | `evolution/e3_scale_vote.py` | E3b |
| 质检证人 DA3 | `evolution/da3_cond.py` | E1 |
| 质检 v1c 与 flags | `pipeline/v13_label_qc.py`（信号来自 `qc/qc_cross.py`，flags 来自 `evolution/e9_hard_rules.py`） | E8、E9 |

完整目录说明见 `docs/CODE_MAP.md`。

## 7. 已知的坑

- **ViPE 不可完全复现**：驾驶片段重跑轨迹可差 3–4%，以 ViPE 为证人的 v1c 跟着变（BTbk 1.94 → 3.03）；v1c 接近 1 的片段结论可能翻转。比较实验时用同一次 ViPE 结果，或多跑几次取中位数。
- **门控拦不住“合理但错”的内参**（BTbk：ViPE 33°，焦距扫描约 61°）。见 KNOWN_ISSUES 第 2 条（E14）。
- **尺度失败率 25%** 是 v1.3 最大问题（KITTI 07 偏小约 30%，人群场景 k 可到 7.5）。
- 入口每张卡只开 1 个 ViPE；成本模型里的“每卡 2 并发”和慢速类别隔帧需要手动开两个 `run_batch.sh` 分别处理不重叠的片段列表。
- vLLM 需要 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（`common.sh` 已设）；起服务约 1–1.5 min。
- 装包：`export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple`，不要同时跑多个 `uv pip install`（缓存锁超时）。大文件下载用 `filter/pdl.py`（ModelScope 分块并行）。
- YouTube：快速连续请求约 300 次后会被软限流；用 yt-dlp 命令行子进程并设超时。
- 仓库里可能残留 macOS 的 `._*` 文件，已被 gitignore，不要提交。
- 旧脚本里写死了 `/data/home/z00533370`；在别的根目录用之前跑 `python3 tools/relocate.py $ANNO_ROOT --apply`（只在本地改，不要提交这个改动）。

## 8. Git 工作流

- 分支：`main` 是可用版本；新工作开分支 `e14-focal-sweep`、`fix-...` 等，回归 PASS 后再合并。
- 提交说明：第一行写改了什么；正文写动机、验证结果（回归 PASS / WARN 数）、相关实验编号。
- 推送：服务器默认没有 GitHub 凭据。需要的话给仓库加一个有写权限的 deploy key，或者在有凭据的机器上 `git pull` 服务器仓库再推送。
- 提交前检查：`git status` 里没有结果文件和大文件；`git grep` 没有密码、token、IP。

## 9. 完成的标准

一项改动算完成，需要同时满足：

1. `tests/regression_A.sh` PASS，WARN 不多于基线。
2. 涉及标签或质检逻辑的改动，在 A 组全量（51 段）上与 `configs/pipeline_v13.yaml` 的参考指标对比过，结果写进 `docs/experiments/LOG.md`。
3. 文档同步：`configs/pipeline_v13.yaml`、`docs/KNOWN_ISSUES.md`，以及受影响的 README。
4. 没有违反第 3 节的任何一条。
