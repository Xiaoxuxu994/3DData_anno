# GLRMap Phase 0：在服务器上执行的实验（沙箱跑不了的部分）

背景：评估报告《GLRMap 方案评估与改进》的 Phase 0 是“先做一周几乎不占算力的离线实验，验证关口能否成立，再决定是否投入工程”。沙箱（4 核 CPU，无 GPU，GLRMap 仓库不可访问）只能做其中依赖公开数据和 CPU 的部分（结果见 `docs/experiments/LOG.md` E15）；下面这些必须在有数据的机器上跑：server13（昇腾，NPU 移植会话的 `npu_port/` 目录，下文记作 `$NPU_PORT`）、server110（纯 CPU，评测会话的 `glr_eval/` 目录）、H200 主机（3DData_anno）。每一步都写了命令、耗时、**通过线 / 放弃线**。通过线在看结果之前就固定，不要边看边改；阈值调整只能用留一窗口交叉验证。

## 0.2 关口离线回放（server13，纯 CPU，约半天）

数据：已经落盘的第一阶段输出（`runs/robust-stage1/gate_power` 的 86 个模型、`runs/kitti_multi` 的 40 次尝试、`runs/ood` 的 11 段、server110 的 6 段 OOD + 4 个 KITTI）。

1. 做标签 `labels.csv`（列 `run,label`，`run` = 相对扫描根目录的第一阶段目录路径）：
   - `accurate`：该次第一阶段的 ATE < 1 m 且 ≥ 5000 个三维点（`kitti-base` 报告里的成功规则；KITTI 窗口）。
   - `silent_wrong`：第一阶段没被任何现有检查拦住，但 ATE ≥ 1 m 或焦距误差 > 30%（评估里的 9 次“静默走偏”）。
   - `degenerate`：被 `camera_sanity` 拦下或模型退化（点数 < 50/帧、大部分图像对平面/全景）。
   - OOD 无真值的片段不进标签，只看信号表。
2. 回放：
   ```bash
   cd code && export ANNO_ROOT=...        # 只用来定位 pipeline/ 里的模块，不读数据
   $PY_GLR tools/glr_gate_replay.py $NPU_PORT/runs/kitti_multi \
       $NPU_PORT/runs/robust-stage1/gate_power \
       --labels labels.csv --out outputs/evolution/E15/gate_signals.csv
   ```
   `PY_GLR` = GLRMap 环境的 python（numpy + pycolmap）。工具只读数据库和模型，不改任何东西。
3. K 证人信号（可选）：对每个窗口算 Pi3X(仅图像) / DA3 / GeoCalib 的 HFOV 中位数，写 `witness.csv`（`run,hfov_deg`），加 `--witness witness.csv`，看 `k_witness_disagree` 的捕获率。
4. **通过线**：9 次静默走偏至少抓到 7 次；全部非精确（静默走偏 + 退化）的 ≥ 90% 被标 stop 或 suspect；26 个精确的运行里被误 **stop** 的 ≤ 2 个（现有 camera_sanity 误拦 2 个）。**放弃线**：达不到则结论是“GLRMap 不上网络视频，除非内参策略改变”，不要去调阈值凑通过。
5. 留一窗口：若要调阈值，按窗口分组（工具输出末行提示的分组数）做留一交叉验证，在报告里写明哪些数字是样本内的。

## 0.2b 第一阶段消融（server110 或任一 CPU 主机，约 3–6 小时）

目的：焦距为什么不稳（评估：自由焦距 40 次尝试 24 次精确、9 次静默走偏），换哪种相机设定更稳。`tools/stage1_sfm.py` 是 pycolmap 的复刻（沙箱里在 Ahlbeck 上和真实 GLRMap 第一阶段的点数、重投影、fx/fy、位姿离群位置都对得上，见 LOG E15），不依赖 GLRMap 代码。

```bash
S=tools/stage1_sfm.py; F=<该窗口的帧目录，按 GLRMap 的抽帧规则>        # KITTI 用 stride 2，高速公路 stride 1
$PY_GLR $S --images $F --out $OUT/w03/v0a --seed 1                      # GLRMap 默认：PINHOLE，焦距初值 1.2 x 宽，主点固定
$PY_GLR $S --reuse-db $OUT/w03/v0a --seed 2 --out $OUT/w03/v0b          # 重复（稳定性）
$PY_GLR $S --reuse-db $OUT/w03/v0a --seed 3 --out $OUT/w03/v0c
$PY_GLR $S --reuse-db $OUT/w03/v0a --db-camera SIMPLE_PINHOLE --out $OUT/w03/v1sp      # fx = fy
$PY_GLR $S --reuse-db $OUT/w03/v0a --focal-w <GeoCalib/Pi3X/DA3 的 f/W 中位数> --out $OUT/w03/v2prior   # 好的初值，仍优化焦距
$PY_GLR $S --reuse-db $OUT/w03/v0a --focal-w <同上> --fixed-focal 1 --out $OUT/w03/v3fixed        # 固定焦距
```
- 11 个窗口 × 以上 6 种 ≈ 66 次；匹配只做一遍（`--reuse-db`），后面每次只是全局建图（分钟级）。**每次用全新输出目录**（脚本拒绝写非空目录）。
- 指标：对真值的 ATE（用你们的 `kitti_metrics.py`）、焦距误差、`glr_gate_replay.py` 给出的信号；成功率定义同 0.2。
- **通过线**：无标定时，v1sp 或 v2prior 的精确率 ≥ 85%（现状 24/40 = 60%），且静默走偏 ≤ 2/40；有标定时固定 K 的精确率 ≥ 95%（现状 17/21）。**放弃线**：三种设定都不比默认好，则第一阶段需要换（前馈初始化 SfM，Phase 2），不再在 COLMAP 参数上花时间。
- 注意：匹配缓存、输出目录必须新建；GLOMAP 线程不受 `--threads` 限制，共享主机上用 `taskset -c` 限核。

## 0.3 前端筛选回放（H200 主机，GPU，约 1 小时）

把评估用过的网络视频（对话 2 的 11 条 + 对话 3 的 6 条）和 KITTI 8 个窗口作为输入跑 3DData_anno 筛选，不标注：

```bash
code/pipeline/run_batch.sh outputs/evolution/E15/filter_replay <视频…> --gpus=<空闲卡> --split-cuts   # 带 VLM；加 --skip-vlm 看规则单独的效果
```
**通过线**：对话 3 的两个硬失败（无人机、室内横移）全部拒掉；对话 2 的“固定 K 后仍退化”的 6 条至少拒掉 4 条；“固定 K 后可用”的 5 条至多误拒 1 条；KITTI 8 窗口 0 误拒。**放弃线**：筛选挡不住硬失败，则 GLRMap 上线前必须再加视差预检（`glr_prepare.py` 的路由就是这个入口）。

## 1. 真实 GLRMap 上的第一次联调（GLRMap 主机 + H200，约 2 小时）

按 `docs/GLR_INTEGRATION.md` §6 的清单走一遍：先 `glr_worker.sh JOB --dry-run`，再对 A 组的一两段（有真值）跑 `run_batch_glr.sh ... --skip-filter --glr-all`，核对 `result.json` 的 `found`、`qc_glr/*.json` 的 `glr.depth_source`；用 `outputs/bench/eval3d_A` 评测 `labels_glr/` 的标签，数字写进 LOG。GLR 轨道在 A 组上的结果（ATE Sim3 %、F@5cm、AbsRel、尺度失败率）没有和 v1.3 的参考指标（`configs/pipeline_v13.yaml`）对上之前，不要把它放进批量生产。

## 2. 不要做的事

- 不要在 KITTI-06 窗口上调阈值再在同一批窗口上宣称通过（这些数字已经被看过）。
- 不要把重投影误差、留出误差、互渲一致性当作唯一关口（评估里它们对真实误差几乎没有预测力）。
- 不要用重试救坏窗口（w03 稳定地错）；重试只用来**检测**不稳定。
