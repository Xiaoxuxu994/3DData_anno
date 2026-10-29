# GLR 轨道：3DData_anno 前后端过滤 + GLRMap 打标

目标：把 GLRMap（SfM + RoMa + PDA 的高精度几何打标器）接到 3DData_anno 现有的**前端筛选**（传统规则 + VLM）和**后端质检**（跨模型证人 + 旗标）之间，得到“能放心批量跑”的流程。设计依据见仓库外的评估报告《GLRMap 方案评估与改进》；本页只写这个仓库里新增的东西、怎么跑、哪些已验证、哪些没有。

> **状态（2026-10-29）**：管线胶水代码已在沙箱里用合成数据和 pycolmap 复刻的 GLRMap 第一阶段测过（`tests/glr/`，全部通过）。**没有**在真实 GLRMap 代码和 GPU/NPU 上跑过（沙箱没有 GPU，GLRMap 仓库当时不可访问），所以 `glr_worker.sh` 里的 GLRMap 命令模板、稠密深度的文件布局都是“按文档推断”，第一次在服务器上跑时必须按 §6 的清单核对。阈值全是预注册的起始值，未校准。

## 1. 数据流

```
                      H200 主机（3DData_anno）                              GLRMap 主机（CPU/NPU，可以是同一台）
 视频 ─▶ run_batch.sh（不改）                                             
          prep → 筛选(s1 切镜/s2 亮度模糊/s3 OCR/s4 光流/s5 GeoCalib + VLM) → decide.py
          → ViPE、GeoCalib、图像版 Pi3X/DA3（K 来源 = 证人）→ Pi3X(K 门控) 标签 → v1c 质检     ← 这些就是 v1.3，全部照旧
          │
          ▼  (a) glr_prepare.py route / jobs
       路由：reject（筛选拒）| glr（静态场景 + 真视差）| v13（其余，留给前馈打标器）
       自适应抽帧（15 fps 的每 stride 帧，60–120 帧）→ OUT/glr_jobs/<clip>/{images/, job.json}
          │   rsync -aL OUT/glr_jobs  ───────────────────────────────▶  (b) glr_worker.sh JOB_DIR
          │                                                              第一阶段(COLMAP SIFT + 顺序匹配 + 全局 SfM)
          │                                                              → 出口关口 glr_stage1_gate.py（stop 就不跑 RoMa/PDA）
          │                                                              → 稠密阶段（RoMa v2 三角化 + Pi3X + PDA）
          │   rsync -a OUT/glr_jobs/*/result*  ◀───────────────────────  result*/{stage1|run, gate.json, result.json}
          ▼  (c) glr_collect.py   GLR 结果 → 标签契约（c2w、K、depth、conf、scale…）+ 跨模型证人质检 v1c + 分级 T1/T2/T3
             glr_merge.py         每段一个最终判定：GLR 标签 / 回退 v1.3 标签 / 拒绝
```

一台机器就够时：`pipeline/run_batch_glr.sh OUT a.mp4 b.mp4 ...`（`--phase=all`）。两台机器时：H200 上 `--phase=a`，把 `OUT/glr_jobs` 拷到 GLRMap 机器跑 `glr_worker.sh`（或在那边 `--phase=b`），结果拷回后 `--phase=c`。

## 2. 新增文件（现有脚本一个都没改）

| 文件 | 作用 |
|---|---|
| `pipeline/run_batch_glr.sh` | 总入口：先调用原样的 `run_batch.sh`，再路由 → 作业包 → worker → 收集 → 合并。输出写进同一个 `OUT`，不覆盖 v1.3 的结果 |
| `pipeline/glr_prepare.py` | 阶段 a：`route` 写 `filter/glr_route.csv`；`jobs` 写 `glr_jobs/<clip>/`。抽帧步长按光流统计选 |
| `pipeline/glr_worker.sh` | 阶段 b（GLRMap 主机，自包含，只依赖 `glr_stage1_gate.py`）：跑 GLRMap + 出口关口，永不覆盖（`result`, `result_1`…） |
| `pipeline/glr_stage1_gate.py` | 第一阶段出口关口：读稀疏模型 + `database.db`，给出 `ok / suspect / stop` 和全部信号；CLI 退出码 0/1/3 |
| `pipeline/glr_collect.py` | 阶段 c：GLRMap 结果 → `labels_glr/<clip>.npz`、`qc_glr/<clip>.json`、`results_glr.csv` |
| `pipeline/glr_merge.py` | 最终判定与回退策略 → `final/{results.csv, labels/, summary.json}` |
| `pipeline/glr_common.py` | 配置、Umeyama、`traj_pct` / `depth_cmp`（`qc/qc_cross.py` 的拷贝）、E3b 尺度规则、网格工具 |
| `configs/pipeline_glr_v0.yaml` | 全部阈值（预注册起始值） |
| `tools/glr_gate_replay.py` | Phase 0.2：在已有的第一阶段输出上离线回放关口信号，对标签算捕获率 / 误拦率，判预注册通过线 |
| `tools/filter_stage_cpu.py` | 在没有 GPU 的机器上原样运行 `filter/s*.py`（冒烟测试用，不是生产） |
| `tools/stage1_sfm.py` | 用 pycolmap 复刻 GLRMap 第一阶段，带消融开关（相机模型、初始焦距、固定内参、种子）；不是 GLRMap 代码 |
| `tests/glr/` | 单元测试（`python -m unittest discover -s tests/glr`，需要 numpy、scipy、opencv、pycolmap） |

## 3. 标签与质检契约

- `labels_glr/<clip>.npz`：与 v1.3 同名键 `c2w`（N,4,4，OpenCV，米；第一台注册相机为世界原点）、`K`（N,3,3，在深度网格上）、`depth`（N,h,w，米）、`conf`、`scale`、`k_depth`、`k_traj`、`hfov`、`k_source`、`frame_hw`；新增 `frame_idx`（每个标签帧在 15 fps 片段里的序号，**只有被抽到并注册成功的帧**）、`n_orig`、`annotator='glr'`、`tier`、`depth_source`（`pda_fine | pda_coarse | none`）。没有稠密深度时 `depth` 是 (N,1,1) 的 NaN，`K` 在原图分辨率上。
- 米制尺度：GLRMap 是 SfM 任意尺度，按 E3b 规则对齐 ViPE：`k = sqrt(k_traj · k_depth)`；GLRMap 没有深度时只用轨迹尺度（不和中性 1.0 取几何平均）。
- `qc_glr/<clip>.json`：v1c 三项（`xtraj_ba` 对 ViPE、`xdepth_k` 对“以 ViPE 相机为条件的 DA3”、`kdis` 对 GeoCalib/Pi3X/DA3 的 HFOV 中位数），阈值沿用 3.27 / 0.194 / 13.2°，加 E9 旗标 `jump60`、`fov_out`；另有 `tier`、`reject_reasons`、第一阶段关口详情、与 v1.3 标签的一致性 `agreement_v13`（信息项）。
- 分级：**T3** 拒绝（v1c > 1 或不可检、`jump60`/`fov_out`、第一阶段关口 stop、GLRMap 失败）；**T1** 通过且三项都 ≤ 阈值的一半、第一阶段无 suspect、有稠密深度、注册帧 ≥ 90%；**T2** 其余通过的。**分级只说明“没有已知的失败特征被触发”，不等于精确**：证人（ViPE、Pi3X ≈ 0.65% ATE）比 GLRMap 好时的误差（0.05–0.2%）噪声大得多，这套关口只能抓粗错。
- 合并策略（`glr_merge.py`）：筛选拒 → 拒绝；路由 glr 且 T1/T2 → GLR 标签；路由 glr 但 T3/失败/被关口拦下 → 回退 v1.3 标签（v1c 通过才算）；路由 v13 → v1.3 标签。两种打标器失败机制不同（SfM 要视差和可观测的焦距；前馈要合理的 K），所以回退不是循环论证，但“二者互补”目前只是假设。

## 4. 遵守 AGENTS.md 第 3 节的地方

只用空闲 GPU：新增代码本身不占 GPU（GLRMap 的 CUDA/NPU 部分由 GLRMap 自己的命令决定，用 GLRMap 主机上的卡调度）。线程：`ANNO_THREADS` 默认 4，`glr_worker.sh` 导出 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`；**GLOMAP 自己会开远多于设置的线程**（评估会话里 3 个窗口并发出现过 > 250 线程、负载到 200），共享主机上务必设 `GLR_TASKSET=<cpu 列表>`（`taskset -c`）硬限核数。不覆盖：worker 永远写新目录；`glr_prepare.py` 对已有作业包只跳过；`run_batch_glr.sh` 的 b 阶段跳过已有 `RESULT_DIR` 的作业。路径：新代码读 `ANNO_ROOT`（`glr_common.ROOT`），无写死路径；`PY_GLR` 指 GLRMap 环境的 python。没有 `pkill`。

## 5. 已验证 / 未验证

（沙箱测试结果表在 §7，跑完后填入。）

**已验证（沙箱，合成数据 + 真实 COLMAP 模型读写）**：路由与抽帧规则；第一阶段关口规则（评估会话里的 KITTI / 无人机 / 室内 / 东京 / Ahlbeck 的数值作回归夹具）；模型和数据库读取（`load_cameras` 与输入相机逐元素一致）；E3b 米制尺度恢复；v1c 三项、`jump60`、`fov_out`、第一阶段 stop/suspect 的分级逻辑；worker 的 split / mono / dry-run / 不覆盖 / 退出码；配置文件与默认值一致。

**未验证**：真实 GLRMap 命令行与输出布局（`result.json` 的 `found` 字段会列出实际找到的文件，用来核对）；PDA 融合深度的文件命名和键；ViPE / DA3 证人在 GLR 标签上的阈值是否合适（沿用 v1.3 的样本内阈值）；路由阈值；`conf` 目前只有“深度是否有限”，没有 PDA 距离置信（评估报告的 Phase 1.4）；GLR 与 v1.3 的失败是否真的互补；回归测试 `tests/regression_A.sh`（没有改任何现有脚本，但按 AGENTS.md 第 9 节仍应在服务器上跑一次）。

## 6. 第一次在服务器上跑：核对清单

1. GLRMap 主机：`export GLRMAP_HOME=<GLRMap 检出目录> PY_GLR=<GLRMap 环境的 python> ANNO_THREADS=8 GLR_TASKSET=<8 个核>`；`bash pipeline/glr_worker.sh JOB --dry-run` 看命令。
2. **mono 模式**（默认）直接跑 GLRMap 自带的 `data_generate_demo/run_all.sh`（它读 `IMAGES`、`OUTPUT`、可选 `MASK_DIR`）。如果你的副本带 NPU 补丁里的 camera_sanity，第一阶段被拦时 `run_all.sh` 自己退出 3，worker 记 `native_gate_stop`。
3. **split 模式**：把 `data_generate_demo/run_roma_pcd.sh` 和 `run_pda_depth.sh` 里 `glomap` 之后的命令抄成 `GLR_DENSE_CMD`（占位符 `{images} {out} {s1} {threads} {py}`），第一阶段用 `GLR_STAGE1_CMD`（默认 `python -m reconstruction_flow.glomap --images {images} --output {s1} --camera-model PINHOLE`）。这样关口 stop 时省下 RoMa/PDA 的小时数。
4. 跑完看 `result*/result.json` 的 `found`：`sparse_models`、`databases`、`pda_fused`（`**/fused/*.npz`，键 `fine`/`coarse`）、`pi3x_prior`。如果 `pda_fused` 为空而你知道深度在别处，改 `glr_collect.read_depth_maps` 的 glob，并在这里记下真实布局。
5. H200 上 `--phase=c`；先看 `qc_glr/*.json` 的 `glr.depth_source` 和 `info_flags`。
6. 在 A 组的一两段（有真值）上用 `--skip-filter --glr-all` 跑，对 `outputs/bench/eval3d_A` 评测 GLR 标签，数字写进 `docs/experiments/LOG.md`，再谈采纳。

## 7. 沙箱测试结果

（待填：OOD 片段集、筛选结果、第一阶段复刻与校验、关口信号表。）
