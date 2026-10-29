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
          │   rsync -a OUT/glr_jobs/  ◀────────────────────────────────  result*/{stage1|run, gate.json, result.json} + RESULT_DIR
          │   （拷整个 glr_jobs 目录：RESULT_DIR 指针最后写，是“作业已完成”的标志）
          ▼  (c) glr_collect.py   GLR 结果 → 标签契约（c2w、K、depth、conf、scale…）+ 跨模型证人质检 v1c + 分级 T1/T2/T3
             glr_merge.py         每段一个最终判定：GLR 标签 / 回退 v1.3 标签 / 拒绝
```

一台机器就够时：`pipeline/run_batch_glr.sh OUT a.mp4 b.mp4 ...`（`--phase=all`）。两台机器时：H200 上 `--phase=a`，把 `OUT/glr_jobs` 拷到 GLRMap 机器跑 `glr_worker.sh`（或在那边 `--phase=b`，同一个 OUT 路径），整个 `glr_jobs` 拷回后 `--phase=c`（`--phase=c` / `--phase=a` 不需要再给输入视频）。`--phase=b` 按 `RESULT_DIR`+`result.json` 判断作业是否完成：被打断或没写完的作业会重跑（新目录），失败的作业默认不自动重跑（`--glr-retry` 才重跑）。

## 2. 新增文件（现有脚本一个都没改）

| 文件 | 作用 |
|---|---|
| `pipeline/run_batch_glr.sh` | 总入口：先调用原样的 `run_batch.sh`，再路由 → 作业包 → worker → 收集 → 合并。输出写进同一个 `OUT`，不覆盖 v1.3 的结果 |
| `pipeline/glr_prepare.py` | 阶段 a：`route` 写 `filter/glr_route.csv`；`jobs` 写 `glr_jobs/<clip>/`。抽帧步长按光流统计选 |
| `pipeline/glr_worker.sh` | 阶段 b（GLRMap 主机，自包含，只依赖 `glr_stage1_gate.py`）：跑 GLRMap + 出口关口，永不覆盖（`result`, `result_1`…）。命令模板整体在 `taskset`/`timeout` 下用 `bash -c` 执行，占位符走环境变量（片段名里的空格、引号、`;`、`&` 只是数据）；收到 INT/TERM 会停掉整组 GLRMap 进程并记 `interrupted`；`result.json` 最先写、`RESULT_DIR` 最后写 |
| `pipeline/glr_stage1_gate.py` | 第一阶段出口关口：读稀疏模型 + `database.db`，给出 `ok / suspect / stop` 和全部信号；CLI 退出码 0/1/3，**4 = 关口自己失败**（verdict `error`：pycolmap 缺失、数据库打不开、不支持的相机模型…），任何非 0/1/3 的退出码或没有 JSON 裁决都按失败处理，**绝不当作 ok**。焦距规则按长边 max(W,H) 算（COLMAP 默认焦距是 1.2×长边），竖屏片段与横屏一致 |
| `pipeline/glr_collect.py` | 阶段 c：GLRMap 结果 → `labels_glr/<clip>.npz`、`qc_glr/<clip>.json`、`results_glr.csv` |
| `pipeline/glr_merge.py` | 最终判定与回退策略 → `final/{results.csv, labels/, summary.json}` |
| `pipeline/glr_common.py` | 配置、Umeyama、`traj_pct` / `depth_cmp`（`qc/qc_cross.py` 的拷贝）、E3b 尺度规则、网格工具 |
| `configs/pipeline_glr_v0.yaml` | 全部阈值（预注册起始值） |
| `tools/glr_gate_replay.py` | Phase 0.2：在已有的第一阶段输出上离线回放关口信号，对标签算捕获率 / 误拦率，判预注册通过线 |
| `tools/filter_stage_cpu.py` | 在没有 GPU 的机器上原样运行 `filter/s*.py`（冒烟测试用，不是生产） |
| `tools/stage1_sfm.py` | 用 pycolmap 复刻 GLRMap 第一阶段，带消融开关（相机模型、初始焦距、固定内参、种子）；不是 GLRMap 代码 |
| `tests/glr/` | 单元测试（`python -m unittest discover -s tests/glr`，需要 numpy、scipy、opencv、pycolmap；`test_fidelity.py` 另需 pandas）。包括：把 `qc_cross`/`eval_vipe`/`eval3d_bench`/`e3_scale_vote`/`e9_hard_rules`/`v13_label_qc` 里被复制的函数用 `ast` 原样抽出来和拷贝逐项对拍 |

## 3. 标签与质检契约

- `labels_glr/<clip>.npz`：与 v1.3 同名键 `c2w`（N,4,4，OpenCV，米；第一台注册相机为世界原点）、`K`（N,3,3，在深度网格上）、`depth`（N,h,w，米）、`conf`、`scale`、`k_depth`、`k_traj`、`hfov`、`k_source`、`frame_hw`；新增 `frame_idx`（每个标签帧在 15 fps 片段里的序号，**只有被抽到并注册成功的帧**）、`n_orig`、`annotator='glr'`、`tier`、`depth_source`（`pda_fine | pda_coarse | none`）。没有稠密深度的结果违反标签契约，默认**拒绝**（`qc.allow_no_depth: false`，原因 `depth_missing`）；这种被拒标签里的 `depth` 是 (N,1,1) 的 NaN、`K` 在原图分辨率上，只用于排查。
- 米制尺度：GLRMap 是 SfM 任意尺度，按 E3b 规则对齐 ViPE：`k = sqrt(k_traj · k_depth)`（有证据时与 `evolution/e3_scale_vote.py` 逐位一致，见 `test_fidelity.py`）。与 E3b 的差别只在 GLRMap 的标签可能缺深度：深度比从“实际有 GLRMap 深度的帧”里取 10 帧；没有深度比时 `k_depth = k_traj`（不和中性 1.0 取几何平均）；两种票都没有时 **`k = NaN`，标签以 `scale_unverified` 拒绝**（E3b 的中性 1.0 对 Pi3X 合理，对 SfM 任意尺度等于把无单位数当米导出）。
- `qc_glr/<clip>.json`：v1c 三项（`xtraj_ba` 对 ViPE、`xdepth_k` 对“以 ViPE 相机为条件的 DA3”、`kdis` 对 GeoCalib/Pi3X/DA3 的 HFOV 中位数），阈值沿用 3.27 / 0.194 / 13.2°，加 E9 旗标 `jump60`、`fov_out`；另有 `tier`、`reject_reasons`、第一阶段关口详情、与 v1.3 标签的一致性 `agreement_v13`（信息项）。`jump20/jump60` 是在**相邻标签帧**（间隔 `stride` 个源帧）之间判的：stride 4 时平滑摇镜超过 225°/s 就会触发 `jump60`（v1.3 是 15 fps 相邻帧 900°/s）——有意不稀释，因为采样帧之间的位姿跳变不该被步长摊薄，而且路由对快镜头会选小步长。
- 分级：**T3** 拒绝（v1c > 1；任一 v1c 分量或尺度算不出来；`jump60`/`fov_out`；第一阶段关口 stop / error / 缺失；没有 GLRMap 深度；GLRMap 失败、被打断或还没跑完）；**T1** 通过且三项**都有限**且 ≤ 阈值的一半、第一阶段关口 ok、深度覆盖 ≥ 90% 的标签帧、注册帧 ≥ 90%；**T2** 其余通过的（深度覆盖不足记 `depth_partial`）。GLR 轨道对“没检查到”比 v1.3 严格（`qc.require_gate / require_all_parts / allow_no_depth / min_depth_cov`）：GLRMap 的失败是静默的、主要是 K，`kdis` 算不出来（没有 K 证人）时标签被拒，而不是按 v1.3 那样只取剩下的分量。**分级只说明“没有已知的失败特征被触发”，不等于精确**：证人（ViPE、Pi3X ≈ 0.65% ATE）比 GLRMap 好时的误差（0.05–0.2%）噪声大得多，这套关口只能抓粗错。
- 合并策略（`glr_merge.py`）：筛选拒 → 拒绝；路由 glr 且 T1/T2 → GLR 标签；路由 glr 但 T3/失败/被关口拦下 → 回退 v1.3 标签（v1c 通过才算）；路由 v13 → v1.3 标签。两种打标器失败机制不同（SfM 要视差和可观测的焦距；前馈要合理的 K），所以回退不是循环论证，但“二者互补”目前只是假设。

## 4. 遵守 AGENTS.md 第 3 节的地方

只用空闲 GPU：新增代码本身不占 GPU（GLRMap 的 CUDA/NPU 部分由 GLRMap 自己的命令决定，用 GLRMap 主机上的卡调度）。线程：`ANNO_THREADS` 默认 4，`glr_worker.sh` 导出 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`；**GLOMAP 自己会开远多于设置的线程**（评估会话里 3 个窗口并发出现过 > 250 线程、负载到 200），共享主机上务必设 `GLR_TASKSET=<cpu 列表>`（`taskset -c`）硬限核数。**`glr_worker.sh` 没有 `GLR_TASKSET` 就拒绝启动**（除非 `GLR_ALLOW_UNCAPPED=1`，只在整机归你时用），`taskset`/`timeout` 套在整个命令模板外，所以 `a && b`、`VAR=x cmd` 形式的模板也受限；`run_batch_glr.sh --phase=b` 会先检查并给出一次性报错。GPU：`--gpus=` 同时给 `run_batch.sh` 和 GLRMap worker（`CUDA_VISIBLE_DEVICES`），没给时 b 阶段用 `pick_gpus 1` 选一张空闲卡，没有空闲卡就不启动；NPU 主机没有 `nvidia-smi` 时不动设备变量，请自己设 `ASCEND_RT_VISIBLE_DEVICES`。不覆盖：v1.3 的所有输出和每个 GLRMap 结果目录（`result`, `result_1`…）永不覆盖；`glr_prepare.py` 对已有作业包只跳过（半成品作业包被改名移开，不删）；`results_glr.csv`、`qc_glr/`、`labels_glr/`、`final/` 是从上述输入**推导出来的视图**，每次阶段 c 重建（“再跑一次就是续跑”），`final/labels/` 里只有软链接（相对路径，整个 OUT 可以搬走），随最新判定改指或移除，从不碰标签文件本身。路径：新代码读 `ANNO_ROOT`（`glr_common.ROOT`），无写死路径；`PY_GLR` 指 GLRMap 环境的 python。没有 `pkill`；worker 被 INT/TERM 打断时按进程组停掉自己启动的 GLRMap 命令。

## 5. 已验证 / 未验证

（沙箱测试结果在 §7。）

**已验证（沙箱，合成数据 + 真实 COLMAP 模型读写，`tests/glr/` 177 项）**：路由与抽帧规则（含无光流统计、长/短片段时的帧数窗口）；第一阶段关口规则（评估会话里的 KITTI / 无人机 / 室内 / 东京 / Ahlbeck 的数值作回归夹具；竖屏焦距规则；关口自身失败 → `error`，不会被读成 ok）；模型和数据库读取（`load_cameras` 与输入相机逐元素一致；路径里带 `#`、`?`、`%` 的数据库）；E3b 米制尺度恢复，以及**被复制的数值代码与原始脚本逐项对拍**（`test_fidelity.py`：`umeyama` / `traj_pct` / `depth_cmp` / `to_grid` / E9 旗标 / `qc_grid` / E3b 从原文件用 `ast` 抽出来比较）；v1c 三项、`jump60`、`fov_out`、关口 stop/suspect/error/缺失、深度缺失/覆盖不足、尺度无证据的分级逻辑；worker 的 split / mono / dry-run / 不覆盖 / 退出码 / 恶意片段名 / SIGTERM 停整组进程 / `taskset`+`timeout` 覆盖整个命令模板 / `RESULT_DIR` 最后写；编排脚本的各阶段、`--gpus` 转发、失败汇总与重跑、`--dry-run` 零写入；merge 的软链接随判定改指；配置文件与默认值一致。

**未验证**：真实 GLRMap 命令行与输出布局（`result.json` 的 `found` / `gate_model` 字段会列出实际找到的文件，用来核对）；PDA 融合深度的文件命名和键（布局猜错时每段以 `depth_missing` 被拒并回退 v1.3，不会悄悄产出无深度标签）；ViPE / DA3 证人在 GLR 标签上的阈值是否合适（沿用 v1.3 的样本内阈值）；路由阈值；`conf` 目前只有“深度是否有限”，没有 PDA 距离置信（评估报告的 Phase 1.4）；GLR 与 v1.3 的失败是否真的互补；回归测试 `tests/regression_A.sh`（没有改任何现有脚本，但按 AGENTS.md 第 9 节仍应在服务器上跑一次）；`glr_collect.load_witnesses` 只用替身模块测过接线（真实的 `eval3d_bench` / `qc_common` 需要 OpenEXR 和服务器上的 bench 目录）。

## 6. 第一次在服务器上跑：核对清单

1. GLRMap 主机：`export GLRMAP_HOME=<GLRMap 检出目录> PY_GLR=<GLRMap 环境的 python（要能 import numpy 和 pycolmap，worker 启动时检查）> ANNO_THREADS=8 GLR_TASKSET=<8 个核>`；`bash pipeline/glr_worker.sh JOB --dry-run` 看命令。作业包里的图片是软链接，拷到别的机器要用 `rsync -aL`（worker 会检查第一张图能不能读）。
2. **mono 模式**（默认）直接跑 GLRMap 自带的 `data_generate_demo/run_all.sh`（它读 `IMAGES`、`OUTPUT`、可选 `MASK_DIR`）。如果你的副本带 NPU 补丁里的 camera_sanity，第一阶段被拦时 `run_all.sh` 自己退出 3，worker 记 `native_gate_stop`。
3. **split 模式**：把 `data_generate_demo/run_roma_pcd.sh` 和 `run_pda_depth.sh` 里 `glomap` 之后的命令抄成 `GLR_DENSE_CMD`（bash 命令行，可以是 `a && b`；占位符 `{images} {out} {s1} {threads} {py} {focal_w} {hfov}` 会变成对环境变量 `GLR_IMAGES GLR_OUT GLR_S1 GLR_THREADS GLR_PY GLR_FOCAL_W GLR_HFOV` 的带引号引用（`{hfov}`/`{focal_w}` 是作业的图像级 K 先验：GeoCalib/Pi3X/DA3 的 HFOV 中位数（度）和同一个量折成焦距/图宽，没有先验时为空；GLRMap 的命令行是否接受初始焦距要看它自己，pycolmap 复刻的 `--focal-w` 可以），裸写或写在 `"..."` 里都行，写在 `'...'` 里不展开），第一阶段用 `GLR_STAGE1_CMD`（默认 `{py} -m reconstruction_flow.glomap --images {images} --output {s1} --camera-model PINHOLE`）。这样关口 stop 时省下 RoMa/PDA 的小时数。`run_batch_glr.sh --worker-mode=split` 没有 `GLR_DENSE_CMD` 会拒绝启动（只有第一阶段的结果没有深度，QC 会拒掉；只想跑第一阶段加关口时设 `GLR_STAGE1_ONLY=1`）。
4. 跑完看 `result*/result.json` 的 `status`、`gate`、`gate_model`（关口看的那个稀疏模型，标签就由它生成；worker 取**路径最浅**的 `sparse/<k>`，默认 `sparse/0` 是最大的那个——COLMAP 增量建图按建出顺序编号，E15 里 4/13 个银参考的最大模型在 `sparse/1`；GLOMAP 若输出多个模型，先确认 0 号是最大的，否则关口会以 `registered_lt_half` 拦下，是安全方向的误拦，不是静默放行）和 `found`：`sparse_models`、`databases`、`pda_fused`（`**/fused/*.npz`，键 `fine`/`coarse`）、`pi3x_prior`。如果 `pda_fused` 为空而你知道深度在别处，改 `glr_collect.read_depth_maps` 的 glob，并在这里记下真实布局（布局猜错时每段都会以 `depth_missing` 被拒并回退 v1.3，不会悄悄产出无深度标签）。状态：`complete`、`complete_gate_stop` / `complete_gate_error`（mono，稠密已跑完但关口 stop / 出错）、`stopped_at_gate`、`native_gate_stop`、`gate_error`（split，没跑稠密）、`stage1_only`、`failed`、`interrupted`。
5. H200 上 `--phase=c`；先看 `qc_glr/*.json` 的 `glr.depth_source` 和 `info_flags`。
6. 在 A 组的一两段（有真值）上用 `--skip-filter --glr-all` 跑，对 `outputs/bench/eval3d_A` 评测 GLR 标签，数字写进 `docs/experiments/LOG.md`，再谈采纳。

## 7. 沙箱测试结果（E15，CPU only）

完整的结果页：`docs/GLR_E15_summary.html`（离线可开）和同内容的 Markdown 版 `docs/GLR_E15_沙箱测试报告.md`；逐片段表在 `docs/experiments/tables/`；片段清单（许可证、sha256、重建方法）在 `docs/experiments/E15_ood_manifest.json`；实验记录在 `docs/experiments/LOG.md` 的 E15 一节。**第一阶段是 pycolmap 复刻（降本版），VLM 是 Claude 替身，没有 GLRMap 代码和 GPU**，所以下面是线索，不是 GLRMap 的验收。

- **片段集**：18 条简单网络视频（Mixkit 7、Pexels 4、Internet Archive 3、美国国家公园局 B-roll 4）+ 16 条硬对照（其中 5 条因许可证只在本地试过）。12 条 Mixkit / Pexels 简单片段各经两个独立代理核验（画面 + 来源）；核验把 `mk_3110` 改成了无视差对照（纯摇镜），发现 `ia_telangana` 带一条淡水印，并纠正了 Mixkit 记录里的作者字段。
- **前端（规则 + VLM 替身）**：简单片段 17 / 18 通过（`px_19834200` 被 `static_text_area` 误拒）；对照 16 / 16 没有变成 GLR 标签（规则拒 4、VLM 拒 9、路由送去 v1.3 的 3 条开车类），15 / 16 条至少被两层独立挡住，只靠一层的是鱼眼那条（只有 VLM 拦住，关口放过）。淡水印那条规则和 VLM 都没看见。
- **出口关口（复刻 v0a）**：简单片段 7 条 ok、11 条 stop（`fx_fy_mismatch` 7、没出模型 2、每图点数太少 2、平面对占比高 1）；对照 15 条 stop，1 条 ok（鱼眼 `ctl_misc_14774`：关口没有拦，靠 VLM 拦）。
- **对银参考**（16 条有参考，2 条参考自己失败）：拦对 9、通过且准 3、**静默错误 4**、误拦 0。轨迹基本是准的（11 / 14 条 ATE ≤ 3.27 %），焦距不是（11 / 14 条差 > 10 %）。换种子（v0b）结果不变；SIMPLE_PINHOLE（v1sp）把过关口的从 7 条提到 13 条，但静默错误从 4 条升到 10 条：它关掉了 `fx_fy_mismatch` 这个焦距不可观测的警报，没有修好焦距。KNOWN_ISSUES 第 10 条。
- **联调演示**（真实片段，稠密深度和证人是替身，没有 v1.3 回退标签）：7 条里 2 条以 T2 出标签（`ia_telangana`；`mk_4206` 焦距偏 22 % 仍通过质检），1 条被前端误拒，2 条被关口拦下（`mk_41574`、`mk_21589`），2 条被质检拒（`ia_victory`：K 证人分歧 16°；`mk_43622`：替身深度对替身证人 `xdepth_k` 0.31，是替身造成的）。
- **没做到**：真实 GLRMap / RoMa / PDA / ViPE / Pi3X 的任何运行；第二轮独立评审（因预算停止）。
