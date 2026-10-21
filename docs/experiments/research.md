# 线 J 调研报告：数据引擎、内参条件化、数据规模与 scaling law

更新：2026-10-17。所有外部数字附链接；“未找到”表示检索后没有拿到可靠数字。我们自己的数字来自本服务器（`outputs/` 各 README）。

## 1. 视频 → 3D 数据引擎（2025–2026）

| 工作 | 来源视频 / 规模 | 筛选 | 标注 | 产出率 / 开销 |
|---|---|---|---|---|
| **VGGT-Ω** [链接](https://arxiv.org/html/2605.15195v1) | 内部约 40M 网络视频；有标注训练集约 33M 序列（公开 + 内部）；另 18M 无标注视频做自监督 | 先去大水印/硬切；VLM 预筛（未写模型名）：约 50% 判太难、40% 可重建但精度低、约 10% 进入下一步 | Grounding DINO 去可动物体 → SIFT / SuperPoint+SuperGlue / ALIKED+LightGlue / VGGSfM 跟踪 → 内点不足时用 VGGT 初始化 → COLMAP BA → patch MVS 深度 | 启发式阈值：注册率 < 99.5%、FoV ∉ [30°,120°]、畸变比 > 0.1、有效深度 < 5% 拒；再用 1000 条人工标注训练的 XGBoost/RF/CatBoost 集成过滤；得到约 0.2M 动态 + 0.6M 静态场景。Sintel 上标注质量 AUC@30° 96.4%、δ1.25 99.3%，MegaSaM 为 62.1% / 77.2%。算力开销：未找到 |
| **SpatialVID** [链接](https://arxiv.org/html/2509.09676v2) | 21,000+ 小时原始 → 2.7M 段、7,089 h；HQ 子集约 1.1K h | 预筛质量/尺寸/标题 → PySceneDetect 切 3–15 s → 美学、亮度、OCR、运动分层评分 | MegaSaM 位姿 + 深度、动态 mask、结构化字幕 | 时长产出率约 34%（7,089/21,000）；GPU 时：未找到。许可：HQ 为 CC-BY-NC-SA 4.0 [HF](https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ) |
| **Sekai** [链接](https://arxiv.org/html/2506.15675v3) | 5,000+ h 步行/无人机，100+ 国家 | 轨迹规则：10 s 内多次 > 150° 反转、相邻帧视角 > 60° 跳变视为异常 | 优化的 MegaSaM + Video Depth Anything（可能 + DepthPro）；Qwen2.5-VL-72B 字幕；GPT-4o 地点 | 约 600 h 带轨迹（二手综述）；许可：HF 卡未写，未找到 |
| **DynPose-100K** [链接](https://arxiv.org/abs/2504.17788) | Panda-70M 中 3.2M 视频 → 100K | 专家模型 + 通用 VLM 级联（去后期处理、变焦、切镜、模糊；要求相机与场景都在动） | 运动 mask + 长程跟踪 + 鲁棒全局 BA | 视频级产出率约 3%（按 100K/3.2M 计算） |
| **Stereo4D** [链接](https://arxiv.org/html/2412.09621v2) | 6,493 个 VR180 视频 → 约 110K 段 | — | 立体视差深度 + 位姿 + 3D 轨迹；基线默认约 6.3 cm（伪米制）[后续工作](https://arxiv.org/pdf/2606.02379) | — |
| **OmniWorld** [链接](https://arxiv.org/html/2509.12201v1) | 合成 OmniWorld-Game + 公开数据整理；二手称 3 亿帧以上 | — | 标注方法：未核对 | — |
| **SAGE** [链接](https://arxiv.org/html/2602.07891v1) | >10K 段 RealEstate10K/DL3DV | ≥ 10 s、≥ 480p；去大运动/动态；只保留 COLMAP 收敛的 | 稀疏 COLMAP 锚点 + 3DGS 渲染监督（不用稠密伪标签） | 4×4090 不到 2 天；直接用 Depth Any Video 伪深度监督“常常无提升甚至变差” |

要点：
1. 头部工作的共同模式是**级联 + 学习式过滤器**，最后一道普遍是人工标注训练的分类器（VGGT-Ω 1000 条）。
2. 产出率普遍很低：VGGT-Ω 的 VLM 只放行约 10%，DynPose 约 3%（视频级），SpatialVID 约 34%（时长）。我们约 7–13%（段级）处在合理区间。
3. 标签质量比数量更关键：VGGT-Ω 附录 B 指出噪声标签会让模型“记住特异噪声”，标准基准未必能发现；SAGE 发现稠密伪深度监督可能有害。

## 2. 内参估计与条件化

| 方法 | 结论 | 来源 |
|---|---|---|
| MapAnything | 两视图平均：加内参后 pose AUC 56.0→64.7、depth τ 49.3→55.1、ray 误差 0.85°→0.19°，点云 rel 0.13 不变；有 Apache 2.0 变体（6 个数据集） | [链接](https://arxiv.org/html/2509.13414v3) |
| WorldMirror | 位姿/内参/深度任一先验都提升；单 token 嵌入优于稠密 raymap；具体分项数字未取到 | [链接](https://arxiv.org/abs/2510.10726) |
| Pi3X / DA3 | 代码支持内参（及位姿）输入；VGGT-Ω 只接受图像 | 本地代码核对 |
| GeoCalib / AnyCalib | AnyCalib 自报 Stanford2D3D vFoV 约 2.55° vs GeoCalib 3.23°（片段数字，表格出处未核对）；第三方 CalibAnyView 称其更优 | [AnyCalib](https://arxiv.org/pdf/2503.12701)、[CalibAnyView](https://arxiv.org/pdf/2605.14615) |
| MoGe-2 / UniDepthV2 | 单图可出 FOV；具体野外 FOV 精度：未找到 | — |

## 3. 3D 基模的训练数据规模

| 模型 | 训练数据 | 来源 |
|---|---|---|
| VGGT-Ω | 约 33M 序列（公开 30+ 数据集 + 内部）+ 0.8M 自建视频标注；称为 VGGT 的 15 倍以上；另 18M 无标注视频；最终 128×H100、240K 步 | [链接](https://arxiv.org/html/2605.15195v1) |
| DA3 | 22 个数据集约 1.26M “场景”（其中 Objaverse ~506K、Trellis ~557K 为物体级）；DA3-Giant 128×H100 约 10 天 | [链接](https://arxiv.org/html/2511.10647) |
| MapAnything | 13 个数据集，场景数合计约 189.5K（按表 1 求和） | [链接](https://arxiv.org/html/2509.13414v3) |
| π³ | 15 个数据集（含内部动态数据）；规模：未找到 | [链接](https://arxiv.org/pdf/2507.13347) |
| VGGT、DUSt3R/MASt3R、CUT3R、Fast3R、MoGe | 本轮未取到可靠数字：未找到 | — |

用户已有约 1M 条开源序列：与 DA3（约 1.26M，含大量物体级）、MapAnything（约 19 万场景）同量级，比 VGGT-Ω 小一个多数量级。

## 4. 已公开的数据 scaling 证据

- **VGGT-Ω**（最直接）：固定约相同训练 token 数、1B 模型，训练序列数按 10× 步长从 2K 增到 2M，3D 点误差单调下降 0.275 → 0.073（6 个数据集平均），作者称“幂律可能刻画这类模型的 scaling”；模型 0.2B→10B 同样单调改善。只给了端点，没有中间值；正文另写“22M 序列”，与图注 2M 不一致。[链接](https://arxiv.org/html/2605.15195v1)
  - 由端点推算（我们的计算，假设为单一幂律）：误差 ∝ D^−α，α = ln(0.275/0.073)/ln(1000) ≈ 0.19；若上端是 22M，则 α ≈ 0.14。
  - 外推：1M → 3M，误差约降 14–19%；1M → 10M，约降 28–35%。
- **SAGE**：100 → 10K 场景呈对数线性提升，附录称 100 → 2K 呈幂律关系；1K → 10K 在 7Scenes DAc 96.0 → 98.0、TUM 91.7 → 93.1。[链接](https://arxiv.org/html/2602.07891v1)
- **DA3** 教师模型：数据 V2 → V3，AbsRel 0.087 → 0.079（数据版本升级，不是规模曲线）。[链接](https://arxiv.org/html/2511.10647)
- **方法论**：Goyal et al. CVPR 2024：高质量数据重复后效用迅速下降，最终低于没见过的低质量数据；最优过滤强度依赖算力预算（质量-数量权衡 QQT）。[链接](https://arxiv.org/abs/2404.07177) 另有 2026 年“A Bitter Lesson for Data Filtering”称高算力下激进过滤可能不必要（只看到摘要，未核对）。[链接](https://www.emergentmind.com/papers/2605.19407)

## 5. 验证 scaling law 需要多少数据：结论与实验设计

**结论**
1. **拟合 scaling law 本身不需要新数据**：用现有 1M 做嵌套子集 10K/30K/100K/300K/1M，就覆盖两个数量级，和 VGGT-Ω 2K→2M 的设计相当。
2. **要检验 1M 以外的外推**，至少再补到 3M（+2M），预期误差下降约 14–19%，足以超过评测噪声（需多种子估计噪声）。更有说服力的是 10M（+9M，一个完整数量级），预期下降约 28–35%。
3. **要回答“伪标签视频数据是否沿同一条曲线”**，不必一开始就做百万级：在 100K、300K 两个点上做“真值 vs 真值 + 伪标签（等量替换 / 额外追加）”和“严 / 松 QC”对比即可，需要约 10 万–30 万条可用伪标签段。

**实验设计**
- 数据轴 D ∈ {10K, 30K, 100K, 300K, 1M}（+3M 若补数据），按来源分层的嵌套子集，保证小集合是大集合的子集。
- 模型轴 N ∈ {约 0.1B, 0.3B, 1B}（VGGT/π³ 结构），每点至少 2–3 个种子估计噪声（小数据点方差大）。
- 两种算力口径：(a) 固定训练 token（同 VGGT-Ω，隔离“数据多样性”）；(b) isoFLOP（Chinchilla 式，拟合 L(N,D) = E + A·N^−a + B·D^−b）。小 D 时数据重复多轮，需要用数据受限 scaling 的修正（重复数据效用递减）。
- 评测：我们的 A 组（7-Scenes、TUM、Bonn、Sintel、KITTI，ETH3D 单列），指标：点误差（同 VGGT-Ω）、ATE、AbsRel、FOV 误差。**严格去重**：训练集排除评测场景（ScanNet++ 泄漏教训）。
- 伪标签臂：D = 100K、300K 时，比较 GT-only / GT + 伪标签等量替换 / GT + 追加伪标签；伪标签分严、松两档 QC；按 SAGE 保留约 3% 真值锚点。
- 多样性：同等帧数下，场景数优先（SAGE）。

**成本**（结合 cost_model.md）
- 现有 1M 的 scaling 实验：只有训练开销。可参考 VGGT-Ω 消融规模（1B、64 卡、150K 步），我们的小模型梯度会便宜很多，具体 GPU 时：未估。
- 30 万条可用伪标签段：v0 管线约 33K–59K GPU·h（8×H200 约 170–310 天）；v1 优化后约 10K–13K GPU·h（8 卡约 55–70 天，64 卡约 7–9 天）。
- +2M 条可用段：v1 约 70K–86K GPU·h（8 卡 1 年以上，64 卡约 1.5–2 个月），8 卡不现实。

## 6. 视频来源与供给

| 来源 | 规模 | 许可 | 来源链接 |
|---|---|---|---|
| YouTube CC-BY | Creative Commons 博客称约 400 万个 CC 视频（日期不明的快照）；YouTube-Commons 含 3,156,703 个视频的转录；Common Pile 从约 2,000 个频道取到 110 万+ 视频、47 万+ 小时 | CC-BY（YouTube 只提供 CC-BY） | [CC](https://creativecommons.org/tag/youtube/)、[YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons)、[Common Pile](https://arxiv.org/pdf/2506.05209) |
| SpatialVID / HQ | 2.7M 段、7,089 h | CC-BY-NC-SA 4.0 | [HF](https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ) |
| Sekai | 5,000+ h；HF 约 34 万行 | 许可：未找到 | [HF](https://huggingface.co/datasets/Lixsp11/Sekai-Project) |
| Panda-70M | 70M 段 | 二手称非商用、仅研究；官方 README 只要求遵守相关许可 | [GitHub](https://github.com/snap-research/Panda-70M/blob/main/README.md) |
| DynPose-100K、Stereo4D | 100K / 约 110K 段 | 未核对 | 见上 |
| OpenVid-1M、HD-VILA-100M | 未核对 | 未找到 | — |

估算（我们的计算）：1M 条可用段在 v1 产出率下需要 4.4–5.6M 个原始 12 s 段、15K–19K 小时原始素材；每个视频取 3–5 段，就要约 90–190 万个视频。CC 视频总量约数百万，但其中适合 3D 的比例（步行/驾驶/无人机/房产/导览）没有公开数字：未找到。我们的定向搜索里 921 个视频有 553 个通过元数据预筛（60%）。判断：CC-BY 单一来源做到 1M 可用段勉强可行，到 10M 不现实；需要非商用数据集（只能用于研究）或另行授权的来源。

## 7. 不确定之处
- VGGT-Ω 的 2M vs 22M 不一致；scaling 指数只能从端点粗估，且是固定 token 口径。
- 我们的 v1 / v2 成本依赖未验证的假设（切镜点切分收益、统一内参后的 QC 保留率、ViPE 降帧精度）。
- 用户 1M 序列的构成（场景数 vs 序列数、单场景多序列、物体级占比）未知；scaling 应按独立场景数计。
- CC-BY 中适合 3D 的视频比例、YouTube 大规模下载的限流：没有公开数据。
