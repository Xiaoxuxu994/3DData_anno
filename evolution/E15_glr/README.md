# evolution/E15_glr：沙箱实验的脚本（E15，CPU only）

实验记录在 `docs/experiments/LOG.md`（E15 一节，**判定标准在看结果之前写下，之后的修订也记在那里**），报告在 `docs/GLR_E15_沙箱测试报告.md`。这里只放脚本；结果（`stage1.json`、稀疏模型、帧、csv）不进 git，写在 `$ANNO_ROOT/outputs/evolution/E15/` 或自己选的新目录。

| 脚本 | 作用 | 输入 → 输出 |
|---|---|---|
| `../../tools/fetch_ood_set.py` | 按 `docs/experiments/E15_ood_manifest.json` 重新下载 OOD 片段（校验 sha256）并切成 15 fps、12 s；不覆盖；打印每条的许可证 | manifest + 输出目录 → `OUTDIR/{raw,clips}/`、`clips.txt` |
| `../../tools/filter_stage_cpu.py` | 在没有 GPU 的机器上**原样**运行 `filter/s*.py`（`.cuda()` 变 no-op） | 池目录 → `pool/<clip>/scores.json` |
| `write_vlm_scores.py` | 把 Claude 替身评审（两位评审，同一份提示词和 JSON 模式，匿名接触表）写成 `vlm_claude*`，供 `filter/decide.py --vlm claude` 读 | 评审 json + 池目录（+ 匿名 id 映射） → `scores.json` |
| `parallax_index.py` | 旋转模型 vs 单应 vs 对极的残差：区分“纯摇镜 / 无视差”和“有真视差”（纯摇镜 ≤ 0.06，有视差 ≥ 0.8） | 视频 → 每片段一行 json |
| `run_sweep.sh` | 一段片段的第一阶段扫描配方：复刻 v0a + v0b / v1sp / v2prior / v3fixed（共用匹配库）+ 银参考 + 可选 960 px | 帧目录 → `OUT/{v0a,…,ref}/` |
| `summarize_sweep.py` | 读 `OUT/<clip>/<run>/stage1.json`，给关口裁决、对银参考的焦距误差 / ATE、预声明的结局（silver-accurate / silent error / false stop），事后的三方焦距表（E 表） | 扫描根目录 → `sweep_runs.csv`、`tables.md` |
| `funnel.py` | 逐片段看哪一层最先挡住：规则 → VLM → 路由 → 第一阶段关口 | `decide.py` 的 csv + `sweep_runs.csv`（+ `glr_route.csv`、manifest）→ `funnel.csv/md`、图 |
| `plot_sweep.py` | 三张图：每条关口规则上各片段的位置、各变体对银参考、结局计数 | `sweep_runs.csv` → png |
| `ocr_recheck.py` | 复核 `filter/s3_ocr.py` 的 int16 溢出（KNOWN_ISSUES 第 9 条）：同一批帧上原算法 vs 改成 float 的算法 | 池目录 → json |
| `dense_standin.py` | **替身**：稀疏点插值成稠密深度，代替 RoMa v2 / Pi3X / PDA（只为让阶段 b → c 跑通；在 `GLR_DENSE_CMD` 里用） | 第一阶段目录 → `OUT/dense/pda/fused/*.npz` |
| `witness_standin.py` | **替身**：用银参考重建当 ViPE / DA3 证人（只为让阶段 c 的质检接线跑通，**不验证质检**） | 参考 + 作业包 → `witness/<clip>/…` |
| `run_demo.sh` | 联调演示：真实筛选 + 路由 + worker（第一阶段复刻 + 真关口）+ 收集 + 质检 + 合并，稠密阶段和证人是替身 | 见脚本头 |

## 复现顺序（单机 CPU，4 核约半天）

```bash
export ANNO_ROOT=...                                    # 写结果的根目录；新目录，不覆盖
python tools/fetch_ood_set.py docs/experiments/E15_ood_manifest.json $ANNO_ROOT/data/e15_ood      # 下载 + 校验 + 切片（需要网络和 ffmpeg）
# 准备帧并跑筛选前三段（pipeline/prep.py → 帧；tools/filter_stage_cpu.py s1_cuts / s2_rules / s3_ocr / s4_flow / s5_geocalib）
# 评审替身：write_vlm_scores.py JUDGES.json POOL_DIR [MAPPING.json]
for c in <clips>; do bash evolution/E15_glr/run_sweep.sh $FRAMES/$c $S1/$c --focal-w <GeoCalib 的 f/W 中位数> --hq960; done
python evolution/E15_glr/summarize_sweep.py $S1 $OUT --decisions decisions.csv --manifest docs/experiments/E15_ood_manifest.json --pool $POOL
python evolution/E15_glr/plot_sweep.py $OUT/sweep_runs.csv $OUT/figs
python evolution/E15_glr/funnel.py decisions.csv $OUT/sweep_runs.csv $OUT/funnel --manifest docs/experiments/E15_ood_manifest.json
```

注意：第一阶段复刻（`tools/stage1_sfm.py`）是 pycolmap 的**降本版**（640 px、60 帧、顺序匹配 overlap 12、无引导匹配），比真实 GLRMap 第一阶段苛刻；银参考是“另一次 SfM”，不是真值。结论的强度见报告第 8 节。
