# 代码地图

仓库 = 原服务器上的 `$ANNO_ROOT/code`。我们自己的代码按目录分工如下；第三方仓库不在 git 里（见 `third_party/`）。

| 目录 | 内容 | 状态 |
|---|---|---|
| `pipeline/` | v1.3 一条命令跑一段视频的入口 `run_clip.sh` 与编排脚本 | **生产入口，从这里开始** |
| `tests/` | 回归测试：A 组有真值片段对比参考指标 | 改代码后必跑 |
| `filter/` | ① 筛选：YouTube 采集池、各规则 stage（`s1_cuts`…`s6_track`）、VLM（`vlm_run.py`）、`decide.py`、校准与对比；详见 `filter/README.md` | 生产 |
| `evolution/` | v0 → v1.3 的迭代实验 E1–E13。v1.3 用到的组件：`vipe_worker.py`（ViPE 常驻进程）、`pi3x_cond.py`（Pi3X 以 K 为条件）、`da3_cond.py`（DA3 用给定相机）、`k_select.py` / `fov_sources.py` / `geocalib_bench.py`（内参来源与门控）、`e3_scale_vote.py`（尺度）、`e8_qc_cheap.py`（QC v1c）、`e9_hard_rules.py`（旗标）、`e2_make_shots.py`（切镜切分）；`E12_moge/`、`E13_dualqc/` 是未采纳实验 | 组件 + 实验 |
| `qc/` | v0 质检：7 种方法变体、无真值信号（`qc_signals.py`、`qc_cross.py`）、校准（`qc_calib.py`） | v0，参考 |
| `hybrid/` | v0 混合打标（ViPE 位姿 + 前馈深度等组合），详见 `hybrid/README.md` | v0，参考 |
| `bench/` | 各开源方法在 A/B/C 组上的跑分与评测（ViPE、Pi3X、DA3、VGGT-Ω、Voyager、PURE、SpatialVID）；YouTube 片段工具；`cwild/` 组装 C 组野外集 | 评测 |
| `eval3d/` | 有真值数据集转换（KITTI、TartanAir、vKITTI2、ScanNet++）与公共评测函数 `eval3d_common.py` | 评测 |
| `vipe_eval/` | 最早的 ViPE 在 ScanNet++ 上的单独评测 | 历史 |
| `viz/` | 演示视频（`render_demo.py`）、PLY 导出（`export_ply.py`）、示例网页 | 工具 |
| `deck/` | 汇报图表 | 工具 |
| `envs/` | 各 Python 环境的包列表与重建方法 | |
| `configs/` | `pipeline_v13.yaml`：当前最佳配置与参考指标 | |
| `docs/` | 架构、代码地图、权重与数据、已知问题、实验记录 | |
| `docker/` | 开发容器：`run_dev.sh`（重建容器）、`Dockerfile.vipe`（镜像） | |
| `tools/relocate.py` | 把脚本里的原根路径换成新根路径 | |

## 路径约定

旧脚本里写死了根路径 `/data/home/z00533370`，布局是：

```
$ANNO_ROOT/
  code/      ← 本仓库（第三方仓库也 clone 在这里，被 gitignore）
  cache/     ← Python 环境、模型权重、HF / ModelScope 缓存
  data/      ← 数据集与片段池
  outputs/   ← 所有结果（bench、evolution、filter、qc、viz…）
```

在别的目录或机器上用：`git clone` 到 `$NEW_ROOT/code`，然后 `python3 tools/relocate.py $NEW_ROOT --apply`。新代码请从环境变量 `ANNO_ROOT` 读根路径，不要再写死。
