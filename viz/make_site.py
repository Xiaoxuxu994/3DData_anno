"""Build the demo web page for the 3D pseudo-label videos.

  python3 make_site.py META_DIR SITE_DIR      (META_DIR: the <id>.json files written by render_demo.py;
                                               SITE_DIR/videos/<id>.mp4 + <id>.jpg must exist)
"""
import html
import json
import sys
from pathlib import Path

META, SITE = Path(sys.argv[1]), Path(sys.argv[2])
M = {p.stem: json.loads(p.read_text()) for p in META.glob("*.json")}

# Chinese descriptions, keyed by video id (numbers are filled in from the metadata)
DESC = {
    "gt_scannetpp_1c7a683c92": "ScanNet++ 未参与调参的场景，办公室环视扫描（DSLR 广角，HFOV 124°）。轨迹和 GT（黑色虚线）几乎完全重合，点云墙面干净。",
    "gt_tum_fr1_room": "TUM RGB-D 室内静态场景，手持环绕整个房间。深度误差在所有 GT 片段中最低。",
    "gt_tum_fr3_walking_xyz": "TUM 动态场景，两人在镜头前走动。ViPE 的动态 mask 已从点云中去掉行人，相机轨迹没有被行人带偏。",
    "gt_kitti_07": "KITTI 07 城区驾驶。轨迹形状正确（ATE 很低），但米制尺度偏小约 30%：ViPE 的轨迹尺度和深度尺度不一致（k_traj≈2.9，k_depth≈1.15），几何平均后仍不准。这是 v1.3 已知的尺度问题，见 pipeline_best.yaml 的 open_issues。KITTI 的 GT 是稀疏 LiDAR，所以没有 F@5cm。",
    "yt_indoor_static_taipei": "Sekai（来源 YouTube）台北室内步行，走廊和楼梯。QC 分数很低，轨迹平滑，点云结构完整。",
    "yt_indoor_dynamic_mall": "YouTube 曼谷商场步行，画面中行人较多。行人由 ViPE mask 去掉，静态结构保留。",
    "yt_outdoor_static_trail": "YouTube 阿巴拉契亚步道徒步，森林场景，HFOV 99°。树干和地面的结构清楚。",
    "yt_outdoor_dynamic_nanjing": "YouTube 南京城市夜游，灯光复杂，行人多。QC 分数很低。",
    "yt_driving_boston_snow": "YouTube 波士顿夜间降雪驾驶。虽然光照差，但 QC 通过，道路和两侧停放的车辆都重建出来了。",
    "yt_embodied_gopro_mowing": "YouTube 头戴 GoPro 第一视角割草，属于具身/第一人称数据，草坪平面重建良好。",
    "fail_driving_BTbk_wrong_fov": "失败原因：ViPE 估计的 HFOV 是 33°，在 [25°,130°] 门限内，所以 v1.3 直接用了这个 K。但 E13 focal sweep 的自洽性测试（不依赖模型）表明真实 HFOV 约 61°。K 错了，点云会被横向压窄、纵深拉长。QC v1c 判为拒绝（1.95 &gt; 1）。对比见最后一节。",
    "fail_hard_cinematic_cuts": "失败原因：电影感剪辑，片段内有未检出的镜头切换。相邻帧视角跳变超过 60°，HFOV 只有 26°（长焦）。把几个不同镜头当成一条连续轨迹，轨迹会来回乱跳，点云相互重叠。QC 的 jump60 和 fov_out 两个 flag 都触发，v1c 为 2.62。",
    "fail_hard_night_rain": "失败原因：夜间雨天驾驶，车内视角，挡风玻璃上有雨滴和运动模糊。深度和 DA3 相差 1.6 倍阈值，K 的分歧达到 60°，轨迹抖动，点云几乎不可用。QC v1c 为 8.34。",
    "fail_crowd_oxford_street": "失败原因：伦敦牛津街，行人密集，左侧又有近距离反光围挡。深度和 DA3 的分歧是阈值的 2 倍，米制尺度 k=7.5 明显不合理（深度上限被拉到数百米），点云沿视线方向拖尾。QC v1c 为 1.99。",
    "cmp_scannetpp_vipe_vs_v13": "同一场景、同一虚拟相机，两组标注都用 Sim3 对齐到 GT，并且都不做 conf 过滤。ViPE 单独使用时，多帧融合后墙面和桌面会分成几层（F@5cm 0.45）；v1.3（Pi3X 以 ViPE K 为条件）的多帧深度一致，墙面是单层（F@5cm 0.96）。两者 ATE 都很低，差别主要在深度的多视角一致性上。",
    "cmp_BTbk_fov33_vs_fov61": "左边是实际发布的 v1.3 标注（ViPE K，HFOV 33°），右边是把 K×0.5（HFOV 61°）后重新求解的 Pi3X，尺度规则相同，并用 Sim3 对齐到左边。焦距错了以后，左边的街道被压成窄长条，右边街道宽度和房屋比例正常。这说明门限检查不出 K 错误，需要靠 QC 拒绝或 focal sweep 来纠正。",
}

SECTIONS = [
    ("gt", "有真值的评测片段", "A/B 组和 ScanNet++，指标对 GT 计算：ATE 用 Sim3 对齐、以轨迹长度的百分比表示；AbsRel 是米制深度误差；F@5cm 是融合点云的 F-score。"),
    ("yt", "YouTube 野外视频（成功）", "C 组野外视频没有 GT，显示的是 QC v1c 分数（分数 &gt; 1 即拒绝）。这些片段都通过了 QC。"),
    ("fail", "失败样例", "下面的片段都被 QC 拒绝，标题栏是红色。保留在这里，用来说明标注会在哪些情况下出错，以及 QC 为什么要拒绝它们。"),
    ("cmp", "ViPE vs v1.3 对比", "布局：上排是两种标注的深度，下排是两种点云；两边用同一个虚拟相机，深度色标也相同。"),
]


def numbers(m):
    li = m.get("label_info") or {}
    out = []
    if "metrics" in m and isinstance(m["metrics"], dict):
        x = m["metrics"]
        f5 = "n/a（稀疏 GT）" if x["f5"] != x["f5"] else f"{x['f5']:.2f}"
        out += [f"ATE {x['ate_pct']:.2f}%", f"AbsRel {x['absrel']:.3f}", f"F@5cm {f5}", f"Sim3 scale {x['scale']:.2f}"]
    elif "metrics" in m:
        for lab, x in zip(("ViPE", "v1.3"), m["metrics"]):
            out.append(f"{lab}: ATE {x['ate_pct']:.2f}% · AbsRel {x['absrel']:.3f} · F@5cm {x['f5']:.2f}")
    if "qc" in m:
        q = m["qc"]
        dec = "通过" if q["accept"] else "拒绝"
        fl = f"（flags: {', '.join(q['flags'])}）" if q["flags"] else ""
        out.append(f"QC v1c {q['v1c']:.2f} → {dec}{fl}")
    if li:
        out.append(f"HFOV {li['hfov']:.0f}°")
        out.append(f"scale k={li['k']:.2f}")
    return out


def label_version(m):
    if "compare" in m:
        return "对比：" + " vs ".join("ViPE" if w == "vipe" else "v1.3" if w == "v1.3" else "v1.3 (K×0.5)" for w in m["compare"])
    return "v1.3"


def card(vid, m):
    fail = m.get("failure")
    tags = f'<span class="tag ver">{html.escape(label_version(m))}</span>'
    if fail:
        tags += '<span class="tag fail">FAILURE / 失败样例</span>'
    nums = "".join(f"<li>{html.escape(x)}</li>" for x in numbers(m))
    return f"""
  <figure class="card{' failcard' if fail else ''}">
    <video controls muted loop playsinline preload="none" poster="videos/{vid}.jpg">
      <source src="videos/{vid}.mp4" type="video/mp4">
    </video>
    <figcaption>
      <div class="ttl">{html.escape(m['name'])}</div>
      <div class="meta">{html.escape(m['cat'])} · {m['n_frames']} 帧 · <code>{html.escape(m['clip'])}</code></div>
      <div class="tags">{tags}</div>
      <p>{DESC.get(vid, '')}</p>
      <ul class="nums">{nums}</ul>
    </figcaption>
  </figure>"""


ORDER = ["gt_scannetpp_1c7a683c92", "gt_tum_fr1_room", "gt_tum_fr3_walking_xyz", "gt_kitti_07",
         "yt_indoor_static_taipei", "yt_indoor_dynamic_mall", "yt_outdoor_static_trail", "yt_outdoor_dynamic_nanjing",
         "yt_driving_boston_snow", "yt_embodied_gopro_mowing",
         "fail_driving_BTbk_wrong_fov", "fail_hard_cinematic_cuts", "fail_hard_night_rain", "fail_crowd_oxford_street",
         "cmp_scannetpp_vipe_vs_v13", "cmp_BTbk_fov33_vs_fov61"]

body = []
for g, title, intro in SECTIONS:
    ids = [i for i in ORDER if i in M and M[i]["group"] == g]
    body.append(f'<section><h2>{title} <span class="cnt">{len(ids)}</span></h2><p class="intro">{intro}</p>'
                f'<div class="grid{" wide" if g == "cmp" else ""}">' + "".join(card(i, M[i]) for i in ids) + "</div></section>")

page = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D 伪标签演示</title>
<style>
:root {{ --bg:#f6f7f9; --card:#ffffff; --ink:#1b1f27; --mute:#5b6372; --line:#e2e5ea; --accent:#3557c9; --fail:#c62828; --failbg:#fdecec; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14161b; --card:#1d2027; --ink:#e7e9ee; --mute:#9aa2b1; --line:#2c3039; --accent:#8ea6ff; --fail:#ff6b6b; --failbg:#3a1d1f; }} }}
:root[data-theme="dark"] {{ --bg:#14161b; --card:#1d2027; --ink:#e7e9ee; --mute:#9aa2b1; --line:#2c3039; --accent:#8ea6ff; --fail:#ff6b6b; --failbg:#3a1d1f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system, "PingFang SC", "Noto Sans SC", "Segoe UI", sans-serif; }}
main {{ max-width:1320px; margin:0 auto; padding:28px 16px 60px; }}
h1 {{ font-size:26px; margin:0 0 6px; }}
h2 {{ font-size:20px; margin:40px 0 4px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
.cnt {{ font-size:13px; color:var(--mute); font-weight:400; }}
.lead, .intro {{ color:var(--mute); margin:4px 0 14px; max-width:980px; }}
.legend {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; margin:18px 0 6px; }}
.legend div {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; font-size:14px; }}
.legend b {{ display:block; margin-bottom:2px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(400px,1fr)); gap:18px; }}
.grid.wide {{ grid-template-columns:repeat(auto-fill,minmax(560px,1fr)); }}
@media (max-width:640px) {{ .grid, .grid.wide {{ grid-template-columns:1fr; }} }}
.card {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
.card.failcard {{ border-color:var(--fail); }}
video {{ display:block; width:100%; aspect-ratio:16/9; background:#000; }}
figcaption {{ padding:10px 14px 14px; }}
.ttl {{ font-weight:600; }}
.meta {{ color:var(--mute); font-size:13px; word-break:break-all; }}
code {{ font-size:12px; }}
.tags {{ margin:6px 0 2px; }}
.tag {{ display:inline-block; font-size:12px; padding:1px 8px; border-radius:99px; margin-right:6px; border:1px solid var(--line); }}
.tag.ver {{ color:var(--accent); border-color:var(--accent); }}
.tag.fail {{ color:#fff; background:var(--fail); border-color:var(--fail); font-weight:600; }}
.card p {{ margin:6px 0; font-size:14px; }}
.failcard p {{ background:var(--failbg); padding:6px 8px; border-radius:6px; }}
.nums {{ list-style:none; padding:0; margin:6px 0 0; display:flex; flex-wrap:wrap; gap:4px 12px; font-size:13px; color:var(--mute); font-variant-numeric:tabular-nums; }}
footer {{ margin-top:40px; color:var(--mute); font-size:13px; }}
</style>
</head>
<body><main>
<h1>视频 3D 伪标签演示</h1>
<p class="lead">每段视频约 12 秒，15 fps，逐帧播放标注流水线输出的深度、相机轨迹和融合点云。除非标题另有说明，标注都是 <b>v1.3</b>：Pi3X 以门控后的 ViPE K 为条件，米制尺度 k = sqrt(ViPE 深度比 × ViPE 轨迹比)。标题栏给出片段名、类别、标注版本和关键指标。</p>
<div class="legend">
  <div><b>左上 · RGB</b>原始视频帧，左上角显示帧号。</div>
  <div><b>右上 · 深度</b>inverse-depth，turbo 色图（红色近、紫色远）。每个片段使用固定色域，不会闪烁；右侧 colorbar 的单位是米。压暗的像素表示 Pi3X conf 低于该片段的第 25 百分位，黑色表示无效深度。</div>
  <div><b>左下 · 俯视轨迹</b>相机路径随时间增长，颜色表示时间（紫色早，黄色晚）；红色扇形是当前相机朝向和视场角。有 GT 的片段，用黑色虚线画出 Sim3 对齐后的 GT 轨迹。</div>
  <div><b>右下 · 融合点云</b>每 5 帧取一帧反投影并逐步累积，颜色取自 RGB。去掉了飞点、ViPE 动态/天空 mask 和低 conf 像素；室内片段还去掉了天花板。虚拟相机从场景后上方缓慢环绕；黄线是相机路径，红框是当前相机的 frustum。渲染器是 numpy z-buffer splat。</div>
</div>
{''.join(body)}
<footer>QC v1c = max(轨迹与 ViPE 差异 / 3.27%，深度与 DA3 差异 / 0.194，K 分歧 / 13.2°)，大于 1 即拒绝；jump60 和 fov_out 是额外的拒绝 flag（E9）。“learned QC risk” 是 outputs/qc 中学习型 QC 对 Pi3X 纯图像标注给出的风险分，仅供参考。数据来源：outputs/bench/*、outputs/evolution/E3、E3b、E1、E9、E13；渲染代码：code/viz/render_demo.py。</footer>
</main></body></html>
"""
(SITE / "index.html").write_text(page, encoding="utf-8")
print("wrote", SITE / "index.html", len(M), "videos")
