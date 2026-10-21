"""Assemble C_wild: link the 12 existing YouTube clips, write clips.json with review notes, bench layout, overview grid."""
import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

H = Path("/data/home/z00533370")
C = H / "data/eval3d/C_wild"
CL = C / "clips"
YT = H / "data/youtube_samples/clips"
B = H / "outputs/bench/wild"

# 1) existing YouTube CC clips (outputs/bench/youtube) -> C_wild/clips/<cat>__youtube_cc__<id>
for d in sorted(p for p in YT.iterdir() if p.is_dir() and not p.name.startswith("_")):
    m = json.loads((d / "meta.json").read_text())
    cat, vid = m["cat"], m["id"]
    name = f"{cat}__youtube_cc__{vid}"
    dst = CL / name
    if dst.exists():
        continue
    dst.mkdir(parents=True)
    os.symlink(d / "frames", dst / "frames")
    shutil.copy(d / "sheet.jpg", dst / "sheet.jpg")
    m2 = {"clip": name, "cat": cat, "source": "youtube_cc", "n_frames": m["n_frames"], "hw": m["hw"], "fps_out": 15,
          "license": m.get("license"), "url_or_ref": f"{m.get('webpage_url')}&t={m.get('start')}", "title": m.get("title"),
          "hard_case": False, "notes": "沿用第一轮 YouTube 基准片段", "cuts": m.get("cuts", []), "start": m.get("start")}
    (dst / "meta.json").write_text(json.dumps(m2, indent=1, ensure_ascii=False))

# 2) review notes from the manual sheet check
REVIEW = {
    "indoor_dynamic__sekai__gsZiNIsB7kM_1881": "从户外走进商店，室内外混合",
    "indoor_dynamic__yt__08fFVerIjb0": "机场走廊，人很少，动态程度低",
    "indoor_dynamic__youtube_cc__wsNOEwkBnag": "机场捷运车厢，相机近乎静止，乘客在动",
    "indoor_static__sekai__u53G1pbVumM_375": "服装店内，有两人坐着",
    "indoor_static__yt__IuA1A2s86Qs": "公寓导览，客厅有人",
    "indoor_static__youtube_cc__C8ktx0ohhYQ": "厨房手持，低视差，开头有手指遮挡",
    "outdoor_dynamic__sekai__-YmDhFlqPV8_1155": "夜间，开头透过玻璃拍摄有反光",
    "outdoor_dynamic__sekai__8NCaTPLVyHM_1321": "夜间纽约，相机几乎不动",
    "outdoor_static__sekai__ss0CxHCCybE_135": "远景海岸，运动很慢，视差小",
    "outdoor_static__sekai__FMopxWj7NS4_2517": "航拍峡谷远景，运动慢，视差小",
    "outdoor_static__sekai__b06ThXzoqRI_10369": "湖面倒影，画面有水印",
    "outdoor_static__sekai__qzinq4hBvv4_1114": "俯拍水面和船只，水面难重建",
    "outdoor_static__youtube_cc__IGYh7GIuShM": "远景山景，第一轮中 ViPE 尺度失败",
    "driving__yt__IS38VzYSU_w": "雨天，雨刷频繁遮挡（场景检测误报为切换）",
    "driving__yt__g5iWvBE5SXM": "强逆光，暗部多",
    "driving__yt__kVSRt8p4mlU": "夜间高速，非常暗",
    "driving__yt__FolKZmaRA4A": "夜间降雪",
    "driving__yt__YQl9c5gCjhg": "夜间城市，摩托车多",
    "driving__youtube_cc__BTbkSEotmX0": "底部有字幕叠加",
    "driving__youtube_cc__uRbDaKgz1_w": "底部有字幕叠加",
    "hard__yt__bWVQU5WjYII": "难例（固定机位）：无人机悬停俯拍十字路口",
    "hard__yt__762Iqw-QW2E": "难例（固定机位）：机场人群剪影",
    "embodied__youtube_cc__-jHQSVR8QT0": "第三人称固定机位，机器人擦拭，左上有画中画",
    "embodied__youtube_cc__DbpscUqeXYA": "头戴第一视角割草",
}
rows = []
for d in sorted(CL.iterdir()):
    if not (d / "meta.json").exists():
        continue
    m = json.loads((d / "meta.json").read_text())
    notes = "；".join(x for x in [m.get("notes", ""), REVIEW.get(d.name, "")] if x)
    rows.append({"clip": d.name, "cat": m["cat"], "source": m["source"], "n_frames": m["n_frames"], "hw": m["hw"],
                 "license": m.get("license"), "url_or_ref": m.get("url_or_ref"), "title": m.get("title"),
                 "hard_case": bool(m.get("hard_case", m["cat"] == "hard")), "notes": notes, "cuts": m.get("cuts", [])})
(C / "clips.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

# 3) bench layout
(B / "frames").mkdir(parents=True, exist_ok=True)
for r in rows:
    link = B / "frames" / r["clip"]
    if not link.exists():
        os.symlink(os.path.realpath(CL / r["clip"] / "frames"), link)
(B / "clips.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

# 4) overview grid, 3 columns
ims = []
for r in rows:
    im = Image.open(CL / r["clip"] / "sheet.jpg")
    im = im.resize((540, int(540 * im.height / im.width)))
    c = Image.new("RGB", (540, im.height + 18), "white")
    c.paste(im, (0, 18))
    ImageDraw.Draw(c).text((3, 3), r["clip"][:80], fill="black")
    ims.append(c)
cols = 3
rws = [ims[i:i + cols] for i in range(0, len(ims), cols)]
g = Image.new("RGB", (540 * cols, sum(max(x.height for x in rw) for rw in rws)), "white")
y = 0
for rw in rws:
    for j, im in enumerate(rw):
        g.paste(im, (540 * j, y))
    y += max(x.height for x in rw)
g.save(C / "overview.jpg", quality=70)

from collections import Counter
print(len(rows), "clips,", sum(r["n_frames"] for r in rows), "frames")
print(Counter(r["cat"] for r in rows))
print(Counter((r["cat"], r["source"]) for r in rows))
