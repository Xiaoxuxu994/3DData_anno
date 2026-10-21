"""Collect report assets: merged per-segment data, correlations, trimmed images, light point clouds."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

ROOT = Path("/data/home/z00533370/outputs/vipe/scannetpp_sample30")
E = ROOT / "eval"
OUT = E / "report_assets"
OUT.mkdir(exist_ok=True)

a = pd.read_csv(E / "per_segment.csv")
b = pd.read_csv(E / "geo_per_segment.csv").drop(columns="n_frames")
df = a.merge(b, on="segment")
df.to_csv(E / "all_metrics.csv", index=False, float_format="%.4f")

cols = ["segment", "scene", "n_frames", "traj_len_m", "ate_sim3_cm", "ate_se3_cm", "sim3_scale", "rte_cm", "rre_deg",
        "fov_err_deg", "depth_metric_absrel", "depth_metric_delta1", "depth_seq_absrel", "depth_frame_absrel",
        "pc_acc_m", "pc_comp_m", "pc_fscore_5cm", "pc_pair_scale", "g5_epe_med_px", "g20_epe_med_px",
        "g5_photo_gt", "g5_photo_vipe", "g5_vipe_selfcons_5pct", "sky_frac", "fps", "fx_group"]
rows = df[cols].round(4).to_dict(orient="records")

geo_cols = [c for c in b.columns if c != "segment"]
agg = {
    "geo_mean": df[geo_cols].mean().round(4).to_dict(),
    "geo_median": df[geo_cols].median().round(4).to_dict(),
}
# can ViPE's own multi-view self-consistency (no GT needed) flag bad labels?
q = df[["g5_vipe_selfcons_5pct", "sky_frac", "ate_sim3_cm", "depth_metric_absrel", "pc_fscore_5cm", "g5_epe_med_px"]]
agg["spearman"] = q.corr(method="spearman").round(3).to_dict()
# quality gate example
gate = df.g5_vipe_selfcons_5pct >= 0.5
agg["gate_selfcons_0.5"] = {
    "kept_segments": int(gate.sum()), "kept_frames": int(df.n_frames[gate].sum()),
    "kept_absrel_median": float(df.depth_metric_absrel[gate].median()),
    "dropped_absrel_median": float(df.depth_metric_absrel[~gate].median()),
    "kept_epe_median": float(df.g5_epe_med_px[gate].median()),
    "dropped_epe_median": float(df.g5_epe_med_px[~gate].median()),
}
(OUT / "report_data.json").write_text(json.dumps({"rows": rows, "agg": agg, "summary": json.loads((E / "summary.json").read_text())}))
print(json.dumps(agg, indent=1))


def trim(src, dst, maxw=1800):
    im = Image.open(src).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im = im.crop((max(box[0] - 8, 0), max(box[1] - 8, 0), min(box[2] + 8, im.width), min(box[3] + 8, im.height)))
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    im.save(dst, quality=82)


for p in sorted((E / "vis").glob("*")):
    if p.suffix in (".png", ".jpg"):
        trim(p, OUT / (p.stem + ".jpg"))
    elif p.suffix == ".json":
        d = json.loads(p.read_text())
        rng = np.random.default_rng(0)
        for k in ("gt", "vipe"):
            xyz = np.array(d[k]["xyz"]).reshape(-1, 3)
            c = np.array(d[k]["rgb"]).reshape(-1, 3)
            sel = rng.choice(len(xyz), min(len(xyz), 18000), replace=False)
            d[k] = {"xyz": xyz[sel].round(3).ravel().tolist(), "rgb": c[sel].ravel().tolist()}
        (OUT / p.name).write_text(json.dumps(d, separators=(",", ":")))
for p in E.glob("sky_*.jpg"):
    trim(p, OUT / p.name)
print(sorted(x.name for x in OUT.iterdir()))
