"""Stratified sample of Sekai real clips -> picks json for wild_clip.py (with spares)."""
import json
import pandas as pd

R = "/data/home/z00533370/data/eval3d/C_wild/_raw/sekai/train"
W = pd.read_csv(f"{R}/sekai-real-walking-hq.csv")
D = pd.read_csv(f"{R}/sekai-real-drone.csv")
for df, src in ((W, "walking-hq"), (D, "drone")):
    p = df.videoFile.str.replace(".mp4", "", regex=False).str.rsplit("_", n=2, expand=True)
    df["ytid"], df["f0"], df["f1"] = p[0], p[1].astype(int), p[2].astype(int)
    df["subset"] = src

FPS_SRC = 30.0  # Sekai frame indices are at 30 fps (clips standardised to 30 fps)
groups = [
    # (cat, frame, query, n_main, n_spare, notes)
    ("indoor_static", W, (W.scene.str.startswith("indoor")) & (W.crowdDensity == "empty"), 6, 6, "Sekai 室内、无人"),
    ("indoor_dynamic", W, (W.scene.str.startswith("indoor")) & (W.crowdDensity.isin(["bustling", "crowded"])), 5, 5, "Sekai 室内、人多"),
    ("outdoor_static", W, (W.scene == "outdoor-natural") & (W.crowdDensity == "empty") & (W.timeOfDay == "day"), 3, 4, "Sekai 自然步行、无人"),
    ("outdoor_static", D, (D.scene == "outdoor-natural") & (D.timeOfDay == "day"), 3, 4, "Sekai 航拍、自然"),
    ("outdoor_dynamic", W, (W.scene == "outdoor-urban") & (W.crowdDensity.isin(["bustling", "crowded"])) & (W.timeOfDay == "day"), 6, 6, "Sekai 城市步行、人多、白天"),
    ("outdoor_dynamic", W, (W.scene == "outdoor-urban") & (W.crowdDensity.isin(["moderate", "bustling"])) & ((W.timeOfDay == "night") | (W.weather == "rainy")), 2, 3, "Sekai 城市步行、夜间或雨天"),
]
picks, used = [], set()
for cat, df, q, n, spare, notes in groups:
    sub = df[q & ~df.ytid.isin(used)].drop_duplicates("ytid").sample(frac=1, random_state=7)
    for k, (_, r) in enumerate(sub.head(n + spare).iterrows()):
        used.add(r.ytid)
        dur = (r.f1 - r.f0) / FPS_SRC
        start = int(r.f0 / FPS_SRC + max(0.0, min(10.0, dur - 12)))  # skip into the clip a bit
        picks.append({"name": f"{cat}__sekai__{r.ytid}_{start}", "cat": cat, "source": f"sekai_{r.subset}", "id": r.ytid,
                      "start": start, "len": 12, "license": "YouTube (via Sekai, non-commercial research)",
                      "notes": f"{notes}；{r.location}；{r.weather}/{r.timeOfDay}/{r.crowdDensity}",
                      "spare": k >= n, "extra": {"sekai_videoFile": r.videoFile, "sekai_cameraFile": r.cameraFile,
                                                  "sekai_scene": r.scene}})
json.dump(picks, open("/data/home/z00533370/data/eval3d/C_wild/_raw/picks_sekai.json", "w"), indent=1, ensure_ascii=False)
print(len(picks), sum(not p["spare"] for p in picks))
