import sys, json, zipfile, numpy as np, cv2
sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
from eval_vipe import read_exr_zip, DATA, PRED
seg = sys.argv[1]
s = [x for x in json.load(open("/data/home/z00533370/outputs/vipe/scannetpp_sample30/segments.json")) if x["segment"] == seg][0]
dz = read_exr_zip(PRED / "depth" / f"{seg}.zip")
z = zipfile.ZipFile(PRED / "mask" / f"{seg}.zip")
for k in sorted(dz)[::12]:
    m = cv2.imdecode(np.frombuffer(z.read(f"{k:05d}.png"), np.uint8), cv2.IMREAD_UNCHANGED) == 1
    g = cv2.imread(str(DATA / s["scene"] / "depth" / f"{s['frames'][k]}.png"), -1).astype(np.float32)[:, :1034] * 1e-3
    p = dz[k]
    v = g > 0
    for tag, mm in (("sky", m & v), ("non-sky", ~m & v)):
        if mm.sum():
            print(k, tag, f"frac={mm.mean():.2f} gt_med={np.median(g[mm]):.2f} pred_med={np.median(p[mm]):.2f} pred_finite={np.isfinite(p[mm]).mean():.2f} absrel={np.mean(np.abs(p[mm]-g[mm])/g[mm]):.3f}")
img = cv2.imread(f"/data/home/z00533370/outputs/vipe/scannetpp_sample30/frames/{seg}/000000.jpg")
m0 = cv2.imdecode(np.frombuffer(z.read("00000.png"), np.uint8), cv2.IMREAD_UNCHANGED) == 1
img[m0] = (img[m0] * 0.4 + np.array([255, 120, 0]) * 0.6).astype(np.uint8)
cv2.imwrite(f"/data/home/z00533370/outputs/vipe/scannetpp_sample30/eval/sky_{seg}.jpg", img)
