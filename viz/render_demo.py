"""Demo videos of the 3D pseudo-labels (v1.3 label, ViPE baseline) for the video annotation pipeline.

  OMP_NUM_THREADS=1 cache/qc-venv/bin/python code/viz/render_demo.py [--workers=5] [--only=id1,id2] [--frames=N]
  -> outputs/viz/videos/<id>.mp4 + <id>.jpg (poster) + <id>.json (caption data)

Single-label video (2x2, 1280x720, 15 fps): RGB | inverse-depth (turbo, fixed range per clip, low-conf dimmed)
                                            top-down trajectory (+ Sim3-aligned GT in black) | fused point cloud
Compare video: depth A | depth B / point cloud A | point cloud B, both labels Sim3-aligned into one frame.

Labels
  v1.3  : Pi3X conditioned on the gated ViPE K (bench/<b>/pi3x_kgate), translation and depth multiplied by
          k_vipe_td = sqrt(k_depth * k_traj) from outputs/evolution/E3/scales_<kind>.csv (e3_scale_vote.py, E3b)
  vipe  : ViPE default profile (bench/<b>/vipe)
  fsweep: v1.3 re-solved with K x s (E13 focal sweep), scaled with the same sqrt(k_depth * k_traj) rule vs ViPE
Point cloud: every 5th frame, ~18k px per frame, flying-pixel filter (relative depth jump > 8 %), ViPE dynamic / sky
masks removed; in single videos also Pi3X conf below the clip's 25th percentile removed.
Renderer: numpy z-buffer splat (2x2 px, nearest point wins), no OpenGL.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

cv2.setNumThreads(1)
ROOT = Path("/data/home/z00533370")
sys.path.insert(0, str(ROOT / "code/bench"))
sys.path.insert(0, str(ROOT / "code/vipe_eval"))
sys.path.insert(0, str(ROOT / "code/qc"))
import eval3d_bench as E3D  # noqa: E402
import eval_bench as ESP  # noqa: E402
from eval_vipe import DATA as SP_DATA  # noqa: E402
from eval_vipe import umeyama  # noqa: E402
from qc_common import vipe_static_mask  # noqa: E402

FF = str(ROOT / "cache/bin/ffmpeg")
OUT = ROOT / "outputs/viz/videos"
EVO = ROOT / "outputs/evolution"
BENCH = {"A": ROOT / "outputs/bench/eval3d_A", "B": ROOT / "outputs/bench/eval3d_B",
         "scannetpp": ROOT / "outputs/bench/scannetpp", "wild": ROOT / "outputs/bench/wild"}
FW, FH, TB = 1280, 720, 56          # frame size, title bar height
PW, PH = 640, (FH - TB) // 2        # panel size (640 x 332)
FPS = 15
FONT = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
FONTB = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf")
FONTCJK = str(ROOT / "cache/fonts/STHeitiMedium.ttc")  # for the Chinese failure tag only
TURBO = (colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
VIRID = (colormaps["plasma"](np.linspace(0.05, 0.85, 256))[:, :3] * 255).astype(np.uint8)
LABEL_DESC = {"v1.3": "v1.3 = Pi3X | gated ViPE K, scale sqrt(k_depth*k_traj) vs ViPE",
              "vipe": "ViPE (default profile)",
              "v0": "v0 hybrid = ViPE pose/K + Pi3X depth (median scale)"}

# --------------------------------------------------------------------------------------------- video specs
SPECS = [
    # --- GT benches
    dict(id="gt_scannetpp_1c7a683c92", group="gt", kind="scannetpp", clip="1c7a683c92_b", indoor=True,
         cat="indoor scan (ScanNet++ unseen)", name="ScanNet++ 1c7a683c92 (unseen scene)"),
    dict(id="gt_tum_fr1_room", group="gt", kind="A", clip="tum__freiburg1_room", indoor=True,
         cat="indoor static (TUM RGB-D)", name="TUM fr1_room"),
    dict(id="gt_tum_fr3_walking_xyz", group="gt", kind="A", clip="tum__freiburg3_walking_xyz", indoor=True,
         cat="indoor dynamic (TUM RGB-D, walking people)", name="TUM fr3_walking_xyz"),
    dict(id="gt_kitti_07", group="gt", kind="A", clip="kitti__kitti_07", cat="driving (KITTI odometry)", name="KITTI 07"),
    # --- YouTube wild, accepted by QC
    dict(id="yt_indoor_static_taipei", group="yt", kind="wild", clip="indoor_static__sekai__ZrrfsI7Y6f0_1455", indoor=True,
         cat="C wild / indoor static", name="Sekai/YouTube ZrrfsI7Y6f0 (Taipei, indoor walk)"),
    dict(id="yt_indoor_dynamic_mall", group="yt", kind="wild", clip="indoor_dynamic__yt__UNn-xlbWc9M", indoor=True,
         cat="C wild / indoor dynamic", name="YouTube UNn-xlbWc9M (Bangkok mall walk)"),
    dict(id="yt_outdoor_static_trail", group="yt", kind="wild", clip="outdoor_static__yt__UHzp18tdeRo",
         cat="C wild / outdoor static", name="YouTube UHzp18tdeRo (Appalachian Trail hike)"),
    dict(id="yt_outdoor_dynamic_nanjing", group="yt", kind="wild", clip="outdoor_dynamic__youtube_cc__PkJw72ytAtU",
         cat="C wild / outdoor dynamic", name="YouTube PkJw72ytAtU (Nanjing city walk)"),
    dict(id="yt_driving_boston_snow", group="yt", kind="wild", clip="driving__yt__FolKZmaRA4A",
         cat="C wild / driving", name="YouTube FolKZmaRA4A (Boston, night snowfall)"),
    dict(id="yt_embodied_gopro_mowing", group="yt", kind="wild", clip="embodied__youtube_cc__DbpscUqeXYA",
         cat="C wild / embodied (head-mounted GoPro)", name="YouTube DbpscUqeXYA (GoPro POV mowing)"),
    # --- failures
    dict(id="fail_driving_BTbk_wrong_fov", group="fail", kind="wild", clip="driving__youtube_cc__BTbkSEotmX0", failure=True,
         cat="C wild / driving", name="YouTube BTbkSEotmX0 (Haarlem drive)",
         reason="ViPE K is wrong (HFOV 33 deg; focal sweep says ~61 deg) and v1.3 inherits it"),
    dict(id="fail_hard_cinematic_cuts", group="fail", kind="wild", clip="hard__yt__PXDArVNjIhs", failure=True,
         cat="C wild / hard: cinematic edit", name="YouTube PXDArVNjIhs (London cinematic)",
         reason="hidden shot cuts: view jump > 60 deg, HFOV 26 deg -> one 3D trajectory is meaningless"),
    dict(id="fail_hard_night_rain", group="fail", kind="wild", clip="hard__yt__YIDw7HIUZt4", failure=True,
         cat="C wild / hard: night rain drive", name="YouTube YIDw7HIUZt4 (night rain, motion blur)",
         reason="night + rain + blur: depth disagrees with DA3 by 1.6x the threshold, K disagreement 60 deg"),
    dict(id="fail_crowd_oxford_street", group="fail", kind="wild", clip="outdoor_dynamic__sekai__CQ-JcP-eRI0_2788", failure=True,
         cat="C wild / outdoor dynamic (dense crowd)", name="Sekai/YouTube CQ-JcP-eRI0 (Oxford Street crowd)",
         reason="dense moving crowd: depth disagrees with DA3 (2x threshold), metric scale k=7.5 implausible"),
    # --- comparisons
    dict(id="cmp_scannetpp_vipe_vs_v13", group="cmp", kind="scannetpp", clip="1c7a683c92_b", indoor=True,
         cat="indoor scan (ScanNet++ unseen)", name="ScanNet++ 1c7a683c92: ViPE vs v1.3", compare=["vipe", "v1.3"], frame="gt"),
    dict(id="cmp_BTbk_fov33_vs_fov61", group="cmp", kind="wild", clip="driving__youtube_cc__BTbkSEotmX0", cat="C wild / driving",
         name="BTbkSEotmX0: v1.3 with ViPE K (HFOV 33 deg) vs focal-sweep K (61 deg)", compare=["v1.3", "fsweep0.50"],
         frame="left"),
]


# --------------------------------------------------------------------------------------------- data
def clip_meta(kind, clip):
    return next(x for x in json.loads((BENCH[kind] / "clips.json").read_text()) if x["clip"] == clip)


def grid(kind, c):
    if kind in ("A", "B"):
        z = np.load(BENCH[kind] / "gt" / f"{c['clip']}.npz", allow_pickle=True)
        return tuple(z["depth"].shape[1:])
    if kind == "scannetpp":
        return ESP.H, ESP.W
    h, w = c["hw"]
    s = min(1.0, 960 / max(h, w))
    return int(round(h * s)) // 2 * 2, int(round(w * s)) // 2 * 2


def load_raw(kind, method, c, H, W, B=None, clip=None):
    clip = clip or c["clip"]
    if kind == "scannetpp" and B is None:
        c2w, K, D, _ = ESP.load_method(method, clip, c["n_frames"])
    else:
        c2w, K, D, _ = E3D.load_method(B or BENCH[kind], method, clip, c["n_frames"], H, W)
    return c2w.astype(np.float64), K.astype(np.float64), D.astype(np.float32)


def conf_map(path, H, W):
    z = np.load(path)
    if "conf" not in z.files:
        return None
    cf = z["conf"].astype(np.float32)
    return np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in cf])


def scale_td(c2w, D, c2w_v, D_v):
    """Same rule as e3_scale_vote.py: k = sqrt(k_traj * k_depth) towards ViPE (k_traj falls back to k_depth)."""
    ks = np.linspace(0, len(D) - 1, 10).astype(int)
    a, b = D_v[ks], D[ks]
    mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    r = float(np.median(a[mk]) / np.median(b[mk]))
    pb, pm = c2w[:, :3, 3], c2w_v[:, :3, 3]
    Lm = np.linalg.norm(np.diff(pm, axis=0), axis=1).sum()
    s, R, t = umeyama(pb, pm, True)
    res = np.sqrt(((s * pb @ R.T + t - pm) ** 2).sum(1).mean()) / max(Lm, 1e-9)
    st = s if (Lm > 0.02 * np.median(a[mk]) and res < 0.2 and s > 0) else r
    return float(np.sqrt(st * r)), r, st


def load_label(kind, c, H, W, which):
    clip = c["clip"]
    info = {"which": which}
    if which == "v1.3":
        c2w, K, D = load_raw(kind, "pi3x_kgate", c, H, W)
        sc = pd.read_csv(EVO / "E3" / f"scales_{kind}.csv").set_index("clip")
        k = float(sc.loc[clip, "k_vipe_td"])
        info.update(k=k, k_d=float(sc.loc[clip, "k_vipe_d"]), k_t=float(sc.loc[clip, "k_vipe_t"]))
        conf = conf_map(BENCH[kind] / "pi3x_kgate" / clip / "pred.npz", H, W)
    elif which == "vipe":
        c2w, K, D = load_raw(kind, "vipe", c, H, W)
        k, conf = 1.0, None
    elif which.startswith("fsweep"):
        s = which[len("fsweep"):]
        fs = EVO / "E13" / "fsweep"
        c2w, K, D = load_raw(kind, clip, c, H, W, B=fs, clip=f"s{s}")
        cv_, _, Dv = load_raw(kind, "vipe", c, H, W)
        k, kd, kt = scale_td(c2w, D, cv_, Dv)
        info.update(k=k, k_d=kd, k_t=kt, focal_scale=float(s))
        conf = conf_map(fs / clip / f"s{s}" / "pred.npz", H, W)
    else:
        raise ValueError(which)
    c2w = c2w.copy()
    c2w[:, :3, 3] *= k
    D = D * k
    info["hfov"] = float(np.degrees(2 * np.arctan(W / 2 / np.median(K[:, 0, 0]))))
    return dict(c2w=c2w, K=K, D=D, conf=conf, info=info)


def gt_c2w(kind, c):
    if kind in ("A", "B"):
        return np.load(BENCH[kind] / "gt" / f"{c['clip']}.npz", allow_pickle=True)["c2w"].astype(np.float64)
    if kind == "scannetpp":
        meta = np.load(SP_DATA / c["scene"] / "scene_metadata.npz")
        idx = {f: i for i, f in enumerate(meta["images"])}
        return meta["trajectories"][[idx[f] for f in c["frames"]]].astype(np.float64)
    return None


def transform(L, s, R, t):
    """Apply x -> s R x + t to a label (cameras + depth)."""
    c2w = L["c2w"].copy()
    c2w[:, :3, :3] = R @ c2w[:, :3, :3]
    c2w[:, :3, 3] = s * c2w[:, :3, 3] @ R.T + t
    out = dict(L, c2w=c2w, D=L["D"] * s)
    return out


class Frames:
    def __init__(self, kind, clip, H, W):
        self.d, self.H, self.W = BENCH[kind] / "frames" / clip, H, W

    def __call__(self, k):
        im = cv2.imread(str(self.d / f"{k:06d}.jpg"))
        return cv2.cvtColor(cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------- metrics for titles
def gt_metrics(kind, clip, method):
    f = EVO / "E3b" / "raw" / f"{'A' if kind == 'A' else kind}.csv"
    if kind == "B":
        f = EVO / "E3b" / "raw" / "B.csv"
    d = pd.read_csv(f)
    name = {"v1.3": "pi3x_kgate@vipe_td", "vipe": "vipe"}[method]
    r = d[(d["method"] == name) & (d["clip"] == clip)].iloc[0]
    return dict(ate_pct=float(r["ate_sim3_cm"] / r["traj_len_m"]), absrel=float(r["depth_metric_absrel"]),
                f5=float(r["pc_fscore_5cm"]), scale=float(r["sim3_scale"]), traj_len=float(r["traj_len_m"]))


def qc_metrics(clip):
    q = pd.read_csv(EVO / "E1" / "qc_rule_cheap.csv")
    q = q[(q["group"] == "C") & (q["clip"] == clip)].iloc[0]
    f = pd.read_csv(EVO / "E9" / "flags.csv")
    f = f[(f["group"] == "C") & (f["clip"] == clip)].iloc[0]
    r = pd.read_csv(ROOT / "outputs/qc/qc_C_by_clip.csv").set_index("clip").loc[clip]
    score = float(q["cheap"])
    flags = [n for n in ("jump60", "fov_out") if int(f[n])]
    return dict(v1c=score, xtraj=float(q["xtraj_ba"]), xdepth=float(q["xdepth_k"]), kdis=float(q["kdis"]),
                flags=flags, accept=bool(score <= 1 and not flags), risk_pi3x=float(r["pi3x"]),
                risk_min=float(r["min_risk"]))


# --------------------------------------------------------------------------------------------- drawing helpers
_fonts = {}


def font(sz, bold=False):
    key = (sz, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONTCJK if bold == "cjk" else FONTB if bold else FONT, sz)
    return _fonts[key]


def draw_texts(frame, texts):
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    for x, y, s, sz, col, *rest in texts:
        bold = rest[0] if rest else False
        if rest and len(rest) > 1 and rest[1]:  # background box
            bb = dr.textbbox((x, y), s, font=font(sz, bold))
            dr.rectangle((bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3), fill=rest[1])
        dr.text((x, y), s, font=font(sz, bold), fill=col)
    return np.asarray(im)


def fit(img, w, h):
    """Letterbox img into (h, w): returns canvas, (x0, y0, scale)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, (x0, y0, s)


def nice_step(span, target=6):
    for m in 10.0 ** np.arange(-2, 4):
        for b in (1, 2, 5):
            if span / (b * m) <= target:
                return b * m
    return 1000.0


def fmt_m(v):
    return f"{v:g} m" if v >= 1 else f"{v * 100:g} cm"


class DepthPanel:
    CBW = 74  # width reserved for the colorbar

    def __init__(self, D, conf, dmin, dmax, dim_thr=None):
        self.dmin, self.dmax, self.thr = dmin, dmax, dim_thr
        self.imin, self.imax = 1 / dmax, 1 / dmin
        # static colorbar
        bh = int(PH * 0.72)
        self.bh, self.by = bh, (PH - bh) // 2 + 6
        v = np.linspace(1, 0, bh)
        self.bar = np.repeat(TURBO[np.clip((v * 255).astype(int), 0, 255)][:, None], 14, 1)
        cands = [0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200]
        ticks = [d for d in cands if dmin * 1.02 <= d <= dmax * 0.98]
        if len(ticks) > 6:
            ticks = ticks[:: int(np.ceil(len(ticks) / 6))]
        self.ticks = [(dmin, True), (dmax, True)] + [(d, False) for d in ticks]

    def color(self, d, c=None):
        valid = np.isfinite(d) & (d > 0)
        inv = np.where(valid, 1 / np.where(valid, d, 1), 0)
        x = np.clip((inv - self.imin) / (self.imax - self.imin), 0, 1)
        img = TURBO[(x * 255).astype(np.uint8)]
        if c is not None and self.thr is not None:
            img = np.where((c < self.thr)[..., None], (img * 0.3).astype(np.uint8), img)
        img[~valid] = 0
        return img

    def panel(self, d, c=None, label="Depth"):
        img = self.color(d, c)
        out = np.zeros((PH, PW, 3), np.uint8)
        body, (x0, y0, s) = fit(img, PW - self.CBW, PH - 26)
        out[26:, :PW - self.CBW] = body
        bx = PW - self.CBW + 8
        out[self.by:self.by + self.bh, bx:bx + 14] = self.bar
        texts = [(8, 4, label, 14, (235, 235, 235))]
        used, labs = [], set()
        for dv, edge in self.ticks:
            y = self.by + int((1 - (1 / dv - self.imin) / (self.imax - self.imin)) * (self.bh - 1))
            lab = f"{dv:.2f}" if edge and dv < 1 else f"{dv:.1f}" if edge and dv < 10 else f"{dv:.0f}" if dv >= 10 else f"{dv:g}"
            if any(abs(y - yy) < 13 for yy in used) or lab in labs:
                continue
            used.append(y)
            labs.add(lab)
            out[y, bx - 3:bx + 17] = 255
            texts.append((bx + 19, y - 7, lab, 11, (220, 220, 220)))
        texts.append((bx - 2, self.by - 20, "depth m", 11, (220, 220, 220)))
        return out, texts


def up_vector(c2w):
    u = -c2w[:, :3, 1].mean(0)
    return u / np.linalg.norm(u)


def plane_axes(c2w, up, pts=None):
    P = c2w[:, :3, 3]
    a = np.eye(3)[np.argmin(np.abs(up))]
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    Q = np.stack([P @ e1, P @ e2], 1)
    Q = Q - Q.mean(0)
    if np.ptp(Q, 0).max() > 1e-6:
        w, V = np.linalg.eigh(Q.T @ Q)
        v = V[:, -1]
        f1 = v[0] * e1 + v[1] * e2
    else:
        f1 = e1
    # orient so that the mean camera heading points "up" in the panel
    fwd = c2w[:, :3, 2].mean(0)
    f2 = np.cross(up, f1)
    if fwd @ f2 < 0:
        f2 = -f2
    f1 = np.cross(f2, up)
    return f1 / np.linalg.norm(f1), f2 / np.linalg.norm(f2)


def dashes(P, on, off):
    """Split a pixel polyline into dash segments (on/off lengths in px)."""
    out, pos, period = [], 0.0, on + off
    for a, b in zip(P[:-1], P[1:]):
        L = float(np.linalg.norm(b - a))
        t = 0.0
        while t < L:
            ph = (pos + t) % period
            if ph < on:
                t1 = min(L, t + on - ph)
                out.append((a + (b - a) * t / max(L, 1e-9), a + (b - a) * t1 / max(L, 1e-9)))
            else:
                t1 = min(L, t + period - ph)
            t = t1
        pos += L
    return out


class TrajPanel:
    def __init__(self, c2w, K, W, up, gt=None, label="v1.3"):
        self.c2w, self.gt, self.label = c2w, gt, label
        self.e1, self.e2 = plane_axes(c2w, up)
        self.hfov = 2 * np.arctan(W / 2 / K[:, 0, 0])
        P = [c2w[:, :3, 3]] + ([gt] if gt is not None else [])
        Q = np.concatenate([self.proj(p) for p in P])
        lo, hi = Q.min(0), Q.max(0)
        span = np.maximum(hi - lo, 1e-3)
        pad = 0.12 * span.max() + 1e-3
        lo, hi = lo - pad, hi + pad
        aw, ah = PW - 40, PH - 56
        self.sc = min(aw / (hi[0] - lo[0]), ah / (hi[1] - lo[1]))
        cx, cy = (lo + hi) / 2
        self.o = np.array([PW / 2 - cx * self.sc, 30 + ah / 2 + 10 + cy * self.sc])
        self.lo, self.hi = lo, hi
        self.step = nice_step(max(hi - lo), 7)
        self.bg = self._background()
        self.pred_px = self.px(self.proj(c2w[:, :3, 3]))
        self.gt_px = self.px(self.proj(gt)) if gt is not None else None

    def proj(self, P):
        return np.stack([P @ self.e1, P @ self.e2], 1)

    def px(self, Q):
        return np.stack([self.o[0] + Q[:, 0] * self.sc, self.o[1] - Q[:, 1] * self.sc], 1)

    def _background(self):
        img = np.full((PH, PW, 3), 246, np.uint8)
        s = self.step
        for v in np.arange(np.floor(self.lo[0] / s) * s, self.hi[0] + s, s):
            x = int(round(self.o[0] + v * self.sc))
            if 0 <= x < PW:
                img[:, x] = 222
        for v in np.arange(np.floor(self.lo[1] / s) * s, self.hi[1] + s, s):
            y = int(round(self.o[1] - v * self.sc))
            if 0 <= y < PH:
                img[y, :] = 222
        return img

    def panel(self, i):
        img = self.bg.copy()
        p = self.pred_px
        n = len(p)
        for j in range(1, i + 1):
            col = tuple(int(x) for x in VIRID[int(255 * j / max(n - 1, 1))])
            cv2.line(img, tuple(np.round(p[j - 1] * 4).astype(int)), tuple(np.round(p[j] * 4).astype(int)), col, 3,
                     cv2.LINE_AA, shift=2)
        if self.gt_px is not None:
            for a, b in dashes(self.gt_px[: i + 1], 7.0, 5.0):
                cv2.line(img, tuple(np.round(a * 4).astype(int)), tuple(np.round(b * 4).astype(int)), (0, 0, 0), 1,
                         cv2.LINE_AA, shift=2)
        # current camera: frustum wedge in the ground plane
        f = self.c2w[i, :3, 2]
        fx, fy = f @ self.e1, f @ self.e2
        ang = np.arctan2(-fy, fx)  # pixel space (y down)
        h = self.hfov[i] / 2
        L = 34
        c = p[i]
        tri = np.array([c, c + L * np.array([np.cos(ang - h), np.sin(ang - h)]),
                        c + L * np.array([np.cos(ang + h), np.sin(ang + h)])])
        ov = img.copy()
        cv2.fillPoly(ov, [np.round(tri * 4).astype(np.int32)], (230, 60, 40), cv2.LINE_AA, shift=2)
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        cv2.polylines(img, [np.round(tri * 4).astype(np.int32)], True, (200, 40, 30), 2, cv2.LINE_AA, shift=2)
        cv2.circle(img, tuple(np.round(c).astype(int)), 4, (200, 40, 30), -1, cv2.LINE_AA)
        texts = [(8, 4, "Top-down camera trajectory", 14, (30, 30, 30)),
                 (8, PH - 20, f"grid = {fmt_m(self.step)}", 12, (90, 90, 90))]
        lx = PW - 250
        texts.append((lx + 26, 6, self.label + " (color = time)", 12, (30, 30, 30)))
        for xa in range(20):
            cv2.line(img, (lx + xa, 14), (lx + xa + 1, 14), tuple(int(x) for x in VIRID[int(255 * xa / 19)]), 3)
        tb = np.repeat(VIRID[None, ::2], 8, 0)
        img[PH - 18:PH - 10, PW - 190:PW - 190 + tb.shape[1]] = tb
        texts.append((PW - 190 - 42, PH - 22, "early", 11, (60, 60, 60)))
        texts.append((PW - 190 + tb.shape[1] + 6, PH - 22, "late", 11, (60, 60, 60)))
        if self.gt_px is not None:
            texts.append((lx + 26, 24, "GT (Sim3-aligned, dashed)", 12, (30, 30, 30)))
            for xa in range(lx, lx + 20, 8):
                cv2.line(img, (xa, 32), (xa + 4, 32), (0, 0, 0), 1, cv2.LINE_AA)
        return img, texts


# --------------------------------------------------------------------------------------------- point cloud
def build_cloud(L, frames, mask, every=5, target=18000, use_conf=True, dcap=None):
    D, K, c2w, conf = L["D"], L["K"], L["c2w"], L["conf"]
    n, H, W = D.shape
    thr = float(np.percentile(conf[::10], 25)) if (conf is not None and use_conf) else None
    if dcap is None:
        sm = D[::10]
        sm = sm[np.isfinite(sm) & (sm > 0)]
        dcap = float(min(np.percentile(sm, 97), 6 * np.median(sm)))
    s = max(1, int(round(np.sqrt(H * W / target))))
    vv, uu = np.mgrid[s // 2:H:s, s // 2:W:s]
    P, C, F = [], [], []
    for k in range(0, n, every):
        d = D[k]
        ok = np.isfinite(d) & (d > 0) & (d < dcap)
        dd = np.where(ok, d, 0)
        jump = np.zeros_like(dd)
        jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(np.diff(dd, axis=1)))
        jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(np.diff(dd, axis=1)))
        jump[1:] = np.maximum(jump[1:], np.abs(np.diff(dd, axis=0)))
        jump[:-1] = np.maximum(jump[:-1], np.abs(np.diff(dd, axis=0)))
        ok &= jump < 0.08 * np.maximum(dd, 1e-6)
        if thr is not None:
            ok &= conf[k] >= thr
        if mask is not None:
            ok &= mask[k] == 0
        m = ok[vv, uu]
        z = d[vv, uu][m]
        u, v = uu[m].astype(np.float64), vv[m].astype(np.float64)
        Kk = K[k]
        X = np.stack([(u - Kk[0, 2]) / Kk[0, 0] * z, (v - Kk[1, 2]) / Kk[1, 1] * z, z], 1)
        P.append(X @ c2w[k, :3, :3].T + c2w[k, :3, 3])
        C.append(frames(k)[vv[m], uu[m]])
        F.append(np.full(len(z), k, np.int32))
    return dict(P=np.concatenate(P).astype(np.float32), C=np.concatenate(C), F=np.concatenate(F), thr=thr, dcap=dcap)


def ceiling_cut(cloud, c2w, up):
    """Indoor only: drop points well above the cameras (ceiling) so the orbit view can look into the room."""
    h = cloud["P"] @ up.astype(np.float32)
    hc = c2w[:, :3, 3] @ up
    cut = hc.max() + 0.5 * max(hc.max() - np.percentile(h, 3), 0.3)
    keep = h < cut
    return dict(cloud, P=cloud["P"][keep], C=cloud["C"][keep], F=cloud["F"][keep])


def look_at(pos, target, up):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.eye(3)[np.argmin(np.abs(z))])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # world -> cam rotation rows


class Orbit:
    """Virtual camera above and behind the scene, slowly sweeping +-35 deg in azimuth over the clip."""

    def __init__(self, P, c2w, up, n, el=45.0, sweep=35.0, fov=60.0, zoom=1.0):
        self.up, self.n = up, n
        allp = np.concatenate([P, c2w[:, :3, 3].astype(np.float32)])
        self.center = 0.5 * (np.percentile(allp, 5, 0) + np.percentile(allp, 95, 0))
        r = np.percentile(np.linalg.norm(allp - self.center, axis=1), 85)
        fwd = c2w[:, :3, 2].mean(0)
        fwd = fwd - (fwd @ up) * up
        if np.linalg.norm(fwd) < 1e-6:
            fwd = np.cross(up, np.eye(3)[np.argmin(np.abs(up))])
        self.e1 = fwd / np.linalg.norm(fwd)
        self.e2 = np.cross(up, self.e1)
        self.f = (PW / 2) / np.tan(np.radians(fov / 2))
        self.dist = r / np.tan(np.radians(fov / 2)) * 1.6 / zoom
        self.el, self.sweep = np.radians(el), np.radians(sweep)

    def cam(self, i):
        t = i / max(self.n - 1, 1)
        az = -self.sweep + 2 * self.sweep * t
        back = -(np.cos(az) * self.e1 + np.sin(az) * self.e2)
        pos = self.center + self.dist * (np.cos(self.el) * back + np.sin(self.el) * self.up)
        return look_at(pos, self.center, self.up), pos


def splat(P, C, R, pos, f, W, H, near=1e-3):
    Xc = (P - pos.astype(np.float32)) @ R.T.astype(np.float32)
    z = Xc[:, 2]
    m = z > near
    Xc, z, C = Xc[m], z[m], C[m]
    u = (f * Xc[:, 0] / z + W / 2).astype(np.int32)
    v = (f * Xc[:, 1] / z + H / 2).astype(np.int32)
    m = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, z, C = u[m], v[m], z[m], C[m]
    img = np.full((H * W, 3), 18, np.uint8)
    if len(z) == 0:
        return img.reshape(H, W, 3)
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    shade = 1.0 - 0.45 * np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    C = (C * shade[:, None]).astype(np.uint8)
    pix = np.concatenate([v * W + u, v * W + u + 1, (v + 1) * W + u, (v + 1) * W + u + 1])
    zz = np.tile(z, 4)
    CC = np.tile(C, (4, 1))
    order = np.lexsort((zz, pix))
    ps = pix[order]
    first = np.ones(len(ps), bool)
    first[1:] = ps[1:] != ps[:-1]
    sel = order[first]
    img[pix[sel]] = CC[sel]
    return img.reshape(H, W, 3)


def project(Xw, R, pos, f, W, H):
    Xc = (Xw - pos) @ R.T
    ok = Xc[:, 2] > 1e-3
    uv = np.stack([f * Xc[:, 0] / np.maximum(Xc[:, 2], 1e-3) + W / 2, f * Xc[:, 1] / np.maximum(Xc[:, 2], 1e-3) + H / 2], 1)
    return uv, ok


def cloud_panel(cloud, cnt, orbit, i, c2w, K, Wimg, Himg, label, frus_size):
    R, pos = orbit.cam(i)
    img = splat(cloud["P"][:cnt], cloud["C"][:cnt], R, pos, orbit.f, PW, PH)
    # camera path so far
    uv, ok = project(c2w[: i + 1, :3, 3], R, pos, orbit.f, PW, PH)
    if ok.all() and len(uv) > 1:
        cv2.polylines(img, [np.round(uv * 4).astype(np.int32)], False, (255, 220, 60), 2, cv2.LINE_AA, shift=2)
    # current camera frustum
    Kk = K[i]
    corners = np.array([[0, 0], [Wimg, 0], [Wimg, Himg], [0, Himg]], np.float64)
    rays = np.stack([(corners[:, 0] - Kk[0, 2]) / Kk[0, 0], (corners[:, 1] - Kk[1, 2]) / Kk[1, 1], np.ones(4)], 1)
    rays *= frus_size
    Xw = np.concatenate([c2w[i, :3, 3][None], rays @ c2w[i, :3, :3].T + c2w[i, :3, 3]])
    uv, ok = project(Xw, R, pos, orbit.f, PW, PH)
    if ok.all():
        q = np.round(uv * 4).astype(np.int32)
        for j in range(1, 5):
            cv2.line(img, tuple(q[0]), tuple(q[j]), (255, 80, 60), 2, cv2.LINE_AA, shift=2)
        cv2.polylines(img, [q[1:]], True, (255, 80, 60), 2, cv2.LINE_AA, shift=2)
    return img, [(8, 4, label, 14, (235, 235, 235))]


# --------------------------------------------------------------------------------------------- jobs
def depth_range(D):
    sm = D[::6]
    sm = sm[np.isfinite(sm) & (sm > 0)]
    return float(np.percentile(sm, 2)), float(np.percentile(sm, 98))


def encoder(path):
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW}x{FH}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "27", "-maxrate", "2.4M", "-bufsize", "4.8M",
           "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def title_texts(line1, line2, failure=False):
    t = [(12, 6, line1, 19, (255, 255, 255), True), (12, 32, line2, 13, (215, 215, 215))]
    if failure:
        t.append((FW - 220, 9, "FAILURE / 失败样例", 18, (255, 255, 255), "cjk", (215, 25, 25)))
    return t


def compose(panels, line_texts, failure=False):
    fr = np.zeros((FH, FW, 3), np.uint8)
    fr[:TB] = (120, 28, 28) if failure else (32, 36, 48)
    texts = list(line_texts)
    for (img, tx), (r, cidx) in zip(panels, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        y0, x0 = TB + r * PH, cidx * PW
        fr[y0:y0 + PH, x0:x0 + PW] = img
        texts += [(x + x0, y + y0, *rest) for x, y, *rest in tx]
    fr[TB:, PW - 1:PW + 1] = 70
    fr[TB + PH - 1:TB + PH + 1] = 70
    return draw_texts(fr, texts)


def run(spec):
    t0 = time.time()
    kind, clip = spec["kind"], spec["clip"]
    c = clip_meta(kind, clip)
    H, W = grid(kind, c)
    n = c["n_frames"]
    nf = int(os.environ.get("VIZ_FRAMES", "0")) or n
    frames = Frames(kind, clip, H, W)
    mask = vipe_static_mask(BENCH[kind], clip, n, (H, W))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{spec['id']}.mp4"
    meta = dict(spec, n_frames=n, grid=[H, W], dyn_mask=mask is not None)
    if "compare" in spec:
        render_compare(spec, c, H, W, n, nf, frames, mask, out, meta)
    else:
        render_single(spec, c, H, W, n, nf, frames, mask, out, meta)
    meta["sec"] = round(time.time() - t0, 1)
    meta["mb"] = round(out.stat().st_size / 1e6, 2)
    (OUT / f"{spec['id']}.json").write_text(json.dumps(meta, indent=1, default=float))
    return spec["id"], meta["sec"], meta["mb"]


def render_single(spec, c, H, W, n, nf, frames, mask, out, meta):
    kind, clip = spec["kind"], spec["clip"]
    L = load_label(kind, c, H, W, "v1.3")
    meta["label"] = "v1.3"
    meta["label_info"] = L["info"]
    G = gt_c2w(kind, c)
    gt_al = None
    if G is not None:
        s, R, t = umeyama(G[:, :3, 3], L["c2w"][:, :3, 3], True)
        gt_al = s * G[:, :3, 3] @ R.T + t
        m = gt_metrics(kind, clip, "v1.3")
        mv = gt_metrics(kind, clip, "vipe")
        meta["metrics"], meta["metrics_vipe"] = m, mv
        line2 = (f"label v1.3 (Pi3X | gated ViPE K, k={L['info']['k']:.2f})   ATE {m['ate_pct']:.2f}% of path   "
                 f"AbsRel {m['absrel']:.3f}   F@5cm {m['f5']:.2f}" if np.isfinite(m["f5"]) else
                 f"label v1.3 (Pi3X | gated ViPE K, k={L['info']['k']:.2f})   ATE {m['ate_pct']:.2f}% of path   "
                 f"AbsRel {m['absrel']:.3f}   F@5cm n/a (sparse GT)")
        line2 += f"   Sim3 scale {m['scale']:.2f}"
    else:
        q = qc_metrics(clip)
        meta["qc"] = q
        dec = "ACCEPT" if q["accept"] else "REJECT"
        fl = (" +" + ",".join(q["flags"])) if q["flags"] else ""
        line2 = (f"label v1.3 (Pi3X | gated ViPE K, HFOV {L['info']['hfov']:.0f} deg, k={L['info']['k']:.2f})   "
                 f"QC v1c score {q['v1c']:.2f}{fl} -> {dec}   learned QC risk (Pi3X img-only) {q['risk_pi3x']:.2f}")
        if spec.get("reason"):
            line2 = (f"WHY: {spec['reason']}   |   v1.3 label, HFOV {L['info']['hfov']:.0f} deg, k={L['info']['k']:.2f};  "
                     f"QC v1c {q['v1c']:.2f}{fl} -> {dec}")
    line1 = f"{spec['name']}   |   {spec['cat']}"
    dmin, dmax = depth_range(L["D"])
    thr = float(np.percentile(L["conf"][::10], 25))
    dp = DepthPanel(L["D"], L["conf"], dmin, dmax, thr)
    up = up_vector(L["c2w"])
    tp = TrajPanel(L["c2w"], L["K"], W, up, gt_al, "v1.3 camera path")
    cloud = build_cloud(L, frames, mask)
    if spec.get("indoor"):
        cloud = ceiling_cut(cloud, L["c2w"], up)
    orbit = Orbit(cloud["P"], L["c2w"], up, nf, zoom=spec.get("zoom", 1.0 if spec.get("indoor") else 1.4))
    frus = 0.03 * orbit.dist
    meta.update(depth_range=[dmin, dmax], conf_thr=thr, n_points=int(len(cloud["P"])), dcap=cloud["dcap"])
    enc = encoder(out)
    poster_i = int(nf * 0.7)
    for i in range(nf):
        rgb = frames(i)
        p1, _ = fit(rgb, PW, PH)
        p1t = [(8, 4, f"RGB  frame {i + 1}/{n}", 14, (255, 255, 255), False, (0, 0, 0))]
        p2 = dp.panel(L["D"][i], L["conf"][i], "Depth (inverse-depth, turbo); dark = low Pi3X conf")
        p3 = tp.panel(i)
        cnt = int(np.searchsorted(cloud["F"], i, side="right"))
        p4 = cloud_panel(cloud, cnt, orbit, i, L["c2w"], L["K"], W, H,
                         f"Fused point cloud (every 5th frame, {cnt / 1000:.0f}k pts)", frus)
        fr = compose([(p1, p1t), p2, p3, p4], title_texts(line1, line2, spec.get("failure", False)), spec.get("failure", False))
        enc.stdin.write(fr.tobytes())
        if i == poster_i:
            cv2.imwrite(str(out.with_suffix(".jpg")), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    enc.stdin.close()
    enc.wait()


def render_compare(spec, c, H, W, n, nf, frames, mask, out, meta):
    kind, clip = spec["kind"], spec["clip"]
    Ls = [load_label(kind, c, H, W, w) for w in spec["compare"]]
    if spec["frame"] == "gt":
        G = gt_c2w(kind, c)
        Ls2 = []
        for L in Ls:
            s, R, t = umeyama(L["c2w"][:, :3, 3], G[:, :3, 3], True)
            Ls2.append(transform(L, s, R, t))
        Ls = Ls2
    else:  # align right label onto the left one
        s, R, t = umeyama(Ls[1]["c2w"][:, :3, 3], Ls[0]["c2w"][:, :3, 3], True)
        Ls[1] = transform(Ls[1], s, R, t)
        meta["align_scale_right_to_left"] = float(s)
    names = []
    for w, L in zip(spec["compare"], Ls):
        if w == "vipe":
            names.append(f"ViPE (HFOV {L['info']['hfov']:.0f} deg)")
        elif w == "v1.3":
            names.append(f"v1.3 (HFOV {L['info']['hfov']:.0f} deg)")
        else:
            names.append(f"v1.3, K x {L['info']['focal_scale']:.2f} (HFOV {L['info']['hfov']:.0f} deg)")
    meta["labels"] = [dict(L["info"]) for L in Ls]
    if kind in ("A", "B", "scannetpp"):
        ms = [gt_metrics(kind, clip, "vipe" if w == "vipe" else "v1.3") for w in spec["compare"]]
        meta["metrics"] = ms
        line2 = "   vs   ".join(f"{nm.split(' (')[0]}: ATE {m['ate_pct']:.2f}%  AbsRel {m['absrel']:.3f}  F@5cm {m['f5']:.2f}"
                                for nm, m in zip(names, ms))
        line2 += "   (both Sim3-aligned to GT; no conf filter)"
    else:
        q = qc_metrics(clip)
        meta["qc"] = q
        line2 = (f"left = shipped v1.3 label (ViPE K passes the [25,130] deg gate); right = Pi3X re-solved with K x 0.5 "
                 f"(E13 focal sweep optimum ~61 deg); QC v1c {q['v1c']:.2f} -> REJECT")
    line1 = f"{spec['name']}   |   {spec['cat']}"
    d0 = depth_range(Ls[0]["D"])
    d1 = depth_range(Ls[1]["D"])
    dmin, dmax = min(d0[0], d1[0]), max(d0[1], d1[1])
    dps = [DepthPanel(L["D"], None, dmin, dmax) for L in Ls]
    up = up_vector(Ls[0]["c2w"])
    sm = Ls[0]["D"][::10]
    sm = sm[np.isfinite(sm) & (sm > 0)]
    dcap = float(min(np.percentile(sm, 97), 6 * np.median(sm)))
    clouds = [build_cloud(L, frames, mask, use_conf=False, dcap=dcap) for L in Ls]
    if spec.get("indoor"):
        clouds = [ceiling_cut(cl, L["c2w"], up) for cl, L in zip(clouds, Ls)]
    zoom = spec.get("zoom", 1.0 if spec.get("indoor") else 1.4)
    orbit = Orbit(clouds[1]["P"], Ls[1]["c2w"], up, nf, el=spec.get("el", 50.0), zoom=zoom)
    frus = 0.03 * orbit.dist
    meta.update(depth_range=[dmin, dmax], n_points=[int(len(cl["P"])) for cl in clouds])
    enc = encoder(out)
    poster_i = int(nf * 0.85)
    for i in range(nf):
        rgb = frames(i)
        panels = []
        for j, (L, dp) in enumerate(zip(Ls, dps)):
            img, tx = dp.panel(L["D"][i], None, f"Depth - {names[j]}")
            if j == 0:  # small RGB inset
                th = cv2.resize(rgb, (160, int(160 * H / W)), interpolation=cv2.INTER_AREA)
                img[PH - th.shape[0] - 6:PH - 6, 6:6 + 160] = th
            panels.append((img, tx))
        for j, (L, cl) in enumerate(zip(Ls, clouds)):
            cnt = int(np.searchsorted(cl["F"], i, side="right"))
            panels.append(cloud_panel(cl, cnt, orbit, i, L["c2w"], L["K"], W, H, f"Point cloud - {names[j]}", frus))
        fr = compose(panels, title_texts(line1, line2))
        enc.stdin.write(fr.tobytes())
        if i == poster_i:
            cv2.imwrite(str(out.with_suffix(".jpg")), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    enc.stdin.close()
    enc.wait()


def safe_run(spec):
    try:
        return run(spec)
    except Exception:  # noqa: BLE001
        return spec["id"], "ERROR", traceback.format_exc()[-1500:]


if __name__ == "__main__":
    workers = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--workers=")), 5))
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    specs = [s for s in SPECS if only is None or s["id"] in only]
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, 6, len(specs))) as ex:
        for r in ex.map(safe_run, specs):
            print(*r, flush=True)
    print(f"TOTAL {time.time() - t0:.0f} s", flush=True)
