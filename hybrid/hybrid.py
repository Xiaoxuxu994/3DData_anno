"""Hybrid annotations: cameras (pose + intrinsics) from one annotator, depth from another or a per-pixel vote.

Spec "H:<pose>+<depth>+<align>"  (plain method names are passed through unchanged)
  pose  : vipe | spatialvid_metric | pi3x | da3
  depth : pi3x | da3 | vggt_omega | vipe | vote3 (pi3x, da3, vggt_omega) | vote4 (vote3 + vipe)
  align : raw  depth used as predicted (metric depth sources only)
          med  (a) one scale per clip = median(pose-source depth) / median(depth-source depth) on co-valid pixels
          geo  (b) one scale per clip that maximises multi-view depth consistency under the pose source's cameras
               (falls back to med when the clip has too little parallax to identify the scale)
Vote: each member is aligned to the pose source with the same rule, then per-pixel nanmedian;
      spread = per-pixel median |member - vote| / vote (relative MAD, lower = more confident).

All arrays live on the evaluation grid produced by the caller's loader: base(method) -> (c2w, K, D, sec).
"""

import numpy as np

VOTE = {"vote3": ["pi3x", "da3", "vggt_omega"], "vote4": ["pi3x", "da3", "vggt_omega", "vipe"]}


def parse(spec):
    p, d, a = spec[2:].split("+")
    return p, d, a


def med_scale(Dref, D):
    m = np.isfinite(Dref) & (Dref > 0) & np.isfinite(D) & (D > 0)
    if m.sum() < 1000:
        return np.nan
    return float(np.median(Dref[m]) / np.median(D[m]))


def geo_scale(c2w, K, D, s0, gaps=(5, 15), n_pairs=12, target=120):
    """Scale s such that s*D is most consistent across frames under (c2w, K); returns (s, info)."""
    n, H, W = D.shape
    st = max(1, int(round(max(H, W) / target)))
    v, u = np.mgrid[0:H:st, 0:W:st]
    pre, par = [], []
    for g in gaps:
        if n <= g:
            continue
        for i in np.unique(np.linspace(0, n - 1 - g, n_pairs).astype(int)):
            j = i + g
            z = D[i, ::st, ::st]
            m = np.isfinite(z) & (z > 0)
            if m.sum() < 300:
                continue
            X = np.stack([(u[m] - K[i, 0, 2]) / K[i, 0, 0] * z[m], (v[m] - K[i, 1, 2]) / K[i, 1, 1] * z[m], z[m]], 1)
            T = np.linalg.inv(c2w[j]) @ c2w[i]
            pre.append((X @ T[:3, :3].T, T[:3, 3], j))
            par.append(np.linalg.norm(T[:3, 3]) / (s0 * np.median(z[m])))
    info = {"geo_parallax": float(np.median(par)) if par else 0.0}
    if not pre or not np.isfinite(s0) or info["geo_parallax"] < 0.01:
        info["geo_fallback"] = 1.0
        return s0, info

    def score(s):
        fr = []
        for RX, t, j in pre:
            Xc = s * RX + t
            z = Xc[:, 2]
            ok = z > 1e-3
            uu = K[j, 0, 0] * Xc[:, 0] / np.where(ok, z, 1) + K[j, 0, 2]
            vv = K[j, 1, 1] * Xc[:, 1] / np.where(ok, z, 1) + K[j, 1, 2]
            ok &= (uu >= 0) & (uu <= W - 1) & (vv >= 0) & (vv <= H - 1)
            if ok.sum() < 100:
                fr.append(0.0)
                continue
            dj = s * D[j][np.round(vv[ok]).astype(int), np.round(uu[ok]).astype(int)]
            g = np.isfinite(dj) & (dj > 0)
            fr.append(float(np.sum(np.abs(z[ok][g] - dj[g]) / dj[g] < 0.05)) / len(RX))
        return float(np.mean(fr))

    ls = np.log(s0) + np.linspace(-1.6, 1.6, 49)
    sc = np.array([score(np.exp(x)) for x in ls])
    k = int(np.argmax(sc))
    fine = ls[k] + np.linspace(-0.07, 0.07, 15)
    sf = np.array([score(np.exp(x)) for x in fine])
    best = float(np.exp(fine[int(np.argmax(sf))]))
    info.update({"geo_score": float(sf.max()), "geo_score_at_med": float(score(s0)),
                 "geo_peak_contrast": float(sf.max() - max(sc[0], sc[-1]))})
    if info["geo_peak_contrast"] < 0.01 or k in (0, len(ls) - 1):  # flat or edge-of-range optimum -> not identifiable
        info["geo_fallback"] = 1.0
        return s0, info
    info["geo_fallback"] = 0.0
    return best, info


def nanmed_small(S):
    """nanmedian over a short axis 0 (vote members) via sort; returns (median, n_valid)."""
    k = np.isfinite(S).sum(0)
    So = np.sort(S, axis=0)  # NaN last
    lo = np.clip((k - 1) // 2, 0, len(S) - 1)
    hi = np.clip(k // 2, 0, len(S) - 1)
    a = np.take_along_axis(So, lo[None], 0)[0]
    b = np.take_along_axis(So, hi[None], 0)[0]
    m = (a + b) / 2
    m[k == 0] = np.nan
    return m, k


def align(c2w, K, Dpose, D, how):
    if how == "raw":
        return D, 1.0, {}
    s0 = med_scale(Dpose, D)
    if how == "med":
        return D * s0, s0, {}
    s, info = geo_scale(c2w, K, D, s0)
    return D * s, s, info


def compose(spec, base, memo=None):
    """-> c2w, K, D, sec, info (info: scales, geo diagnostics, 'spread' map for votes).
    memo: optional dict caching (pose, member, align) -> (scale, info) within one clip."""
    if not spec.startswith("H:"):
        return (*base(spec), {})
    memo = {} if memo is None else memo
    p, d, a = parse(spec)
    c2w, K, Dp, sec = base(p)
    members = VOTE.get(d, [p if d == "own" else d])
    out, info = [], {}
    for m in members:
        Dm = Dp if m == p else base(m)[2]
        if (p, m, a) not in memo:
            Da, s, gi = align(c2w, K, Dp, Dm, a)
            memo[(p, m, a)] = (s, gi)
        s, gi = memo[(p, m, a)]
        Da = Dm * s if a != "raw" else Dm
        out.append(Da)
        info[f"scale_{m}"] = s
        for k_, v_ in gi.items():
            info[f"{k_}_{m}"] = v_
    if len(out) == 1:
        return c2w, K, out[0].astype(np.float32), sec, info
    S = np.stack(out)
    S = np.where(S > 0, S, np.nan)
    V, nval = nanmed_small(S)
    with np.errstate(all="ignore"):
        spread = nanmed_small(np.abs(S - V[None]))[0] / V
    spread[nval < 2] = np.inf
    info["spread"] = spread.astype(np.float32)
    return c2w, K, V.astype(np.float32), sec, info


def specs():
    S = []
    for p in ("vipe", "spatialvid_metric"):
        for d in ("pi3x", "da3"):
            S += [f"H:{p}+{d}+{a}" for a in ("raw", "med", "geo")]
        S += [f"H:{p}+vggt_omega+{a}" for a in ("med", "geo")]
        S += [f"H:{p}+{v}+{a}" for v in ("vote3", "vote4") for a in ("med", "geo")]
    for p, o in (("pi3x", "da3"), ("da3", "pi3x")):
        S += [f"H:{p}+{o}+raw", f"H:{p}+{o}+med", f"H:{p}+{o}+geo", f"H:{p}+vote3+med", f"H:{p}+vote3+geo"]
    S += ["H:vipe+own+geo"]  # vipe's own depth re-scaled by the multi-view criterion (does (b) help on its own?)
    return S


SINGLES = ["vipe", "spatialvid_metric", "pi3x", "da3", "vggt_omega"]
