"""Re-check cached GT (after crop/resize/7-Scenes warp): reprojection consistency i->i+gap."""
import json, sys
from pathlib import Path
import numpy as np
B = Path("/data/home/z00533370/outputs/bench")
def check(npz, gap):
    z = np.load(npz, allow_pickle=True); D = z["depth"].astype(np.float32); K = z["K"]; T = z["c2w"]
    H, W = D.shape[1:]; i, j = len(D)//2 - gap//2, len(D)//2 - gap//2 + gap
    v, u = np.mgrid[0:H, 0:W]; m = D[i] > 0
    X = np.stack([(u[m]-K[i,0,2])/K[i,0,0]*D[i][m], (v[m]-K[i,1,2])/K[i,1,1]*D[i][m], D[i][m]], 1)
    Tij = np.linalg.inv(T[j]) @ T[i]; Xc = X @ Tij[:3,:3].T + Tij[:3,3]; z_ = Xc[:,2]
    uu = np.round(K[j,0,0]*Xc[:,0]/z_ + K[j,0,2]).astype(int); vv = np.round(K[j,1,1]*Xc[:,1]/z_ + K[j,1,2]).astype(int)
    ok = (z_>0.05)&(uu>=0)&(uu<W)&(vv>=0)&(vv<H); dj = D[j][vv[ok], uu[ok]]; g = dj > 0
    rel = np.abs(z_[ok][g]-dj[g])/dj[g]
    return round(float(np.median(rel)),4), round(float((rel<0.05).mean()),3), int(g.sum())
for b in ("eval3d_A", "eval3d_B"):
    seen = set()
    for f in sorted((B/b/"gt").glob("*.npz")):
        ds = f.stem.split("__")[0]
        if ds in seen: continue
        seen.add(ds)
        gap = 1 if ds == "eth3d_mvs" else 5
        print(b, f.stem, "gap", gap, check(f, gap))
