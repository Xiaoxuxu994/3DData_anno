"""Minimal convention check for a pred.npz: adjacent-frame reprojection, median |z'-d|/d (as-is vs inverted pose)."""
import sys
import numpy as np
import cv2

z = np.load(sys.argv[1])
c2w, K, D = z["c2w"], z["K"], z["depth"]
H, W = D.shape[1:]
v, u = np.mgrid[0:H, 0:W]


def err(T, k, j):
    d = D[k]
    m = d > 0
    X = np.stack([(u[m] - K[k, 0, 2]) / K[k, 0, 0] * d[m], (v[m] - K[k, 1, 2]) / K[k, 1, 1] * d[m], d[m]], 1)
    Tkj = np.linalg.inv(T[j]) @ T[k]
    Xc = X @ Tkj[:3, :3].T + Tkj[:3, 3]
    zj = Xc[:, 2]
    uu = K[j, 0, 0] * Xc[:, 0] / zj + K[j, 0, 2]
    vv = K[j, 1, 1] * Xc[:, 1] / zj + K[j, 1, 2]
    ok = (zj > 1e-4) & (uu >= 0) & (uu <= W - 1) & (vv >= 0) & (vv <= H - 1)
    dj = D[j][np.round(vv[ok]).astype(int), np.round(uu[ok]).astype(int)]
    g = dj > 0
    return float(np.median(np.abs(zj[ok][g] - dj[g]) / dj[g]))


n = len(D)
ks = range(0, n - 1, max(1, n // 10))
inv = np.linalg.inv(c2w)
print(sys.argv[1].split("/")[-3], sys.argv[1].split("/")[-2], "as-is %.4f  inverted %.4f  depth %s K00 %.1f" % (
    np.median([err(c2w, k, k + 1) for k in ks]), np.median([err(inv, k, k + 1) for k in ks]), D.shape, K[0, 0, 0]))
