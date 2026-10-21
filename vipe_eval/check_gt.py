"""Sanity-check ScanNet++ GT conventions: depth(mm, z) + K + cam2world via cross-frame reprojection."""
import numpy as np, cv2
R = "/data/home/z00533370/data/scannetpp_sample30"
def load(scene, name):
    d = cv2.imread(f"{R}/{scene}/depth/{name}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
    return d
def reproj(scene, a, b, z, idx, inv=False):
    Ta, Tb = z["trajectories"][idx[a]], z["trajectories"][idx[b]]
    if inv: Ta, Tb = np.linalg.inv(Ta), np.linalg.inv(Tb)
    K = z["intrinsics"][idx[a]]
    da, db = load(scene, a), load(scene, b)
    H, W = da.shape
    v, u = np.mgrid[0:H, 0:W]
    m = da > 0
    X = np.stack([(u[m] - K[0, 2]) / K[0, 0] * da[m], (v[m] - K[1, 2]) / K[1, 1] * da[m], da[m]], 0)
    Xw = Ta[:3, :3] @ X + Ta[:3, 3:]
    Tb_inv = np.linalg.inv(Tb)
    Xb = Tb_inv[:3, :3] @ Xw + Tb_inv[:3, 3:]
    zb = Xb[2]; ok = zb > 0.05
    ub = np.round(K[0, 0] * Xb[0][ok] / zb[ok] + K[0, 2]).astype(int); vb = np.round(K[1, 1] * Xb[1][ok] / zb[ok] + K[1, 2]).astype(int)
    inb = (ub >= 0) & (ub < W) & (vb >= 0) & (vb < H)
    zz = zb[ok][inb]; dref = db[vb[inb], ub[inb]]; g = dref > 0
    rel = np.abs(zz[g] - dref[g]) / dref[g]
    return inb.mean(), np.median(rel), (rel < 0.05).mean()
for scene in ["00dd871005", "2a1a3afad9", "d986399f4c"]:
    z = np.load(f"{R}/{scene}/scene_metadata.npz")
    names = list(z["images"]); idx = {n: i for i, n in enumerate(names)}
    order = sorted(names, key=lambda s: int(s[-5:]))
    k = len(order) // 2
    for gap in (1, 5):
        a, b = order[k], order[k + gap]
        for inv in (False, True):
            r = reproj(scene, a, b, z, idx, inv)
            print(f"{scene} {a}->{b} {'world2cam?' if inv else 'cam2world'} in-view {r[0]:.2f} median relerr {r[1]:.3f} inlier(<5%) {r[2]:.2f}")
