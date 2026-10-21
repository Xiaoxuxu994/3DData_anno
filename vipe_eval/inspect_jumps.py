import glob, numpy as np, os
R = "/data/home/z00533370/data/scannetpp_sample30/meta_preview"
def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
tot = {"frames": 0, "segs": 0}
for f in sorted(glob.glob(f"{R}/*/scene_metadata.npz")):
    z = np.load(f); names = list(z["images"]); T = z["trajectories"]
    o = sorted(range(len(names)), key=lambda i: (names[i][:-8] != "", int(names[i][-5:])))  # placeholder order
    o = sorted(range(len(names)), key=lambda i: int(names[i][-5:]))
    num = np.array([int(names[i][-5:]) for i in o])
    t = np.linalg.norm(np.diff(T[o, :3, 3], axis=0), axis=1)
    r = np.array([rot_deg(T[o[i], :3, :3], T[o[i + 1], :3, :3]) for i in range(len(o) - 1)])
    gap = np.diff(num)
    brk = (t > 0.5) | (r > 30)
    seg_len = np.diff(np.r_[0, np.where(brk)[0] + 1, len(o)])
    tot["frames"] += len(o)
    print(f"{os.path.basename(os.path.dirname(f))} n={len(o):5d} t_med={np.median(t):.3f} r_med={np.median(r):5.2f}deg "
          f"numgap>1:{(gap>1).sum():4d} breaks:{brk.sum():3d} segs>=48:{(seg_len>=48).sum():2d} cover={seg_len[seg_len>=48].sum()/len(o):.2f} longest={seg_len.max()}")
