import glob, numpy as np, os
from PIL import Image
R = "/data/home/z00533370/data/scannetpp_sample30/meta_preview"
fs = sorted(glob.glob(f"{R}/*/scene_metadata.npz"))
print(len(fs))
z = np.load(fs[0], allow_pickle=True)
for k in z.files:
    print(k, z[k].shape, z[k].dtype)
print(z["images"][:4], z["images"][-4:])
print("K0\n", z["intrinsics"][0]); print("T0\n", z["trajectories"][0])
im = Image.open(f"{R}/00dd871005/images/8a31bb0c_DSC09626.jpg"); d = np.array(Image.open(f"{R}/00dd871005/depth/8a31bb0c_DSC09626.png"))
print("img", im.size, "depth", d.shape, d.dtype, d.min(), np.percentile(d[d>0], [2, 50, 98]), (d == 0).mean())
for f in fs:
    z = np.load(f, allow_pickle=True); names = list(z["images"])
    pre = sorted({n.split("_")[0] for n in names})
    K = z["intrinsics"]; T = z["trajectories"]
    # order by DSC number as in dataloader; check jumps in camera position
    order = sorted(range(len(names)), key=lambda i: int(names[i][-5:]))
    p = T[order, :3, 3]; step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    j = np.argsort(step)[-2:]
    print(os.path.basename(os.path.dirname(f)), len(names), "prefixes", {q: sum(n.startswith(q) for n in names) for q in pre},
          "fx %.1f-%.1f" % (K[:, 0, 0].min(), K[:, 0, 0].max()), "cx %.1f" % K[0, 0, 2], "cy %.1f" % K[0, 1, 2],
          "step med %.3f max %.3f @%s" % (np.median(step), step.max(), [names[order[i]] + "->" + names[order[i + 1]] for i in j[::-1][:1]]))
