import json, numpy as np, sys
m = json.load(open(sys.argv[1]))
P = np.array(m["map_xyz"]); C = np.array(m["map_rgb"]); T = np.array(m["traj_xyz"]); F = np.array(m["traj_fwd"])
# ground plane via PCA of map points: smallest-variance axis = up
mu = P.mean(0); _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)
n = Vt[2]
if np.dot(n, mu - T.mean(0)) > 0: n = -n  # normal points from ground toward cameras
a = Vt[0] - np.dot(Vt[0], n) * n; a /= np.linalg.norm(a); b = np.cross(n, a)
proj = lambda X: np.stack([(X - mu) @ a, (X - mu) @ b], -1)
p2, t2 = proj(P), proj(T)
f2 = np.stack([F @ a, F @ b], -1); f2 /= np.linalg.norm(f2, axis=1, keepdims=True) + 1e-9
h = (T - mu) @ n
out = {k: m[k] for k in ["motion", "keyframes", "n_map_points", "traj_len_m", "pose0", "pose60", "slam_hw"]}
out.update(pts=p2.round(3).tolist(), col=["#%02x%02x%02x" % tuple(c) for c in C], traj=t2.round(3).tolist(),
           fwd=f2.round(3).tolist(), cam_height_m=[round(float(h.min()), 2), round(float(h.max()), 2)])
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
print(len(json.dumps(out)), out["cam_height_m"], t2.min(0), t2.max(0), p2.min(0), p2.max(0))
