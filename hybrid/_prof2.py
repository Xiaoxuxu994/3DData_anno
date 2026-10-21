import sys, time, json
sys.argv = ["x", "A"]
sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import run_hybrid as R
import hybrid as HY
c = [c for c in json.loads((R.BENCH / "clips.json").read_text()) if c["clip"] == "sintel__alley_1"][0]
GT = R.gt_eval3d(c); H, W = GT[2].shape[1:]
print(H, W, GT[3], GT[4])
cache = {}
def base(m):
    if m not in cache:
        t = time.time(); cache[m] = R.ORIG(R.BENCH, m, c["clip"], c["n_frames"], H, W); print("load", m, round(time.time() - t, 1), flush=True)
    return cache[m]
R.M.load_method = lambda B, m, cl, n, H_, W_: HY.compose(m, base)[:4]
for s in ["vipe", "H:vipe+pi3x+geo"]:
    t = time.time(); r = R.M.eval_clip((R.BENCH, s, c)); print(s, "eval", round(time.time() - t, 1), r.get("error"), flush=True)
