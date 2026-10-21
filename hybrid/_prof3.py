import sys, time, json, faulthandler
faulthandler.dump_traceback_later(400, exit=True)
sys.argv = ["x", "A"]
sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import run_hybrid as R
import hybrid as HY
c = [c for c in json.loads((R.BENCH / "clips.json").read_text()) if c["clip"] == "sintel__alley_1"][0]
GT = R.gt_eval3d(c); H, W = GT[2].shape[1:]
cache = {}
def base(m):
    if m not in cache:
        cache[m] = R.ORIG(R.BENCH, m, c["clip"], c["n_frames"], H, W)
    return cache[m]
R.M.load_method = lambda B, m, cl, n, H_, W_: HY.compose(m, base)[:4]
for s in R.SPECS:
    t = time.time(); r = R.M.eval_clip((R.BENCH, s, c)); print(s, round(time.time() - t, 1), r.get("error"), r.get("pc_fscore_5cm"), flush=True)
