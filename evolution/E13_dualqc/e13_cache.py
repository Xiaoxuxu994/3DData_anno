"""E13: method-independent QC observations (RAFT fw/bw gap 1/5, SuperPoint+LightGlue gap 5/20) for ScanNet++,
identical to code/qc/qc_cache.py but written to outputs/evolution/E13/cache/S (line-G cache is left untouched).
Also logs per-clip GPU seconds (for the gate-2 cost estimate).

  CUDA_VISIBLE_DEVICES=7 cache/qc-venv/bin/python e13_cache.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/home/z00533370/code/qc")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import qc_common as QC  # noqa: E402

QC.BENCH["S"] = Path("/data/home/z00533370/outputs/bench/scannetpp")
OUT = Path("/data/home/z00533370/outputs/evolution/E13/cache/S")
OUT.mkdir(parents=True, exist_ok=True)
sys.argv = ["qc_cache.py"]  # import without running its group loop
import qc_cache as QCC  # noqa: E402

for c in QC.clips("S"):
    clip, n = c["clip"], c["n_frames"]
    out = OUT / f"{clip}.npz"
    if out.exists():
        continue
    torch.cuda.synchronize()
    t0 = time.time()
    hw = QC.qc_grid("S", clip)
    need = set()
    for g in QC.FLOW_GAPS + QC.MATCH_GAPS:
        for i in QC.anchors(n, g):
            need |= {int(i), int(i + g)}
    frames = {k: QC.rgb("S", clip, k, hw) for k in need}
    t1 = time.time()
    res = {"hw": np.array(hw)}
    for g in QC.FLOW_GAPS:
        a = QC.anchors(n, g)
        res[f"flow{g}_i"] = a
        res[f"flow{g}_fw"] = QCC.flow(frames, [(int(i), int(i + g)) for i in a])
        res[f"flow{g}_bw"] = QCC.flow(frames, [(int(i + g), int(i)) for i in a])
    torch.cuda.synchronize()
    t2 = time.time()
    feats = {}
    for g in QC.MATCH_GAPS:
        a = QC.anchors(n, g)
        k0, k1 = QCC.match(frames, [(int(i), int(i + g)) for i in a], feats)
        res[f"match{g}_i"] = a
        res[f"match{g}_cnt"] = np.array([len(x) for x in k0])
        res[f"match{g}_k0"] = np.concatenate(k0).astype(np.float32)
        res[f"match{g}_k1"] = np.concatenate(k1).astype(np.float32)
    torch.cuda.synchronize()
    t3 = time.time()
    np.savez(out, **res)
    print(json.dumps({"clip": clip, "hw": list(hw), "n_frames_read": len(need), "read_s": round(t1 - t0, 2),
                      "raft_s": round(t2 - t1, 2), "lightglue_s": round(t3 - t2, 2), "total_s": round(time.time() - t0, 2)}), flush=True)
print("CACHE_DONE", flush=True)
