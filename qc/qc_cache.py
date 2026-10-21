"""Method-independent observations per clip (computed once, shared by all annotators' QC):
RAFT-large forward/backward flow for FLOW_GAPS and SuperPoint+LightGlue matches for MATCH_GAPS, on the QC grid.

  CUDA_VISIBLE_DEVICES=1 python qc_cache.py GROUP [GROUP ...]   -> outputs/qc/cache/<G>/<clip>.npz
"""
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from lightglue import LightGlue, SuperPoint
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from qc_common import CACHE, FLOW_GAPS, MATCH_GAPS, anchors, clips, qc_grid, rgb

raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).cuda().eval()
sp = SuperPoint(max_num_keypoints=2048).cuda().eval()
lg = LightGlue(features="superpoint").cuda().eval()


def to_t(im):
    return TF.normalize(torch.from_numpy(im).permute(2, 0, 1).float().div(255), [0.5] * 3, [0.5] * 3)


@torch.inference_mode()
def flow(frames, pairs, bs=12):
    out = []
    for s in range(0, len(pairs), bs):
        p = pairs[s:s + bs]
        a = torch.stack([to_t(frames[i]) for i, _ in p]).cuda()
        b = torch.stack([to_t(frames[j]) for _, j in p]).cuda()
        out.append(raft(a, b)[-1].half().cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def match(frames, pairs, feats):
    k0, k1 = [], []
    for i, j in pairs:
        for k in (i, j):
            if k not in feats:
                feats[k] = sp.extract(torch.from_numpy(frames[k]).permute(2, 0, 1).float().div(255).cuda())
        r = lg({"image0": feats[i], "image1": feats[j]})
        m = r["matches"][0]
        k0.append(feats[i]["keypoints"][0][m[:, 0]].cpu().numpy())
        k1.append(feats[j]["keypoints"][0][m[:, 1]].cpu().numpy())
    return k0, k1


for G in sys.argv[1:]:
    (CACHE / G).mkdir(parents=True, exist_ok=True)
    for c in clips(G):
        clip, n = c["clip"], c["n_frames"]
        out = CACHE / G / f"{clip}.npz"
        if out.exists():
            continue
        t0 = time.time()
        hw = qc_grid(G, clip)
        frames = {}
        need = set()
        for g in FLOW_GAPS + MATCH_GAPS:
            for i in anchors(n, g):
                need |= {int(i), int(i + g)}
        for k in need:
            frames[k] = rgb(G, clip, k, hw)
        res = {"hw": np.array(hw)}
        for g in FLOW_GAPS:
            a = anchors(n, g)
            if len(a):
                res[f"flow{g}_i"] = a
                res[f"flow{g}_fw"] = flow(frames, [(int(i), int(i + g)) for i in a])
                res[f"flow{g}_bw"] = flow(frames, [(int(i + g), int(i)) for i in a])
        feats = {}
        for g in MATCH_GAPS:
            a = anchors(n, g)
            if len(a):
                k0, k1 = match(frames, [(int(i), int(i + g)) for i in a], feats)
                res[f"match{g}_i"] = a
                res[f"match{g}_cnt"] = np.array([len(x) for x in k0])
                res[f"match{g}_k0"] = np.concatenate(k0).astype(np.float32) if k0 else np.zeros((0, 2), np.float32)
                res[f"match{g}_k1"] = np.concatenate(k1).astype(np.float32) if k1 else np.zeros((0, 2), np.float32)
        np.savez(out, **res)
        print(G, clip, hw, f"{time.time() - t0:.1f}s", flush=True)
print("CACHE_DONE", flush=True)
