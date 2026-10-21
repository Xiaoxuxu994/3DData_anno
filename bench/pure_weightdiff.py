"""Which base checkpoint is each PURE model fine-tuned from, and which modules changed?

Frozen tensors survive post-training bit-identical, so the base whose shared tensors match exactly is
the starting point. Prints per-candidate identical fraction and the changed-module summary.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

OUT = Path("/data/home/z00533370/outputs/bench/pure_logs/weightdiff.json")


RELAY = Path("/data/home/z00533370/cache/relay")


def load_any(repo, sub=""):
    root = (RELAY / repo if (RELAY / repo).exists() else Path(snapshot_download(repo, local_files_only=True))) / sub
    fs = sorted(root.glob("model*.safetensors"))
    if fs:
        sd = {}
        for f in fs:
            sd.update(load_file(str(f)))
    else:
        sd = torch.load(root / "model.pt", map_location="cpu")
    return {re.sub(r"^model\.", "", k): v for k, v in sd.items()}


def compare(pure, base):
    shared = [k for k in pure if k in base and pure[k].shape == base[k].shape]
    same = [k for k in shared if torch.equal(pure[k].float(), base[k].float())]
    changed = [k for k in shared if k not in same]
    only_pure = [k for k in pure if k not in base or pure[k].shape != base[k].shape]
    rel = {}
    for k in changed:
        a, b = pure[k].float(), base[k].float()
        rel[k] = float((a - b).norm() / (b.norm() + 1e-12))
    group = defaultdict(lambda: [0, 0, 0.0])
    for k in shared:
        g = ".".join(k.split(".")[:3])
        group[g][0] += 1
        if k in rel:
            group[g][1] += 1
            group[g][2] = max(group[g][2], rel[k])
    changed_groups = {g: v for g, v in group.items() if v[1]}
    return dict(n_pure=len(pure), n_shared=len(shared), n_identical=len(same), n_changed=len(changed),
                n_only_pure=len(only_pure), only_pure_prefixes=sorted({".".join(k.split(".")[:2]) for k in only_pure}),
                changed_groups=changed_groups,
                median_rel_change=float(torch.tensor(list(rel.values())).median()) if rel else 0.0)


res = {}
for sub, cands in [("pi3", ["yyfz233/Pi3", "yyfz233/Pi3X"]), ("vggt", ["facebook/VGGT-1B"]),
                   ("da3", ["depth-anything/DA3-GIANT"])]:
    pure = load_any("cyun9286/PURE", sub)
    for c in cands:
        try:
            r = compare(pure, load_any(c))
        except Exception as e:  # noqa: BLE001
            r = {"error": repr(e)[:300]}
        res[f"{sub}<-{c}"] = r
        print(sub, c, {k: v for k, v in r.items() if k != "changed_groups"}, flush=True)
        if "changed_groups" in r:
            for g, (n, nc, mx) in sorted(r["changed_groups"].items())[:40]:
                print(f"    {g}: {nc}/{n} tensors changed, max rel {mx:.2e}")
OUT.write_text(json.dumps(res, indent=1))
