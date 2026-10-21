"""E3 evaluation: BASE (default pi3x_kgate) with a voted metric scale, using the run_hybrid machinery.

  python e3_eval.py A|B|scannetpp|wild -> outputs/evolution/E3/raw/<kind>.csv
Pseudo-method "<BASE>@<vote>" = BASE cameras/depth with translation and depth multiplied by k_<vote> from
outputs/evolution/E3/scales_<kind>.csv (rotation and intrinsics unchanged). Reference rows: vipe, BASE, v0 hybrid.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import run_hybrid as RH  # noqa: E402

KIND = RH.KIND
E3 = Path("/data/home/z00533370/outputs/evolution/E3")
SC = pd.read_csv(E3 / f"scales_{KIND}.csv").set_index("clip")
BASE = str(SC["base"].iloc[0])
KCOLS = [c[2:] for c in SC.columns if c.startswith("k_")]
if os.environ.get("E3_KCOLS"):
    KCOLS = os.environ["E3_KCOLS"].split(",")
POS = (1, 2) if KIND in ("A", "B", "wild") else (0, 1)
_orig = RH.ORIG


def orig(*a):
    m, clip = a[POS[0]], a[POS[1]]
    if "@" not in m:
        return _orig(*a)
    base, kc = m.split("@")
    b = list(a)
    b[POS[0]] = base
    c2w, K, D, sec = _orig(*b)
    k = float(SC.loc[clip, f"k_{kc}"]) if clip in SC.index else 1.0
    k = k if np.isfinite(k) and k > 0 else 1.0
    c2w = c2w.copy()
    c2w[:, :3, 3] *= k
    return c2w, K, D * k, sec


RH.ORIG = orig
RH.SPECS = ["vipe", "H:vipe+pi3x+med", BASE] + [f"{BASE}@{k}" for k in KCOLS if k != "own"]
RH.RES = Path(os.environ["E3_RES"]) if os.environ.get("E3_RES") else E3 / "raw"

if __name__ == "__main__":
    RH.main()
