"""Run one UNCHANGED code/filter stage script on a CPU-only machine (smoke tests, laptops, CI) - not for production runs.

Usage: python filter_stage_cpu.py STAGE_SCRIPT [ARGS ...]        (env: ANNO_ROOT, FILTER_POOL = pool dir, ANNO_LOGS = log dir)
Same contract as pipeline/filter_stage.py (stage scripts of code/filter are executed as they are, logs redirected to ANNO_LOGS), plus:
  * fcommon.H (hard-coded /data/home/z00533370 in the experiment scripts) is pointed at ANNO_ROOT, and fcommon.POOL at FILTER_POOL
    (default ANNO_ROOT/data/filter_pool),
  * torch is patched so the stages' `.cuda()` / device="cuda" calls run on the CPU: Tensor.cuda / Module.cuda are no-ops, `.to("cuda")`
    maps to "cpu", torch.cuda.synchronize / empty_cache / memory counters become no-ops. (torch.zeros & co are left alone on purpose:
    GeoCalib's TorchScript functions resolve them by name and cannot compile a wrapper.)
The numbers it produces are the stage's own (same models, thresholds and preprocessing); only the device differs, so small float
differences against GPU runs are expected (RAFT / GeoCalib / CLIP in fp32 on CPU). Stage vlm_run.py is NOT supported (needs vLLM + a 32B model).
"""

import os
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("ANNO_ROOT", "/data/home/z00533370"))
for _p in reversed([x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]):  # honoured even under `python -I`
    sys.path.insert(0, _p)


def _dev(d):
    if isinstance(d, str) and d.startswith("cuda"):
        return "cpu"
    if isinstance(d, torch.device) and d.type == "cuda":
        return torch.device("cpu")
    return d


def _wrap_to(orig):
    def to(self, *a, **k):
        return orig(self, *tuple(_dev(x) for x in a), **{kk: _dev(v) for kk, v in k.items()})
    return to


torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.Tensor.to = _wrap_to(torch.Tensor.to)
torch.nn.Module.to = _wrap_to(torch.nn.Module.to)
for _name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
    setattr(torch.cuda, _name, lambda *a, **k: None)
for _name in ("max_memory_allocated", "max_memory_reserved", "memory_allocated"):
    setattr(torch.cuda, _name, lambda *a, **k: 0)

FILTER = ROOT / "code" / "filter"
sys.path.insert(0, str(FILTER))
import fcommon  # noqa: E402

fcommon.H = ROOT
fcommon.LOGS = Path(os.environ.get("ANNO_LOGS") or ROOT / "outputs" / "logs" / "filter")
fcommon.LOGS.mkdir(parents=True, exist_ok=True)
# fcommon computed POOL at import from its hard-coded root; follow ANNO_ROOT unless FILTER_POOL says otherwise (a stage ADDS scores to
# the pool it is pointed at, so it must never silently be the shared pool of another root)
fcommon.POOL = Path(os.environ["FILTER_POOL"]) if os.environ.get("FILTER_POOL") else ROOT / "data" / "filter_pool"
script = str(FILTER / sys.argv[1])
sys.argv = [script] + sys.argv[2:]
if __name__ == "__main__":
    runpy.run_path(script, run_name="__main__")
