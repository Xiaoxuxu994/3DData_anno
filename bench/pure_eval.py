"""Run eval3d_bench.py metrics for the PURE comparison without editing the shared script.

  python pure_eval.py BENCH_DIR METHOD [METHOD ...] --tag=_pure
Only change: eval3d_bench applies VGGT-Omega's aspect-ratio center crop when method == "vggt_omega";
here it also applies to the 512 checkpoint run ("vggt_omega512").
"""

import sys
import types

SRC = "/data/home/z00533370/code/bench/eval3d_bench.py"
src = open(SRC).read()
old = 'if method == "vggt_omega":  # official preprocessing'
assert src.count(old) == 1, "eval3d_bench.py changed; re-check the vggt_omega crop branch"
src = src.replace(old, 'if method.startswith("vggt_omega"):  # official preprocessing')
mod = types.ModuleType("eval3d_bench_pure")
mod.__file__ = SRC
sys.modules["eval3d_bench_pure"] = mod  # needed so ProcessPoolExecutor workers can pickle eval_clip
exec(compile(src.replace('if __name__ == "__main__":', "if False:"), SRC, "exec"), mod.__dict__)

if __name__ == "__main__":
    mod.main()
