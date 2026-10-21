"""wild_eval.py (no-GT C group) for the PURE comparison, writing to wild/eval_pure/ instead of wild/eval/.

  python pure_wild_eval.py METHOD [METHOD ...]
Patches (in memory only): output dir eval -> eval_pure; load_method from pure_eval (vggt_omega512 crop).
"""

import sys
import types

sys.path.insert(0, "/data/home/z00533370/code/bench")
import pure_eval  # noqa: E402  (registers the patched eval3d module)

SRC = "/data/home/z00533370/code/bench/wild_eval.py"
src = open(SRC).read()
for old, new in [('B / "eval"', 'B / "eval_pure"'),
                 ("from eval3d_bench import load_method", "from eval3d_bench_pure import load_method")]:
    assert src.count(old) >= 1, f"wild_eval.py changed: {old!r} not found"
    src = src.replace(old, new)
mod = types.ModuleType("wild_eval_pure")
mod.__file__ = SRC
sys.modules["wild_eval_pure"] = mod
exec(compile(src.replace('if __name__ == "__main__":', "if False:"), SRC, "exec"), mod.__dict__)

if __name__ == "__main__":
    mod.main()
