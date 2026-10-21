"""Run one unchanged code/filter stage script on a batch pool, with its per-clip logs redirected to the batch.

Usage: python filter_stage.py STAGE_SCRIPT [ARGS ...]      (env: FILTER_POOL = batch pool dir, ANNO_LOGS = log dir)
The filter scripts write their runtime logs to fcommon.LOGS (= ROOT/outputs/logs/filter, shared with the original
pool run); this wrapper points fcommon.LOGS at ANNO_LOGS before executing the stage, so a pipeline run never appends
to or overwrites the experiment logs. Everything else (thresholds, models, scores.json) is the stage's own code.
"""

import os
import runpy
import sys
from pathlib import Path

FILTER = Path(os.environ.get("ANNO_ROOT", "/data/home/z00533370")) / "code/filter"
sys.path.insert(0, str(FILTER))
import fcommon  # noqa: E402

fcommon.LOGS = Path(os.environ["ANNO_LOGS"])
fcommon.LOGS.mkdir(parents=True, exist_ok=True)
script = str(FILTER / sys.argv[1])
sys.argv = [script] + sys.argv[2:]
if __name__ == "__main__":
    runpy.run_path(script, run_name="__main__")
