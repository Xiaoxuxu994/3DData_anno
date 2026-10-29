"""demo_summary.py DEMO_OUT [OUT_JSON] - one row per clip of an E15 plumbing-demo run (evolution/E15_glr/run_demo.sh), for the summary page.

route (filter/glr_route.csv) -> worker status and stage-1 gate (glr_jobs/<clip>/result*/result.json, the one RESULT_DIR points at) -> tier (results_glr.csv)
-> the final decision (final/results.csv): which annotator's label is used, or rejected, and why. Missing pieces stay empty, nothing is invented.
Prints the table; writes the rows as json when OUT_JSON is given.
"""
import csv
import json
import sys
from pathlib import Path


def read(p):
    p = Path(p)
    if not p.exists():
        return {}
    with open(p, newline="") as fh:
        return {r["clip"]: r for r in csv.DictReader(fh)}


def worker_result(out, clip):
    job = Path(out) / "glr_jobs" / clip
    ptr = job / "RESULT_DIR"
    try:
        res = Path((ptr.read_text().strip() if ptr.exists() else "")) if ptr.exists() else None
        res = res if res and res.is_absolute() else (job / res if res else None)
        return json.loads((res / "result.json").read_text()) if res and (res / "result.json").exists() else {}
    except (OSError, ValueError):
        return {}


def summarize(out):
    out = Path(out)
    route, glr, final = read(out / "filter" / "glr_route.csv"), read(out / "results_glr.csv"), read(out / "final" / "results.csv")
    rows = []
    for clip, rt in route.items():
        w = worker_result(out, clip)
        g, f = glr.get(clip, {}), final.get(clip, {})
        fin = ""
        if f:
            fin = f"{f.get('annotator')} ({f.get('tier')})" if f.get("accept") in ("True", "true", "1") else "拒绝"
        try:
            q = json.loads((out / "qc_glr" / f"{clip}.json").read_text())
        except (OSError, ValueError):
            q = {}
        fmt = lambda k, nd: f"{q[k]:.{nd}f}" if isinstance(q.get(k), (int, float)) else ""  # noqa: E731
        rows.append({"clip": clip, "route": rt.get("route", "") + ((" (" + rt["reasons"] + ")") if rt.get("reasons") else ""), "gate": w.get("gate", ""),
                     "status": w.get("status", ""), "v1c": fmt("v1c", 2), "kdis": fmt("kdis", 1), "xdepth_k": fmt("xdepth_k", 2), "tier": g.get("tier", ""), "final": fin,
                     "why": (f.get("reason") or g.get("reasons") or "")[:160]})
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = summarize(sys.argv[1])
    for r in rows:
        print(" | ".join(f"{k}={v}" for k, v in r.items()))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
