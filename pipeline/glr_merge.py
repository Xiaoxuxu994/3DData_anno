"""GLR track, last step: one verdict and one label per clip from the filter verdict, the route and the QC of the two annotators.

Usage: python glr_merge.py OUT
Reads : OUT/filter/glr_route.csv (glr_prepare.py route), OUT/results.csv + OUT/labels/ (v1.3, run_batch.sh),
        OUT/results_glr.csv + OUT/labels_glr/ (glr_collect.py; optional)
Writes: OUT/final/results.csv, OUT/final/labels/<clip>.npz (symlinks to the chosen label file), OUT/final/summary.json

Policy (a tier is "no known failure signature fired", not "accurate"):
  filter reject                              -> reject      (the filter's reasons are kept)
  route glr and GLRMap tier T1 / T2          -> GLRMap label (annotator glr)
  route glr and GLRMap T3 / failed / stopped -> v1.3 label if its own v1c rule accepts (annotator v13, fallback = yes) else reject
  route v13                                  -> v1.3 label if accepted else reject
Failure mechanisms differ (SfM needs parallax and a focal it can observe; the feed-forward model needs a plausible K), so the fallback is
not circular - but whether the two are really complementary is a hypothesis to test on the Phase-2 corpus (docs/GLR_INTEGRATION.md).
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glr_common as C  # noqa: E402


def read_csv(p):
    p = Path(p)
    if not p.exists():
        return {}
    with open(p) as fh:
        return {r["clip"]: r for r in csv.DictReader(fh)}


def truthy(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def decide(route, glr, v13):
    """route row, GLR row (or None), v1.3 row (or None) -> (annotator, tier, accept, reason, fallback)."""
    r = route.get("route")
    if r == "reject":
        return None, "T3", False, "filter:" + (route.get("filter_reasons") or route.get("reasons") or ""), False
    v13_ok = v13 is not None and truthy(v13.get("accept"))
    if r == "glr":
        if glr is not None and truthy(glr.get("accept")):
            return "glr", glr["tier"], True, "", False
        why = "glr:" + ((glr or {}).get("reasons") or "no_result")
        if v13_ok:
            return "v13", "v13", True, why, True
        return None, "T3", False, why + (";v13:" + ((v13 or {}).get("reject_flags") or "v1c") if v13 is not None else ";v13:none"), False
    if v13_ok:
        return "v13", "v13", True, "", False
    return None, "T3", False, "v13:" + ((v13 or {}).get("reject_flags") or "v1c_or_error"), False


def main():
    out = Path(sys.argv[1])
    routes, v13s, glrs = read_csv(out / "filter" / "glr_route.csv"), read_csv(out / "results.csv"), read_csv(out / "results_glr.csv")
    (out / "final" / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for clip, rt in routes.items():
        ann, tier, ok, why, fb = decide(rt, glrs.get(clip), v13s.get(clip))
        src = None
        if ok:
            src = (out / "labels_glr" / f"{clip}.npz") if ann == "glr" else (out / "labels" / f"{clip}.npz")
            dst = out / "final" / "labels" / f"{clip}.npz"
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())
            elif not src.exists():
                ok, why = False, why + ";label_file_missing"
        rows.append({"clip": clip, "route": rt.get("route"), "accept": ok, "annotator": ann or "", "tier": tier, "fallback": fb,
                     "reason": why, "glr_v1c": (glrs.get(clip) or {}).get("v1c"), "v13_v1c": (v13s.get(clip) or {}).get("v1c")})
    with open(out / "final" / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["clip"])
        w.writeheader()
        w.writerows(rows)
    summ = {"n": len(rows), "accepted": sum(r["accept"] for r in rows), "by_route": dict(Counter(r["route"] for r in rows)),
            "by_annotator": dict(Counter(r["annotator"] for r in rows if r["accept"])),
            "by_tier": dict(Counter(r["tier"] for r in rows if r["accept"])), "fallbacks": sum(r["fallback"] for r in rows),
            "reject_reasons": dict(Counter(x for r in rows if not r["accept"] for x in r["reason"].split(";") if x))}
    C.dump_json(out / "final" / "summary.json", summ)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
