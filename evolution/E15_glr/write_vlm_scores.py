"""write_vlm_scores.py JUDGE_JSON POOL_DIR [MAPPING_JSON]

SURROGATE ONLY: the judges are Claude agents reading the same contact-sheet prompt that filter/vlm_run.py gives Qwen3-VL-32B, labelled `vlm_claude*`
(not the production judge). The tag is meant to be read by `filter/decide.py --vlm claude`.


JUDGE_JSON: list of {clip, judge (1|2), verdict{...}} from the vlm-surrogate-judge workflow.
Writes into POOL_DIR/<clip>/scores.json (merge, same layout filter/vlm_run.py uses): vlm_claude1, vlm_claude2 (one per judge) and
vlm_claude (consensus: reconstructable = "yes" only when both judges say yes, otherwise the first non-yes verdict; other fields from
judge 1; `agree` = both judges gave the same `reconstructable`). Prints per-clip verdicts and the judge agreement.
MAPPING_JSON maps the anonymised sheet ids the judges saw (s00, s01 ...: the file name must not hint at the answer) back to clip names.
"""
import json
import sys
from pathlib import Path

js, pool = Path(sys.argv[1]), Path(sys.argv[2])
mapping = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
rows = json.loads(js.read_text())
by = {}
for r in rows:
    by.setdefault(mapping.get(r["clip"], r["clip"]), {})[int(r["judge"])] = r["verdict"]


def merge(d, key, val):
    p = d / "scores.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s[key] = val
    p.write_text(json.dumps(s, indent=1))


agree = n = 0
for clip, v in sorted(by.items()):
    d = pool / clip
    if not d.exists():
        print("no pool dir for", clip)
        continue
    for j, ver in v.items():
        merge(d, f"vlm_claude{j}", ver)
    if 1 in v and 2 in v:
        a, b = v[1], v[2]
        cons = dict(a)
        if a["reconstructable"] != "yes" or b["reconstructable"] != "yes":
            cons["reconstructable"] = a["reconstructable"] if a["reconstructable"] != "yes" else b["reconstructable"]
            cons["reason"] = a["reason"] if a["reconstructable"] != "yes" else b["reason"]
        cons["agree"] = a["reconstructable"] == b["reconstructable"]
        merge(d, "vlm_claude", cons)
        n += 1
        agree += cons["agree"]
        print(f'{clip:14s} j1={a["reconstructable"]:22s} j2={b["reconstructable"]:22s} motion={a["camera_motion"]}/{b["camera_motion"]} dyn={a["dynamic_level"]}/{b["dynamic_level"]} cuts={a["has_cuts"]}/{b["has_cuts"]}')
print(f"judge agreement on `reconstructable`: {agree}/{n}")
