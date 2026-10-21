#!/usr/bin/env python3
"""Rewrite the original root path in every tracked text file of this checkout.

The scripts were written on one machine with ROOT=/data/home/z00533370 and the layout
ROOT/{code,data,outputs,cache}; this repo is ROOT/code. To run them under another root:

    git clone <repo> $NEW_ROOT/code
    python3 $NEW_ROOT/code/tools/relocate.py $NEW_ROOT          # dry run: lists files
    python3 $NEW_ROOT/code/tools/relocate.py $NEW_ROOT --apply

It only replaces the literal prefix; review `git diff` and keep it as a local, uncommitted change
(or commit on your own branch). New code should read ROOT from $ANNO_ROOT instead.
"""
import subprocess, sys
from pathlib import Path

OLD = "/data/home/z00533370"
if len(sys.argv) < 2:
    sys.exit(__doc__)
new = sys.argv[1].rstrip("/")
apply = "--apply" in sys.argv
repo = Path(__file__).resolve().parents[1]
files = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
n = 0
for f in files:
    p = repo / f
    if p.suffix not in {".py", ".sh", ".yaml", ".yml", ".md", ".json", ".txt"} or p == Path(__file__).resolve():
        continue
    s = p.read_text(encoding="utf-8", errors="surrogateescape")
    if OLD not in s:
        continue
    n += 1
    print(f"{s.count(OLD):4d}  {f}")
    if apply:
        p.write_text(s.replace(OLD, new), encoding="utf-8", errors="surrogateescape")
print(f"{n} files {'rewritten' if apply else 'would change (add --apply)'}: {OLD} -> {new}")
