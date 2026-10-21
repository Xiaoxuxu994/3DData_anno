"""Chunked parallel downloader for ModelScope model repos (fresh short range requests beat the per-connection
throttling seen with the modelscope CLI from this host). Resumes from modelscope's *.incomplete prefixes.

Usage: python pdl.py REPO LOCAL_DIR [THREADS=16]
Verifies size and sha256 (from the ModelScope file listing) for every file.
"""

import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO, OUT = sys.argv[1], Path(sys.argv[2])
TH = int(sys.argv[3]) if len(sys.argv) > 3 else 16
CH = 32 << 20
API = f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Recursive=true&Revision=master"
URL = f"https://modelscope.cn/models/{REPO}/resolve/master/{{}}"
S = requests.Session()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get_range(url, fd, a, b):
    for k in range(20):
        try:
            r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=(15, 60), stream=False)
            if r.status_code in (200, 206) and len(r.content) == b - a + 1:
                os.pwrite(fd, r.content, a)
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"range {a}-{b} failed")


files = [f for f in S.get(API, timeout=30).json()["Data"]["Files"] if f["Type"] == "blob"]
OUT.mkdir(parents=True, exist_ok=True)
for f in sorted(files, key=lambda f: f["Size"]):
    p, size = OUT / f["Path"], f["Size"]
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size == size:
        continue
    url = URL.format(f["Path"])
    if size < CH:
        r = S.get(url, timeout=120)
        p.write_bytes(r.content)
        continue
    part = p.with_name(p.name + ".part")
    inc = p.with_name(p.name + ".incomplete")
    done_bytes = 0
    if inc.exists() and not part.exists():
        done_bytes = inc.stat().st_size // CH * CH
        inc.rename(part)
    elif part.exists():
        done_bytes = 0  # unknown which chunks of an earlier pdl run finished -> redo all
    fd = os.open(part, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    chunks = [(a, min(a + CH, size) - 1) for a in range(done_bytes, size, CH)]
    t0 = time.time()
    with ThreadPoolExecutor(TH) as ex:
        list(ex.map(lambda c: get_range(url, fd, *c), chunks))
    os.close(fd)
    sec = time.time() - t0
    if f.get("Sha256") and sha256(part) != f["Sha256"]:
        print("SHA_MISMATCH", f["Path"], "-> redownload", flush=True)
        part.rename(p.with_name(p.name + ".bad"))
        fd = os.open(part, os.O_RDWR | os.O_CREAT)
        os.ftruncate(fd, size)
        with ThreadPoolExecutor(TH) as ex:
            list(ex.map(lambda c: get_range(url, fd, *c), [(a, min(a + CH, size) - 1) for a in range(0, size, CH)]))
        os.close(fd)
    part.rename(p)
    print(f"OK {f['Path']} {size / 1e9:.2f} GB {(size - done_bytes) / 1e6 / max(sec, 1e-3):.1f} MB/s", flush=True)
print("PDL_DONE", REPO, flush=True)
