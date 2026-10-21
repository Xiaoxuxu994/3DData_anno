"""Fetch single members of a remote ZIP via HTTP Range (no full download).

Usage: python zip_member_fetch.py URL OUT_DIR MEMBER [MEMBER ...]
MEMBER may be a glob-free exact path inside the zip. Stored members are written as-is; deflated ones are inflated.
Parallel range requests (8 streams) per member; resumes from *.part files.
"""

import os
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

URL, OUT, MEMBERS = sys.argv[1], sys.argv[2], sys.argv[3:]


def get(a, b, tries=20):
    for t in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            time.sleep(min(60, 2 ** t))
    raise RuntimeError(f"range {a}-{b} failed")


size = int(urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60).headers["Content-Length"])
tail = get(max(0, size - 70000), size - 1)
eocd = tail.rfind(b"PK\x05\x06")
cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
loc64 = tail.rfind(b"PK\x06\x07")
if cd_off == 0xFFFFFFFF and loc64 >= 0:  # zip64
    eocd64_off = struct.unpack("<Q", tail[loc64 + 8:loc64 + 16])[0]
    e64 = get(eocd64_off, eocd64_off + 55)
    cd_size, cd_off = struct.unpack("<QQ", e64[40:56])
cd = get(cd_off, cd_off + cd_size - 1)
entries, p = {}, 0
while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
    method, = struct.unpack("<H", cd[p + 10:p + 12])
    csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
    nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
    loff, = struct.unpack("<I", cd[p + 42:p + 46])
    name = cd[p + 46:p + 46 + nlen].decode()
    extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
    q = 0
    while q < len(extra):  # zip64 extended info
        hid, hsz = struct.unpack("<HH", extra[q:q + 4])
        if hid == 1:
            vals, r = [], q + 4
            for cond in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, loff == 0xFFFFFFFF):
                if cond:
                    vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                    r += 8
            it = iter(vals)
            if usize == 0xFFFFFFFF:
                usize = next(it)
            if csize == 0xFFFFFFFF:
                csize = next(it)
            if loff == 0xFFFFFFFF:
                loff = next(it)
        q += 4 + hsz
    entries[name] = (method, csize, usize, loff)
    p += 46 + nlen + xlen + clen

os.makedirs(OUT, exist_ok=True)
for m in MEMBERS:
    if m not in entries:
        print("MISSING", m, "available:", [k for k in entries if k.endswith(".zip") or k.endswith(".txt")][:20], flush=True)
        continue
    method, csize, usize, loff = entries[m]
    lh = get(loff, loff + 29)
    nlen, xlen = struct.unpack("<HH", lh[26:30])
    start = loff + 30 + nlen + xlen
    dst = os.path.join(OUT, os.path.basename(m))
    if os.path.exists(dst) and os.path.getsize(dst) == usize:
        print("SKIP", m, flush=True)
        continue
    chunk = 32 * 1024 * 1024
    ranges = [(start + o, min(start + o + chunk, start + csize) - 1) for o in range(0, csize, chunk)]
    part_dir = dst + ".parts"
    os.makedirs(part_dir, exist_ok=True)

    def fetch(i_ab):
        i, (a, b) = i_ab
        pp = os.path.join(part_dir, f"{i:05d}")
        if os.path.exists(pp) and os.path.getsize(pp) == b - a + 1:
            return
        data = get(a, b)
        with open(pp + ".tmp", "wb") as f:
            f.write(data)
        os.replace(pp + ".tmp", pp)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, enumerate(ranges)))
    with open(dst + ".tmp", "wb") as out:
        dec = zlib.decompressobj(-15) if method == 8 else None
        for i in range(len(ranges)):
            with open(os.path.join(part_dir, f"{i:05d}"), "rb") as f:
                buf = f.read()
            out.write(dec.decompress(buf) if dec else buf)
        if dec:
            out.write(dec.flush())
    assert os.path.getsize(dst + ".tmp") == usize, "size mismatch"
    os.replace(dst + ".tmp", dst)
    os.rename(part_dir, part_dir + "_used")
    print("OK", m, round(csize / 1e6), "MB", round(time.time() - t0), "s", flush=True)
