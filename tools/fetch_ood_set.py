"""Rebuild the E15 OOD clip set (docs/experiments/E15_ood_manifest.json) on any machine with network access and ffmpeg.

Usage: python tools/fetch_ood_set.py MANIFEST.json OUTDIR [--only id1,id2] [--role easy|control_negative] [--no-verify]
  For every clip of the manifest: download `download_url` to OUTDIR/raw/<id>/<file> (skipped when it is already there), check its
  sha256 against the manifest (a mismatch means different footage: the clip is skipped unless --no-verify) and cut the 12 s window
  to OUTDIR/clips/<id>.mp4 with the same ffmpeg recipe the set was built with (15 fps, long side <= 1920, x264 crf 17, no audio).
  Nothing is overwritten: an existing raw file or clip is kept, a new OUTDIR is created if needed.
  OUTDIR/clips.txt lists the clips (one path per line) - the input list for pipeline/run_batch_glr.sh.
Licences differ per source (Mixkit free licence, Pexels licence, CC BY / CC BY-SA 2.0 on archive.org, public domain for U.S. government
footage) and are printed for every clip; the clips are for evaluation of the annotation pipeline, check the licence before redistributing
anything. Hosts may rate-limit or change files; a failed clip is reported and skipped, the others continue.
Environment: FFMPEG (default `ffmpeg`).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (research; E15 OOD set rebuild)"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dst, timeout=120):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.rename(dst)


def cut(src, dst, start, dur, fps=15, long=1920):
    vf = f"fps={fps},scale='if(gt(iw,ih),min({long},iw),-2)':'if(gt(iw,ih),-2,min({long},ih))':flags=lanczos"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("." + dst.name)
    subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(src), "-t", str(dur),
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-f", "mp4", str(tmp)], check=True)
    tmp.rename(dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("outdir")
    ap.add_argument("--only")
    ap.add_argument("--role", choices=("easy", "control_negative"))
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    man = json.loads(Path(a.manifest).read_text())
    want = set(a.only.split(",")) if a.only else None
    out = Path(a.outdir)
    done, failed = [], []
    for c in man["clips"]:
        if (want and c["id"] not in want) or (a.role and c["role"] != a.role):
            continue
        try:
            raw = out / "raw" / c["id"] / c["raw_name"]
            clip = out / "clips" / f"{c['id']}.mp4"
            print(f"{c['id']:16s} {c['role']:16s} {c['license']}  {c['page_url']}", flush=True)
            if clip.exists():
                done.append(clip)
                continue
            if not raw.exists():
                download(c["download_url"], raw)
            if not a.no_verify and c.get("raw_sha256") and sha256(raw) != c["raw_sha256"]:
                raise ValueError(f"sha256 of {raw.name} differs from the manifest (the host serves a different file); use --no-verify to accept it")
            cut(raw, clip, c["window_start"], c["window_dur"])
            done.append(clip)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {c['id']}: {e!r}", file=sys.stderr)
            failed.append(c["id"])
    if done:
        (out / "clips.txt").write_text("\n".join(str(p.resolve()) for p in done) + "\n")
    print(f"{len(done)} clips ready in {out / 'clips'}" + (f", failed: {','.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
