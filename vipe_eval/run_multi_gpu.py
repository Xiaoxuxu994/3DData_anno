"""Run ViPE on all segments with one worker per GPU pulling from a shared queue (longest first).

Usage: python run_multi_gpu.py [--gpus 0,1,...] [--only SEG ...]
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/home/z00533370/outputs/vipe/scannetpp_sample30")
VIPE = "/data/home/z00533370/code/vipe/.venv/bin/vipe"


def worker(gpu, q, lock):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        try:
            seg, n = q.get_nowait()
        except Exception:
            return
        if (ROOT / "vipe" / "pose" / f"{seg}.npz").exists():
            continue
        t0 = time.time()
        with open(ROOT / "logs" / f"{seg}.log", "w") as log:
            rc = subprocess.call(
                [VIPE, "infer", "--image-dir", str(ROOT / "frames" / seg), "-o", str(ROOT / "vipe")],
                cwd="/data/home/z00533370/code/vipe", env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        rec = {"segment": seg, "n_frames": n, "gpu": gpu, "rc": rc, "sec": round(time.time() - t0, 1)}
        with lock, open(ROOT / "logs" / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    segs = json.loads((ROOT / "segments.json").read_text())
    if a.only:
        segs = [s for s in segs if s["segment"] in a.only]
    (ROOT / "logs").mkdir(exist_ok=True)
    q = mp.Queue()
    for s in sorted(segs, key=lambda s: -s["n_frames"]):
        q.put((s["segment"], s["n_frames"]))
    lock = mp.Lock()
    ps = [mp.Process(target=worker, args=(int(g), q, lock)) for g in a.gpus.split(",")]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("ALL_DONE", flush=True)
