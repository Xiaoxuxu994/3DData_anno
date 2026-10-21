"""ViPE on the stability variant benches; several launchers (one per GPU) share clips through O_EXCL claim files.

Usage: python vipe_variant.py GPU PER_GPU BENCH_DIR [BENCH_DIR ...]  -> BENCH_DIR/vipe/
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

GPU, PER, BENCHES = sys.argv[1], int(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
VIPE = "/data/home/z00533370/code/vipe/.venv/bin/vipe"


def jobs():
    for B in BENCHES:
        for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
            yield B, c["clip"], c["n_frames"]


def claim(p):
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def worker():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    for B, clip, n in jobs():
        out = B / "vipe"
        (out / "logs").mkdir(parents=True, exist_ok=True)
        if (out / "depth" / f"{clip}.zip").exists() or not claim(out / "logs" / f"{clip}.claim"):
            continue
        t0 = time.time()
        with open(out / "logs" / f"{clip}.log", "w") as log:
            rc = subprocess.call([VIPE, "infer", "--image-dir", str(B / "frames" / clip), "-o", str(out), "-p", "default"],
                                 cwd="/data/home/z00533370/code/vipe", env=env, stdout=log, stderr=subprocess.STDOUT)
        rec = {"clip": clip, "n_frames": n, "gpu": GPU, "rc": rc, "sec": round(time.time() - t0, 1)}
        with open(out / "logs" / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    ps = [mp.Process(target=worker) for _ in range(PER)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("ALL_DONE", flush=True)
