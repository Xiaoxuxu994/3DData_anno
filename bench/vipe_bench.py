"""ViPE on benchmark clips: workers = GPUS x PER_GPU, shared queue, OMP threads capped.

Usage: python vipe_bench.py BENCH_DIR GPUS PER_GPU [pipeline]
  BENCH_DIR has frames/<clip>/ and clips.json; results go to BENCH_DIR/vipe[_<pipeline>]/
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

B, GPUS, PER = Path(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3])
PIPE = sys.argv[4] if len(sys.argv) > 4 else "default"
OUT = B / ("vipe" if PIPE == "default" else f"vipe_{PIPE}")
VIPE = "/data/home/z00533370/code/vipe/.venv/bin/vipe"


def worker(gpu, q, lock):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="12", MKL_NUM_THREADS="12")
    while True:
        try:
            clip, n = q.get_nowait()
        except Exception:
            return
        if (OUT / "depth" / f"{clip}.zip").exists():
            continue
        t0 = time.time()
        with open(OUT / "logs" / f"{clip}.log", "w") as log:
            rc = subprocess.call([VIPE, "infer", "--image-dir", str(B / "frames" / clip), "-o", str(OUT), "-p", PIPE],
                                 cwd="/data/home/z00533370/code/vipe", env=env, stdout=log, stderr=subprocess.STDOUT)
        rec = {"clip": clip, "n_frames": n, "gpu": gpu, "rc": rc, "sec": round(time.time() - t0, 1)}
        with lock, open(OUT / "logs" / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    clips = json.loads((B / "clips.json").read_text())
    q = mp.Queue()
    for c in sorted(clips, key=lambda c: -c["n_frames"]):
        q.put((c["clip"], c["n_frames"]))
    lock = mp.Lock()
    ps = [mp.Process(target=worker, args=(g, q, lock)) for g in GPUS for _ in range(PER)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("ALL_DONE", flush=True)
