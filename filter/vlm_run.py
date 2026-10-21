"""VLM judge for the filter pool with vLLM (offline batch), one shared prompt / JSON schema for every model.

Usage: CUDA_VISIBLE_DEVICES=4,5 python vlm_run.py MODEL_TAG MODEL_PATH TP POOL_DIR [POOL_DIR ...]
  12 evenly spaced frames per clip, long side 640 px, sent as interleaved images with timestamps.
  No guided decoding: JSON parse rate is one of the measured quantities.
Writes POOL/<clip>/scores.json["vlm_<TAG>"] (parsed JSON or {"parse_error": raw}) and
outputs/logs/filter/vlm_<TAG>.json (load sec, wall sec, clips/GPU-hour, parse rate, GPU memory).
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from fcommon import LOGS, clips, even_idx, frame_paths, has_stage, write_scores  # noqa: E402

TAG, PATH, TP, POOLS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
NF, LONG = 12, 640
STAGE = f"vlm_{TAG}"

PROMPT = """You are a strict data curator for 3D reconstruction training data (camera poses and depth will be estimated \
from this clip by SLAM / structure-from-motion). The images above are {nf} frames uniformly sampled, in temporal order, \
from one 12-second video clip.

Answer ONLY with one JSON object (no markdown, no extra text) with exactly these keys:
"scene_category": one of "indoor_static", "indoor_dynamic", "outdoor_static", "outdoor_dynamic", "driving", "embodied", "other"
   (dynamic = moving people / vehicles / objects cover a noticeable part of the frame; driving = camera on a road vehicle;
    embodied = first-person head / chest / robot view with hands or the body interacting with objects)
"indoor_outdoor": "indoor" | "outdoor" | "mixed"
"dynamic_level": "none" | "low" | "medium" | "high"
"camera_motion": "static" | "mostly_translation" | "mostly_rotation" | "zoom" | "mixed"
"overlays": {{"subtitles": true/false, "watermark": true/false, "picture_in_picture": true/false}}
"has_cuts": true/false   (any shot change, scene cut or transition between frames)
"is_cg_or_game": true/false
"fisheye": true/false
"low_light_or_blur": true/false
"reconstructable": "yes" | "no_unreconstructable" | "no_inaccurate"
   - "no_unreconstructable": no usable geometry, e.g. camera static or only rotating / panning / zooming (no parallax),
     scene extremely far away (high-altitude aerial), shot cuts, CG or game footage, frames dominated by moving objects,
     textureless views
   - "no_inaccurate": geometry exists but poses / depth would be unreliable, e.g. heavy motion blur, very low light,
     strong fisheye distortion, large overlays, extreme dynamics, violent shake
   - "yes": a moving camera with real parallax over a mostly static, real (non-CG) scene
"reason": one short sentence
"confidence": number between 0 and 1"""

KEYS = ["scene_category", "indoor_outdoor", "dynamic_level", "camera_motion", "overlays", "has_cuts", "is_cg_or_game",
        "fisheye", "low_light_or_blur", "reconstructable", "reason", "confidence"]


def b64(img):
    h, w = img.shape[:2]
    s = LONG / max(h, w)
    img = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA) if s < 1 else img
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()


def message(d):
    ps = frame_paths(d)
    idx = even_idx(len(ps), NF)
    content = []
    for k, i in enumerate(idx):
        content.append({"type": "text", "text": f"Frame {k + 1} (t={i / 15:.1f}s):"})
        content.append({"type": "image_url", "image_url": {"url": b64(cv2.imread(str(ps[i])))}})
    content.append({"type": "text", "text": PROMPT.format(nf=NF)})
    return [{"role": "user", "content": content}]


def parse(txt):
    t = txt.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1]
    a, b = t.find("{"), t.rfind("}")
    try:
        j = json.loads(t[a:b + 1])
        return j if all(k in j for k in ("reconstructable", "camera_motion")) else None
    except Exception:  # noqa: BLE001
        return None


def gpu_mem():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return r


def main():
    todo = [d for p in POOLS for d in clips(p) if (d / "meta.json").exists() and not has_stage(d, STAGE)]
    print(STAGE, len(todo), "clips", flush=True)
    t0 = time.time()
    llm = LLM(model=PATH, tensor_parallel_size=TP, max_model_len=int(os.environ.get("VLM_MAXLEN", 16384)), gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": NF}, trust_remote_code=True, seed=0,
              enforce_eager=os.environ.get("VLM_EAGER") == "1")
    load_sec = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=600)
    ck = {"enable_thinking": False}
    n_ok = n = 0
    t1 = time.time()
    for k in range(0, len(todo), 64):
        batch = todo[k:k + 64]
        outs = llm.chat([message(d) for d in batch], sp, chat_template_kwargs=ck, use_tqdm=False)
        for d, o in zip(batch, outs):
            txt = o.outputs[0].text
            j = parse(txt)
            n += 1
            n_ok += j is not None
            write_scores(d, STAGE, j if j is not None else {"parse_error": txt[:2000]})
        el = time.time() - t1
        print(f"{n}/{len(todo)} parse_ok={n_ok} {el:.0f}s {n / el * 3600 / TP:.0f} clips/GPU-h", flush=True)
    wall = time.time() - t1
    import torch

    stat = {"tag": TAG, "path": PATH, "tp": TP, "n": n, "parse_ok": n_ok, "parse_rate": n_ok / max(n, 1),
            "load_sec": round(load_sec, 1), "wall_sec": round(wall, 1),
            "clips_per_gpu_hour": round(n / max(wall, 1e-6) * 3600 / TP, 1),
            "nvidia_smi_apps": gpu_mem(), "torch_reserved_gb_this_proc": round(torch.cuda.max_memory_reserved() / 1e9, 1),
            "pools": POOLS, "n_frames": NF, "long_side": LONG}
    LOGS.mkdir(parents=True, exist_ok=True)
    prev = LOGS / f"vlm_{TAG}.json"
    hist = json.loads(prev.read_text()) if prev.exists() else []
    (prev).write_text(json.dumps(hist + [stat], indent=1))
    print("VLM_DONE", json.dumps({k: v for k, v in stat.items() if k != "nvidia_smi_apps"}), flush=True)


if __name__ == "__main__":  # guard needed for VLLM_WORKER_MULTIPROC_METHOD=spawn
    main()
