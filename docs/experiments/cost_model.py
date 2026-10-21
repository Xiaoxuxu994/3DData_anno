"""Cost model for the video -> 3D pseudo-label pipeline.

All per-clip numbers are measured on the H200 server (12 s clips, 15 fps, 180 frames, ~540x960 frames) unless the
scenario marks them as ASSUMED. Usage: python cost_model.py  (prints a markdown table)
"""

CLIP_FRAMES = 180

# ---- measured (see cost_model.md for the source of each number) ----
M = dict(
    filter_rules_s=6.0,          # TransNet 1.7 + flow 2.0 + GeoCalib 1.0 + OCR 1.0 (CPU) + rules 0.2; single process
    vlm32_s=3600 / 6300,         # Qwen3-VL-32B, 12 frames, 1 GPU, vLLM
    vipe_s_per_frame=0.463,      # filter_check median (442 clips); wild 0.61, eval3d_A 0.71
    pi3x_s_per_frame=0.091,
    da3_s_per_frame=0.049,       # wild median (A group 0.027)
    vggt_omega_s_per_frame=0.058,
    qc_cache_s=5.0,              # RAFT flow + LightGlue cache per clip
    raw_mb_per_clip=10.0,        # 18 GB _raw / 1835 clips
    frames_mb_per_clip=25.0,     # (63-18) GB / 1835 clips (jpg frames + sheets)
    label_mb_per_clip=70.0,      # ViPE artifacts ~68 MB; FF preds are 190-350 MB as float32 npz (intermediate)
    bw_mb_s=12.5,                # measured server download, shared
)

SCEN = {
    # v0: what we ran. ViPE + 3 FF models (Pi3X depth, DA3/VGGT-Ω for cross-model QC) on every filter-passed clip.
    "v0_current": dict(filter_pass=0.364, qc_keep=(0.20, 0.36), vipe_frac=1.0, vipe_speed=1.0,
                       ff=("pi3x", "da3", "vggt_omega"), extra_s=0.0),
    # v1: ASSUMED optimizations, to be validated: cut-splitting (+~9 pt pass), unified K (QC keep 0.40-0.50),
    # cheap FF+QC before ViPE (ViPE only on 60 % of passed clips), ViPE at 7.5 fps (0.5x) with 2 procs/GPU (1.5x).
    "v1_optimized": dict(filter_pass=0.45, qc_keep=(0.40, 0.50), vipe_frac=0.6, vipe_speed=1 / (0.5 / 1.5),
                         ff=("pi3x", "da3"), extra_s=1.0),
    # v2: ASSUMED, only if Pi3X conditioned on GeoCalib/ViPE K matches ViPE pose: no ViPE except 10 % audit.
    "v2_ff_only": dict(filter_pass=0.45, qc_keep=(0.40, 0.50), vipe_frac=0.1, vipe_speed=1.0,
                       ff=("pi3x", "da3"), extra_s=1.0),
}


def per_raw_clip_gpu_s(s):
    pre = M["filter_rules_s"] + M["vlm32_s"]
    ff = sum(M[f"{f}_s_per_frame"] for f in s["ff"]) * CLIP_FRAMES
    vipe = M["vipe_s_per_frame"] * CLIP_FRAMES * s["vipe_frac"] / s["vipe_speed"]
    post = ff + vipe + M["qc_cache_s"] + s["extra_s"]
    return pre + s["filter_pass"] * post, pre, post


def report(usable=1_000_000, gpus=(8, 64)):
    rows = []
    for name, s in SCEN.items():
        per_raw, pre, post = per_raw_clip_gpu_s(s)
        for keep in s["qc_keep"]:
            yld = s["filter_pass"] * keep
            raw = usable / yld
            gpu_h = raw * per_raw / 3600
            dl_tb = raw * M["raw_mb_per_clip"] / 1e6
            dl_days = raw * M["raw_mb_per_clip"] / M["bw_mb_s"] / 86400
            store_tb = (raw * M["frames_mb_per_clip"] * 0 + usable * (M["frames_mb_per_clip"] + M["label_mb_per_clip"])) / 1e6
            rows.append((name, keep, yld, raw, raw * 12 / 3600, per_raw, post, gpu_h,
                         *[gpu_h / g / 24 for g in gpus], dl_tb, dl_days, store_tb))
    hdr = ("scenario", "QC keep", "yield", "raw clips", "raw hours", "GPU-s/raw clip", "GPU-s/passed clip",
           "GPU-h", *[f"days@{g}GPU" for g in gpus], "download TB", "download days@12.5MB/s", "kept storage TB")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        f = [r[0], f"{r[1]:.2f}", f"{r[2]:.1%}", f"{r[3] / 1e6:.1f}M", f"{r[4] / 1e3:.0f}K", f"{r[5]:.1f}",
             f"{r[6]:.0f}", f"{r[7] / 1e3:.0f}K", *[f"{x:.0f}" for x in r[8:8 + len(gpus)]],
             f"{r[-3]:.0f}", f"{r[-2]:.0f}", f"{r[-1]:.0f}"]
        print("| " + " | ".join(f) + " |")


if __name__ == "__main__":
    print("## per 1M usable clips\n")
    report(1_000_000)
    print("\n## per 100K usable clips\n")
    report(100_000)
