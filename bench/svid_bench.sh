#!/bin/bash
# SpatialVID (MegaSaM variant) on the ScanNet++ benchmark clips, same frames as ViPE.
# Depth stages run once per resolution group (they batch frames across clips).
# Runs inside z00533370_spatialvid. Skips frame extraction: img/ is filled from bench frames.
R=/data/home/z00533370/code/SpatialVID
B=${B:-/data/home/z00533370/outputs/bench/scannetpp}
O=$B/spatialvid
mkdir -p $O/logs
cd $R && source .venv/bin/activate
export PATH=/data/home/z00533370/cache/bin:$PATH
export HF_HOME=/data/home/z00533370/cache/huggingface TORCH_HOME=/data/home/z00533370/cache/torch
export PYTHONPATH=$R:$R/utils:$PYTHONPATH OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
NG=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
python - <<PY
import json, os, cv2
B = "$B"; O = "$O"
clips = json.load(open(f"{B}/clips.json"))
rows = ["video_path,id,num_frames,height,width,aspect_ratio,fps,resolution"]
for c in clips:
    d = f"{O}/{c['clip']}/img"; os.makedirs(d, exist_ok=True)
    for j in range(c["n_frames"]):
        dst = f"{d}/frame_{j:06d}.jpg"
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(f"{B}/frames/{c['clip']}/{j:06d}.jpg"), dst)
    h, w = cv2.imread(f"{d}/frame_000000.jpg").shape[:2]
    rows.append(f"{B}/frames/{c['clip']},{c['clip']},{c['n_frames']},{h},{w},{w/h},30,{w}x{h}")
open(f"{O}/bench.csv", "w").write("\n".join(rows) + "\n")
# the batched depth stages stack frames across clips -> one CSV per resolution
import collections, glob
for f in glob.glob(f"{O}/bench_res_*.csv"):
    os.rename(f, f + ".old")
groups = collections.defaultdict(list)
for r in rows[1:]:
    groups[r.split(",")[-1]].append(r)
for res, rs in groups.items():
    open(f"{O}/bench_res_{res}.csv", "w").write("\n".join(rows[:1] + rs) + "\n")
print(len(groups), "resolution groups")
print(len(clips), "clips prepared")
PY
CSV=$O/bench.csv
t() { local s=$(date +%s); echo "== $1 start $(date +%T)"; shift; "$@"; echo "== done in $(( $(date +%s) - s )) s"; }
for G in $O/bench_res_*.csv; do
t depth-anything torchrun --standalone --nproc_per_node $NG camera_pose_annotation/depth_estimation/Depth-Anything/inference_batch.py --csv_path $G --encoder vitl --checkpoints_path checkpoints --output_dir $O --bs 16 --num_workers $NG
t unidepth torchrun --standalone --nproc_per_node $NG camera_pose_annotation/depth_estimation/UniDepth/inference_batch.py --csv_path $G --output_dir $O --checkpoints_path checkpoints --bs 32 --num_workers $NG
done
t tracking python camera_pose_annotation/camera_tracking/inference_batch.py --csv_path $CSV --dir_path $O --checkpoints_path checkpoints --gpu_id $CUDA_VISIBLE_DEVICES --num_workers $((NG * 2))
t cvd-pre python camera_pose_annotation/cvd_opt/preprocess/inference_batch.py --csv_path $CSV --dir_path $O --checkpoints_path checkpoints --gpu_id $CUDA_VISIBLE_DEVICES --num_workers $((NG * 2))
t cvd-opt python camera_pose_annotation/cvd_opt/inference_batch.py --csv_path $CSV --dir_path $O --gpu_id $CUDA_VISIBLE_DEVICES --num_workers $((NG * 2))
echo SVID_BENCH_DONE
