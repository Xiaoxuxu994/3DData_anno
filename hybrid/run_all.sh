#!/bin/bash
# CPU-only hybrid evaluation chain (no GPU). Logs -> outputs/hybrid/logs/
H=/data/home/z00533370
L=$H/outputs/hybrid/logs; mkdir -p $L
for k in A B scannetpp wild; do
  docker exec -w $H/code/hybrid -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 z00533370_dev \
    $H/code/SpatialVID/.venv/bin/python run_hybrid.py $k --workers=32 > $L/$k.log 2>&1
done
touch $L/all.done
