#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

MOTION_DIR=/home/hlz/datasets/humos_proto_neutral
ASSET_ROOT=/home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor_neutral

for i in 0000 0001 0002 0003 0004 0005; do
    echo "=== chunk ${i} ==="
    python tools/compute_humos_frame0_offsets.py \
        --motion-file  "${MOTION_DIR}/humanml3d_neutral_20951_${i}.pt" \
        --asset-root   "${ASSET_ROOT}" \
        --out-motion-file "${MOTION_DIR}/humanml3d_neutral_20951_${i}_offset.pt" \
        --overwrite
done

echo "=== all chunks done ==="
