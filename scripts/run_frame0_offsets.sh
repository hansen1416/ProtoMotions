#!/bin/bash
set -e

for i in $(seq -f "%04g" 0 15); do
    echo "=== Processing chunk $i ==="
    python scripts/compute_humos_frame0_offsets.py \
        --motion-file /home/hlz/datasets/humos_proto/humos_131072_${i}.pt \
        --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
        --out-motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_${i}_offset.pt \
        --limit -1 \
        --overwrite
done

echo "=== All 16 chunks done ==="
