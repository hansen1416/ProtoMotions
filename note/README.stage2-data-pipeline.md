# Stage 2 Data Pipeline

Converts the raw HUMOS inference output (22,459 `.pt` files on Google Drive) into
grounding-offset MotionLib shards on R2, ready for Stage 2 training (20,946 clips × 128 shapes).

Everything runs on a **remote server** (not local). The orchestration script handles all steps
and resumes automatically if interrupted.

---

## Overview

**Source:** `gdrive:humos_output/` — 22,459 `.pt` files, one per HumanML3D clip,
each containing 128 shape variants (64 betas × 2 genders).

**Target:** `r2:proto-data/20946_humos_offset/` — ~335 offset shards × ~3.4 GB each ≈ 1.1 TB total.

**Script:** `tools/prepare_stage2_data.py` — processes in batches of 4096 clips,
peak disk ~750 GB, cleans up intermediates after each batch.

**Shard guarantee:** all 128 shape variants of each clip always land in the same shard
(8192 motions/shard = 64 clips × 128 shapes exactly).

---

## Remote Server Requirements

| Resource | Minimum |
|----------|---------|
| GPU VRAM | 16 GB (IsaacGym frame-0 offset) |
| Disk | 900 GB free (750 GB peak + headroom) |
| RAM | 64 GB |
| rclone | configured with `gdrive:` and `r2:` remotes |
| Conda envs | `smplsim` (ProtoMotions + numpy/torch), `isaacgym` (IsaacGym) |

---

## Files to Upload to Remote Server Before Starting

| File | Source (local) | Destination (remote) |
|------|---------------|----------------------|
| `valid_ids_sorted_by_difficulty.txt` | `/home/hlz/repos/hhi/data-processing/` | `/workspace/` |
| ProtoMotions repo | `/home/hlz/repos/ProtoMotions` | `/workspace/ProtoMotions` |
| `all_betas.pt` | `protomotions/data/assets/` | (already in repo) |
| `mjcf/smpl_mor/` | `protomotions/data/assets/` | (already in repo) |

---

## Per-Batch Disk Usage (4096 clips)

| Step | Action | Peak added | Freed after |
|------|--------|-----------|-------------|
| 1 | Download .pt from GDrive | +144 GB | — |
| 2 | Export .pt → NPZ | +254 GB | .pt deleted (−144 GB) |
| 3 | Convert NPZ → MotionLib shards | +216 GB | NPZ deleted (−254 GB) |
| 4 | Apply grounding offset | +216 GB | pre-offset deleted (−216 GB) |
| 5 | Upload offset shards → R2 | — | local deleted (−216 GB) |

**Peak:** ~470 GB (during step 3, NPZ + MotionLib both on disk). Back to ~0 before next batch.

---

## Setup on Remote Server

```bash
# 1. Clone repo and install
git clone <ProtoMotions repo> /workspace/ProtoMotions
cd /workspace/ProtoMotions
pip install -e .

# 2. Configure rclone remotes (if not already set up)
rclone config   # add gdrive: and r2: remotes

# 3. Verify GDrive access
rclone lsf gdrive:humos_output/ --files-only | wc -l
# should print 22459

# 4. For headless servers — start a virtual display (required by IsaacGym)
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
```

---

## Run the Pipeline

### Local (R drive)

```bash
conda activate smplsim

python tools/prepare_stage2_data.py \
    --valid-ids /path/to/valid_ids_sorted_by_difficulty.txt \
    --workspace /media/hlz/R/stage2_prep \
    --proto-root /home/hlz/repos/ProtoMotions \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --output-dir /media/hlz/R/stage2_data \
    --local-humos-cache /media/hlz/R/humos_output \
    --batch-clips 512 \
    --motions-per-shard 8192 \
    --isaacgym-env isaacgym \
    --device cuda
```

- **~41 batches** (20,946 / 512). 6,947 clips already in local cache — those batches skip GDrive download.
- **Peak disk per batch:** ~59 GB in workspace, cleans up after each batch.
- **Output accumulates** in `/media/hlz/R/stage2_data/` (~1.1 TB when complete).

### Remote server (R2 upload)

```bash
# Headless display for IsaacGym:
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99

conda activate smplsim

python tools/prepare_stage2_data.py \
    --valid-ids /workspace/valid_ids_sorted_by_difficulty.txt \
    --workspace /workspace/stage2_prep \
    --proto-root /workspace/ProtoMotions \
    --asset-root /workspace/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --r2-dest r2:proto-data/20946_humos_offset \
    --batch-clips 4096 \
    --motions-per-shard 8192 \
    --isaacgym-env isaacgym \
    --device cuda
```

- **6 batches** (5 × 4096 + 1 × 946). Peak disk ~750 GB.

### Resume

Re-run the exact same command. Completed batches are recorded in
`{workspace}/pipeline_log.txt` and skipped automatically.

Each log entry looks like:
```
2026-06-24 14:30:00  batch_0000  STARTED  512 clips
2026-06-24 16:45:00  batch_0000  DONE  512 clips  8 shards  cached=512  downloaded=0
```

---

## What the Script Calls (internal steps)

Each batch runs these tools in sequence:

```
# Step 1 — rclone (filter to batch keyids only)
rclone copy gdrive:humos_output/ {raw_dir}/ --filter-from rclone_filter.txt --transfers=16

# Step 2 — export .pt → AMASS NPZ + manifest YAML
python tools/export_humos_to_amass_npz.py \
    --input-dir {raw_dir} --out-root {npz_dir} \
    --yaml-name batch_NNNN.yaml --fps 30.0 --skip-existing

# Step 3 — NPZ → MotionLib shards (8192 motions each = 64 clips × 128 shapes)
python tools/convert_amass_to_motionlib_with_morphology.py \
    {npz_dir} {proto_dir} \
    --motion-config {npz_dir}/batch_NNNN.yaml \
    --humanoid-type smpl --output-fps 30 --device cuda --batch-size 8192

# Step 4 — grounding offset per shard (loops over all .pt in proto_dir)
conda run -n isaacgym python tools/compute_humos_frame0_offsets.py \
    --motion-file {chunk} --asset-root {asset_root} \
    --out-motion-file {offset_dir}/{chunk}_offset.pt --limit -1 --overwrite

# Step 5 — upload and clean
rclone copy {offset_dir}/ r2:proto-data/20946_humos_offset/ \
    --transfers=4 --s3-upload-concurrency=4 --s3-chunk-size=64M \
    --retries=10 --retries-sleep=30s
```

---

## Output on R2

```
r2:proto-data/20946_humos_offset/
    batch_0000_0000_offset.pt   # batch 0, shard 0  (~3.4 GB, 64 clips × 128 shapes)
    batch_0000_0001_offset.pt
    ...
    batch_0004_0063_offset.pt   # batch 4, last full shard
    batch_0005_0000_offset.pt   # batch 5 (946 clips, partial last batch)
    ...
    # ~335 shards total, ~1.1 TB
```

---

## Using on RunPod for Training

After all shards are on R2, download to RunPod and create a slurmrank pointer:

```bash
# On RunPod
rclone copy r2:proto-data/20946_humos_offset/ /workspace/stage2_data/ --transfers=8 --progress

# Create slurmrank.pt pointing to all shards
# (adapt tools/merge_motion_shards.py or write a simple pointer script)
# Then point training at the slurmrank file:
python protomotions/train_agent.py \
    --motion-file /workspace/stage2_data/stage2_slurmrank.pt \
    ...
```

---

## Scale Reference

| Dataset | Clips | Shapes | Motions | Size |
|---------|-------|--------|---------|------|
| Pilot (Stage 2 baseline) | 1,024 | 128 | 131,072 | ~54 GB |
| Stage 1 neutral | 20,946 | 1 | 20,946 | ~16 GB |
| **Stage 2 (this pipeline)** | **20,946** | **128** | **2,681,088** | **~1.1 TB** |



python tools/prepare_stage2_data.py \
    --valid-ids /home/hlz/repos/hhi/data-processing/valid_ids_sorted_by_difficulty.txt \
    --workspace /media/hlz/R/stage2_prep \
    --proto-root /home/hlz/repos/ProtoMotions \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --output-dir /media/hlz/R/stage2_data \
    --local-humos-cache /media/hlz/R/humos_output \
    --batch-clips 512 \
    --isaacgym-env isaacgym \
    --device cuda
