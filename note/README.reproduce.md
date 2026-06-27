# Data Pipeline — Full Reproduction Guide

This document is the single authoritative reference for reproducing all datasets from scratch.
Development history and experiment notes live in `README.note.md`; this file is a clean checklist.

---

## Prerequisites

| Repo / resource | Path | Notes |
|---|---|---|
| ProtoMotions | `/home/hlz/repos/ProtoMotions` | This repo |
| HUMOS | `/home/hlz/repos/humos` | Motion generation model |
| SMPLSim | `/home/hlz/repos/SMPLSim` | SMPL MJCF generation |
| HUMOS checkpoint | `/home/hlz/repos/humos/logs/humos/q6zbv2tu/checkpoints/latest-epoch=1599.ckpt` | Trained diffusion model |
| HUMOS 3D features | `/home/hlz/repos/humos/datasets/humos3dfeats/` | 2.7 GB, `.tensor` files for all HumanML3D clips |
| all_betas.pt | `/home/hlz/repos/humos/all_betas.pt` | 64 unique beta vectors for 128-shape training set |
| Conda envs | `isaacgym`, `smplsim` | IsaacGym runs in `isaacgym`; SMPLSim/HUMOS in `smplsim` |

---

## R2 Storage Status

All permanent/large data lives at `r2:proto-data/`. Run `rclone ls r2:proto-data/ 2>&1` to verify.

| R2 path | Size | Contents | Status |
|---|---|---|---|
| `merged4/humos_{0-3}.pt` | 57 GB | **Pilot** training data (1024 clips × 128 shapes, merged shards) | ✓ uploaded |
| `20946_neutral_offset/humanml3d_neutral_20946_000X.pt` | 16 GB | Stage 1 training data (20,946 neutral motions, 6 shards) | ✓ uploaded |
| `hhi_stage2/` | ~1.1 TB projected | **Stage 2** full training data (20,946 clips × 128 shapes) | In progress — see `README.stage2-data-pipeline.md` |
| `difficult-motions/failed_clips.pt` | 10 GB | Pilot hard-clip fine-tune set (192 clips × 128 shapes) | ✓ uploaded |
| `ckpt/hhi_1024_transfer.zip` | 932 MB | Pilot transfer checkpoint (raw betas) — ablation reference | ✓ uploaded |
| `ckpt/hhi_1024_phy_transfer.zip` | 1.1 GB | Pilot transfer checkpoint (physics features) — ablation reference | ✓ uploaded |
| `ckpt/20951_neutral.zip` | 117 MB | Old 199-epoch neutral checkpoint (pre-fix, **do not** use for Stage 1 warm-start) | ✓ uploaded |
| `humos_output/` | 36 GB | HUMOS inference output for 128 shapes × 1024 clips | **NOT UPLOADED** |
| `humos_output/interp/` | (part of above) | HUMOS inference for 16 interp held-out betas (717 files) | **NOT UPLOADED** |

### Files to upload to R2

```bash
# HUMOS inference output — high GPU-cost to regenerate, upload once
rclone copy /home/hlz/datasets/humos_output/ \
    r2:proto-data/humos_output/ \
    --transfers=4 --s3-upload-concurrency=4 --s3-chunk-size=64M \
    --retries=10 --retries-sleep=30s --low-level-retries=20 --progress
```

---

## Pipeline A — Main Training Dataset (1024 clips × 128 shapes)

Final output on R2: `merged4/humos_{0-3}.pt` + `humos_slurmrank.pt` (on RunPod)

### A1 — Generate SMPL assets (in-repo, already done)

```bash
# Run in smplsim env from SMPLSim repo
cd /home/hlz/repos/SMPLSim
conda run -n smplsim python run.py   # generates 128 XMLs + all_betas.pt

# Back in ProtoMotions root
python tools/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor \
    --betas-file protomotions/data/assets/all_betas.pt \
    --out protomotions/data/assets/mjcf/smpl_mor/assets.yaml
```

Output (committed to repo): `protomotions/data/assets/mjcf/smpl_mor/*.xml` + `assets.yaml`

### A2 — HUMOS inference (36 GB output)

```bash
cd /home/hlz/repos/humos
conda run -n smplsim python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas.pt \
    --local-out-dir /home/hlz/datasets/humos_output
```

- Input: `humos3dfeats/*.tensor` (20,951 keyids from `valid_sorted.json`)
- Output: `/home/hlz/datasets/humos_output/{keyid}.pt` — one file per clip, each containing 128 shape variants
- 18 AMASS sub-datasets (see `README.humos-data.md`); treadmill/skate clips pre-filtered

### A3 — Export to AMASS NPZ (63 GB intermediate)

```bash
cd /home/hlz/repos/ProtoMotions
python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/ \
    --out-root /home/hlz/datasets/humos_proto_interm/ \
    --skip-existing
```

Output: `humos_proto_interm/HUMOS/*.npz` + `humos_131072.yaml`

### A4 — Convert to MotionLib .pt (54 GB offset shards)

```bash
# Step 4a: Convert NPZ → MotionLib chunks
python tools/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/humos_proto_interm/ \
    /home/hlz/datasets/humos_proto/ \
    --motion-config /home/hlz/datasets/humos_proto_interm/humos_131072.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cuda \
    --batch-size 8192 \
    --force-remake

# Step 4b: Frame-0 grounding offset (IsaacGym, run all shards)
tools/run_frame0_offsets.sh
```

Output: `humos_proto/offset/humos_131072_000X_offset.pt` (16 shards, ~54 GB)

### A5 — Merge and upload to RunPod

```bash
# On RunPod: merge offset shards into 4 large files + create slurmrank
python tools/merge_motion_shards.py

# Upload merged shards back to R2
rclone copy /workspace/merged4/ r2:proto-data/merged4/ --progress
```

Output on R2: `merged4/humos_{0-3}.pt` (57 GB) + `humos_slurmrank.pt` on RunPod

---

## Pipeline B — Stage 1 Neutral Dataset (20,946 motions)

Final output on R2: `20946_neutral_offset/humanml3d_neutral_20946_000X.pt` (6 shards, 16 GB)

### B1 — Generate neutral SMPL assets (in-repo, already done)

```bash
cd /home/hlz/repos/SMPLSim
conda run -n smplsim python run_neutral.py
# Generates: protomotions/data/assets/mjcf/smpl_mor_neutral/{male,female}_neutral_smpl.xml

cd /home/hlz/repos/ProtoMotions
python tools/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor_neutral \
    --betas-file protomotions/data/assets/all_betas_neutral.pt \
    --out protomotions/data/assets/mjcf/smpl_mor_neutral/assets.yaml
```

Also adds `neutral_neutral_smpl.xml` manually (betas=0, gender=neutral).
Adds `smpl_mor_neutral` entry to `protomotions/robot_configs/factory.py`.

### B2 — Export neutral NPZ (18 GB)

```bash
cd /home/hlz/repos/ProtoMotions
python tools/export_tensor_to_amass_npz.py \
    --out-root /home/hlz/datasets/amass_neutral \
    --skip-existing
```

- Source: `humos/datasets/humos3dfeats/*.tensor` — betas zeroed out, poses kept
- Output: `amass_neutral/HML3D/{keyid}_v00_{gender}_neutral.npz` (20,951 files) + `humanml3d_neutral_20951.yaml`

### B3 — Convert to MotionLib .pt (15 GB raw, then 15 GB offset)

```bash
# Step 3a: Convert NPZ → MotionLib chunks (batch-size ≤ 4096 avoids ZIP64 issue)
python tools/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/amass_neutral \
    /home/hlz/datasets/humos_proto_neutral \
    --motion-config /home/hlz/datasets/amass_neutral/humanml3d_neutral_20951.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cpu \
    --batch-size 4096
```

Output: `humos_proto_neutral/humanml3d_neutral_20951_000X.pt` (6 chunks, mixed female_neutral/male_neutral labels — intermediate only, do not use for training)

```bash
# Step 3b: Frame-0 grounding offset (IsaacGym, reads 20951 raw chunks)
tools/run_frame0_offsets_neutral.sh
```

Output: `humos_proto_neutral/humanml3d_neutral_20951_000X_offset.pt` (6 chunks — intermediate)

```bash
# Step 3c: Equalize shards — drops clips that failed grounding, rebalances so all
#          6 output files have equal clip counts; renames output to reflect true count
python tools/equalize_slurmrank_files.py \
    --input-dir /home/hlz/datasets/humos_proto_neutral \
    --output-dir /home/hlz/datasets/humos_proto_neutral/offset \
    --base-name humanml3d_neutral_20951
```

Output: `humos_proto_neutral/offset/humanml3d_neutral_20946_000X.pt`
- 20,951 → 20,946 clips (5 dropped; name encodes the final count)
- Note: `--base-name` refers to the *input* files (`20951_*_offset.pt`); the output name is auto-derived from the actual clip count

```bash
# Step 3d: Fix asset IDs — overwrites mixed female_neutral/male_neutral labels with neutral_neutral
python tools/fix_neutral_asset_ids.py \
    --dir /home/hlz/datasets/humos_proto_neutral/offset \
    --base-name humanml3d_neutral_20946
```

Final output: `humos_proto_neutral/offset/humanml3d_neutral_20946_000X.pt`
- All `asset_id = neutral_neutral`, `gender_id = 0`, `betas = 0` ✓

### B4 — Upload to R2 (already done)

```bash
rclone copy /home/hlz/datasets/humos_proto_neutral/offset/ \
    r2:proto-data/20946_neutral_offset/ \
    --transfers=2 --s3-upload-concurrency=4 --s3-chunk-size=64M \
    --retries=10 --retries-sleep=30s --low-level-retries=20 --progress
```

### B5 — Download to RunPod and create slurmrank

```bash
# On RunPod: pull from R2
rclone copy r2:proto-data/20946_neutral_offset/ /workspace/20946_neutral_offset/ --progress

# Also rsync the neutral XMLs
rsync -avz protomotions/data/assets/mjcf/smpl_mor_neutral/ \
    runpod:/workspace/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor_neutral/

# Create slurmrank file (points each rank to its chunk)
# → humanml3d_neutral_20946_slurmrank.pt (on RunPod)
```

---

## Pipeline D — Stage 2 Full Dataset (20,946 clips × 128 shapes ≈ 1.1 TB)

Final output on R2: `hhi_stage2/batch_NNNN_MMMM_offset.pt` (~335 shards × 3.4 GB)

See `README.stage2-data-pipeline.md` for the full pipeline spec and run commands.

```bash
# Local (R drive), resumable:
python tools/prepare_stage2_data.py \
    --valid-ids /home/hlz/repos/hhi/data-processing/valid_ids_sorted_by_difficulty.txt \
    --workspace /media/hlz/R/stage2_prep \
    --proto-root /home/hlz/repos/ProtoMotions \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --output-dir /media/hlz/R/stage2_data \
    --local-humos-cache /media/hlz/R/humos_output \
    --batch-clips 512 --isaacgym-env isaacgym --device cuda
```

Progress is tracked in `{workspace}/pipeline_log.txt` and re-run resumes automatically.

---

## Pipeline C — Held-out Beta Evaluation (E7 generalization)

Final output: `heldout_interp_offset.pt` + `heldout_extrap_offset.pt` (to RunPod for eval)

Assets (in-repo, already done):
- `protomotions/data/assets/all_betas_interp.pt` — 16 betas, seed=99, range=[-3, 3]
- `protomotions/data/assets/all_betas_extrap.pt` — 16 betas, seed=99, range=[-5, 5]
- `protomotions/data/assets/mjcf/smpl_mor_interp/` and `smpl_mor_extrap/` — XMLs + assets.yaml

### C1 — HUMOS inference

```bash
cd /home/hlz/repos/humos
# interp (DONE — 717 files at /home/hlz/datasets/humos_output/interp/)
conda run -n smplsim python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas_interp.pt \
    --local-out-dir /home/hlz/datasets/humos_output/interp

# extrap (TODO)
conda run -n smplsim python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas_extrap.pt \
    --local-out-dir /home/hlz/datasets/humos_output/extrap
```

### C2 — Export to AMASS NPZ

```bash
cd /home/hlz/repos/ProtoMotions
python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/interp \
    --out-root /home/hlz/datasets/amass_heldout/interp \
    --genders male female --apply-offset-height --skip-existing --fps 30.0

python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/extrap \
    --out-root /home/hlz/datasets/amass_heldout/extrap \
    --genders male female --apply-offset-height --skip-existing --fps 30.0
```

### C3 — Convert to MotionLib .pt

```bash
python tools/convert_amass_to_motionlib_with_morphology.py \
    --motion-yaml /home/hlz/datasets/amass_heldout/interp/humos_*.yaml \
    --assets-yaml protomotions/data/assets/mjcf/smpl_mor_interp/assets.yaml \
    --out /home/hlz/datasets/heldout_interp.pt

python tools/convert_amass_to_motionlib_with_morphology.py \
    --motion-yaml /home/hlz/datasets/amass_heldout/extrap/humos_*.yaml \
    --assets-yaml protomotions/data/assets/mjcf/smpl_mor_extrap/assets.yaml \
    --out /home/hlz/datasets/heldout_extrap.pt
```

### C4 — Frame-0 grounding offset

```bash
python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/heldout_interp.pt \
    --asset-root protomotions/data/assets/mjcf/smpl_mor_interp \
    --out-motion-file /home/hlz/datasets/heldout_interp_offset.pt --overwrite

python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/heldout_extrap.pt \
    --asset-root protomotions/data/assets/mjcf/smpl_mor_extrap \
    --out-motion-file /home/hlz/datasets/heldout_extrap_offset.pt --overwrite
```

### C5 — Upload to RunPod for evaluation

```bash
rsync -avz /home/hlz/datasets/heldout_interp_offset.pt runpod:/workspace/
rsync -avz /home/hlz/datasets/heldout_extrap_offset.pt runpod:/workspace/
```

See `README.heldout-pipeline.md` for the IsaacGym eval commands.

---

## Local Large Files — Status Summary

| Local path | Size | On R2? | Action |
|---|---|---|---|
| `datasets/humos_output/` (main) | ~29 GB | **No** | Upload to `r2:proto-data/humos_output/` |
| `datasets/humos_output/interp/` | ~7 GB | **No** | Upload (included above) |
| `datasets/humos_output/extrap/` | — | **No** | Upload after extrap inference |
| `datasets/humos_proto/offset/` | 54 GB | No (merged4 covers it) | Keep locally for inference; skip R2 |
| `datasets/humos_proto_interm/` | 63 GB | No | Intermediate only — safe to delete after verifying merged4 |
| `datasets/humos_proto_neutral/` (raw 20951) | 15 GB | No | Intermediate only — offset shards are on R2 |
| `datasets/humos_proto_neutral/offset/` | 15 GB | **Yes** (`20946_neutral_offset/`) | No action needed |
| `datasets/amass_neutral/HML3D/` | 18 GB | No | Reproducible from humos3dfeats (2.7 GB) — skip R2 |
| `repos/humos/datasets/humos3dfeats/` | 2.7 GB | **No** | Small but critical source — consider uploading |
