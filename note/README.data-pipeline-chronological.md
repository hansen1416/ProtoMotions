# Data Preparation — Chronological Record

Everything in the order it actually happened, from raw AMASS download to RunPod training data.

---

## Phase 0 — Prerequisites

Three repos involved:

| Repo | Path | Purpose |
|---|---|---|
| ProtoMotions | `/home/hlz/repos/ProtoMotions` | RL training framework (this repo) |
| HUMOS | `/home/hlz/repos/humos` | Diffusion model for shape-conditioned motion retargeting |
| SMPLSim | `/home/hlz/repos/SMPLSim` | Generates per-shape MJCF `.xml` files and `all_betas.pt` |

Conda environments:
- `isaacgym` — used for IsaacGym-dependent steps (frame-0 grounding offsets)
- `smplsim` — used for HUMOS inference, SMPLSim XML generation, and NPZ conversion

---

## Phase 1 — Install HUMOS

HUMOS is an ECCV 2024 paper (Tripathi et al.) that generates body-shape-conditioned motion from SMPL pose sequences.
The repo lives at `/home/hlz/repos/humos` (forked from `sha2nkt/humos_website_backend`).

```bash
cd /home/hlz/repos/humos
conda create -n smplsim python=3.10
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
cd aitviewer_humos && pip install -e . && cd ..
pip install -r requirements.txt
pip install -e .
```

Download the pretrained HUMOS checkpoint:

```bash
sh fetch_data.sh
# Downloads logs/humos/q6zbv2tu/checkpoints/latest-epoch=1599.ckpt   (autoencoder init)
#           logs/humos/5bhgscl8/checkpoints/latest-epoch=199.ckpt     (final model)
```

The checkpoint used for all our inference is the **autoencoder** one:
`logs/humos/q6zbv2tu/checkpoints/latest-epoch=1599.ckpt`

Also download SMPL body models manually from https://smpl.is.tue.mpg.de and place under
`body_models/smpl/` (female, male, neutral `.pkl`/`.npz` files).

---

## Phase 2 — Download AMASS Sub-datasets

AMASS is a unified SMPL+H motion capture dataset. HUMOS uses **18 of the ~23 available sub-datasets**.

Download the **SMPL+H G** versions of the following from https://amass.is.tue.mpg.de/download.php
and unzip into `/home/hlz/repos/humos/datasets/amass_data/`:

```
ACCAD              BioMotionLab_NTroje    BMLhandball    BMLmovi
CMU                DFaust_67              EKUT           Eyes_Japan_Dataset
HumanEva           KIT                    MPI_HDM05      MPI_Limits
MPI_mosh           SFU                    SSM_synced     TCD_handMocap
TotalCapture       Transitions_mocap
```

Five sub-datasets that are present in `amass_data/` but were **not used**:

| Excluded | Reason |
|---|---|
| `DanceDB` | Not in HUMOS processing list |
| `GRAB` | Not in HUMOS processing list |
| `HUMAN4D` | Not in HUMOS processing list |
| `SOMA` | Not in HUMOS processing list |
| `WEIZMANN` | Not in HUMOS processing list |

---

## Phase 3 — Process Raw AMASS → `pose_data/`

Run the HUMOS notebook to extract per-frame SMPL poses from all 18 sub-datasets,
resampled to **20 fps**, with Z-up → Y-up axis conversion:

```bash
cd /home/hlz/repos/humos
# run all cells of:
jupyter nbconvert --to notebook --execute humos/prepare/raw_pose_processing_humos.ipynb
```

What it does:
- Loads each `.npz` from `datasets/amass_data/`
- Runs SMPL+H forward kinematics at 20 fps using `human_body_prior.BodyModel`
- Saves per-clip `.npy` (joint positions) + `.npz` (full pose data) under `datasets/pose_data/`
- 14,055 source clips → `pose_data/` with identical sub-dataset folder structure

---

## Phase 4 — Clean Treadmill and Skating Clips

```bash
cd /home/hlz/repos/humos
conda run -n smplsim python humos/prepare/clean_amass_data.py \
    --data datasets/pose_data \
    --backup datasets/pose_data_backup
```

Removes two categories of physically degenerate clips:
- **`BioMotionLab_NTroje`** — treadmill (`*_treadmill_*`) and "normal" walk (`*_normal_*`) clips moved to backup
- **`MPI_HDM05`** — inline skating sequences (`HDM_dg_07-01*`) moved to backup

Clips are moved (not deleted) to `datasets/pose_data_backup/` for recovery.

---

## Phase 5 — Extract HUMOS 3D Features

```bash
conda run -n smplsim python humos/prepare/compute_3dfeats.py --fps 20
```

Converts each cleaned `pose_data/*.npz` into a `.tensor` file containing the HumanML3D feature
representation (263-dim: root data, RIC joint positions, rotations, velocities, foot contact flags).
Also generates a mirrored version (swap left/right) for each clip.

Output: `datasets/humos3dfeats/` — 2.7 GB, one `.tensor` per clip.
This is the direct input to HUMOS inference (not the raw NPZs).

Additional prep steps run once:
```bash
# process text annotation paths
python humos/prepare/process_text_annotations.py
# compute dataset feature mean/std for normalization
python humos/prepare/motion_stats.py
```

---

## Phase 6 — Generate 64 Body Shape Betas

Body shape diversity for the training set is defined by **64 unique SMPL beta vectors** (10-dim each),
sampled uniformly in `[-3.0, 3.0]` with `numpy.random.default_rng(seed=46)`.

```bash
cd /home/hlz/repos/SMPLSim
conda run -n smplsim python run.py
```

`run.py` calls `sample_betas_uniform(batch_size=64, low=-3.0, high=3.0, rng=np.random.default_rng(46))`
and saves:
- `/home/hlz/repos/humos/all_betas.pt` — dict of 64 entries `{hex4_key: tensor(10)}`
- `/home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas.pt` — same file, copy for ProtoMotions

Each beta vector gets a deterministic 4-byte hex key (e.g., `0e26b88d`) derived from the same RNG,
used as a consistent identifier across all downstream files.

---

## Phase 7 — Generate SMPL MJCF XMLs (128 humanoids)

Still in SMPLSim, `run.py` iterates over the 64 betas × 2 genders (male, female):

```bash
# run.py already does this in the same invocation as Phase 6
# generates 128 XML files into ProtoMotions
```

For each `(gender, beta_key)` pair, `generate_yaml()` calls `SMPL_Robot.load_from_skeleton(betas, gender)`
and writes a MuJoCo MJCF XML to:
`/home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor/{gender}_{hex4_key}_smpl.xml`

Key robot config flags: `real_weight=True`, `replace_feet=True`, `remove_toe=True`, `freeze_hand=True`.
Joint gains follow the `GAINS_PHC` table (stiffness 800–1000, damping 80–100 for large joints).

Then generate the asset manifest:

```bash
cd /home/hlz/repos/ProtoMotions
conda run -n smplsim python tools/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor \
    --betas-file protomotions/data/assets/all_betas.pt \
    --out protomotions/data/assets/mjcf/smpl_mor/assets.yaml
```

Output (committed to repo): 128 `.xml` files + `assets.yaml` with per-asset beta/gender metadata.

---

## Phase 8 — HUMOS Inference: Main Training Set (22,459 clips × 128 shapes)

HUMOS takes each motion clip from `humos3dfeats/` and regenerates it for each of the 128 body shapes
using the diffusion model checkpoint. This is the most GPU-expensive step.

```bash
cd /home/hlz/repos/humos
conda run -n smplsim python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas.pt \
    --local-out-dir /home/hlz/datasets/humos_output
```

What it does:
- Iterates over all keyids in the full HumanML3D set (including mirrored clips)
- For each clip, loops over `for gender in [-1, 1]` (female, male) × 64 betas = **128 shapes**
- Runs the HUMOS diffusion forward pass conditioned on the clip's 3D features + body shape
- Saves one `.pt` file per clip: `humos_output/{keyid}.pt`
  Each file contains 128 retargeted motion variants as SMPL pose sequences

Output: **22,459 files** — the full HumanML3D set (originals + mirrored).
Authoritative copy on Google Drive: `gdrive:humos_output/` (flat directory, no subdirs).
Confirmed count: `rclone lsf gdrive:humos_output/ --files-only | wc -l` → **22,459**.
Local partial copy may exist at `/home/hlz/datasets/humos_output/`.

## Phase 8b — HUMOS Inference: Held-out Beta Sets (interp / extrap)

Two additional inference runs with **different beta files** to generate data for out-of-distribution
generalization evaluation. Each uses 16 betas × 2 genders = **32 shapes** per clip.

### Bug fix required before running

The original `infer.py` skip-existing logic checks `gdrive:humos_output/` unconditionally even when
`--local-out-dir` is given. Since all 22,459 keyids already exist on gdrive from the main run,
it skips everything immediately. Fix applied directly to `humos/infer.py`:

1. Remote cache is now built **once before the loop**, and **only** when no `--local-out-dir` is set.
2. When `--local-out-dir` is set, skip check is against the **local file** only.
3. The redundant existence check in the save block is removed (skip already happened at loop top).

The key restructure in `run_inference()`:

```python
# Before the loop — only in remote mode
if local_out_dir is None:
    existing_remote_names = build_remote_name_cache(RCLONE_MOUNT_ROOT, REMOTE_INDEX_CACHE, ...)

for _, batch in enumerate(tqdm(dataloader, ...)):
    remote_name = f"{batch['keyid'][0]}.pt"

    if local_out_dir is not None:
        local_path = os.path.join(local_out_dir, remote_name)
        if os.path.exists(local_path):
            print(f"Skip existing: {local_path}")
            continue
    else:
        remote_path = f"{RCLONE_REMOTE_DIR}/{remote_name}"
        if remote_name in existing_remote_names:
            print(f"Skip existing remote (cached): {remote_path}")
            continue

    # ... inference ...

    if local_out_dir is not None:
        torch.save(motion_out, local_path)
    else:
        save_torch_to_rclone(motion_out, remote_path)
        ...
```

### Commands

Output goes to the portable hard drive (`/media/hlz/R/`). Each run ~200 GB, ~4 hours on one GPU.
Safe to interrupt and resume — skip logic checks per-file on disk.

```bash
cd /home/hlz/repos/humos

# interp — 16 betas, seed=99, range [-3, 3]  (within training distribution)
conda run -n smplsim python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas_interp.pt \
    --local-out-dir /media/hlz/R/humos_output/interp

# extrap — 16 betas, seed=99, range [-5, 5]  (outside training distribution)
# run after interp finishes (single GPU)
conda run -n smplsim python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas_extrap.pt \
    --local-out-dir /media/hlz/R/humos_output/extrap
```

Status:
- **interp**: **in progress** — started 2026-06-21, ETA ~4 hours, saving to `/media/hlz/R/humos_output/interp/`
- **extrap**: **TODO** — run after interp completes

Output per run: one `.pt` per clip (22,459 total), each containing 32 shape variants.
These are **not** in `gdrive:humos_output/` (which is flat, main-run only).

---

## Phase 9 — Semantic Filtering: Remove Clips Requiring External Objects

After HUMOS inference, the full 22,459-clip HumanML3D set was filtered to remove motions that
cannot be physically reproduced in an empty flat-ground simulation (no chairs, stairs, walls, etc.).

This work was done in the predecessor project at `/home/hlz/repos/hhi/data-processing/`.

### Method

Text annotations from `motion_id_text.json` (natural-language descriptions of each clip, e.g.,
"a person sits down on a chair") were reviewed and each invalid clip was assigned to one of
9 semantic categories:

| Category | Count |
|---|---|
| `seat_support` (chairs, benches, stools) | 657 |
| `terrain_or_structure` (stairs, ramps, platforms) | 198 |
| `other_person_or_animal` (partner/animal interaction) | 187 |
| `table_shelf_surface` (leaning on, placing objects) | 174 |
| `wall_door_window` (pushing, opening) | 111 |
| `obstacle_or_gap` (jumping over, crawling under) | 83 |
| `external_support` (poles, handles, rails) | 53 |
| `forceful_object_interaction` (hitting, throwing) | 32 |
| `environment_medium` (swimming, wading) | 13 |
| **Total removed** | **1,508** |

Note: IDs prefixed with `M` are the mirrored (left/right flipped) versions of the same clip;
both the original and its mirror are labelled and filtered together.

### Output files

| File | Content |
|---|---|
| `invalid_categories.txt` | 1,508 lines: `{keyid}: {category}` |
| `invalid_category_counts.txt` | Per-category totals |
| `invalid_motions.txt` | 1,508 invalid keyids (no category) |
| `valid_motions.txt` | **20,951** valid keyids + text descriptions |
| `valid_sorted.json` | Same 20,951 clips, sorted by kinematic difficulty score |
| `valid_ids_sorted_by_difficulty.txt` | Plain text version of the above |
| `difficulty_scores.json` | Full per-clip difficulty breakdown (all 4 sub-scores + raw_score) |

### Difficulty scoring (`hhi/scripts/compute_difficulty_score.py`)

Each valid clip is assigned a scalar difficulty score based on kinematic features extracted from
the HUMOS inference output PKL files (source data at `/media/hlz/R/humos_phc_results`):

```
score = 0.4 × max_root_horizontal_velocity
      + 0.3 × flight_ratio          (fraction of frames airborne)
      + 0.2 × max_dof_velocity
      + 0.1 × kinetic_energy_variance
```

Weights follow the principle of the MDS paper (arXiv:2512.07248). The 20,951 valid clips are
sorted ascending by this score → easiest first, hardest last.

**Result**: 22,459 − 1,508 = **20,951 valid clips**, sorted by difficulty, stored in
`valid_sorted.json`. This file is what determines which HUMOS outputs are used for training.

---

## Phase 10 — Export HUMOS Output → AMASS-style NPZ

Converts the HUMOS `.pt` files (SMPL pose tensors) into AMASS-compatible `.npz` files,
one NPZ per `(clip, gender, beta_key)` triple.

```bash
cd /home/hlz/repos/ProtoMotions
python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/ \
    --out-root /home/hlz/datasets/humos_proto_interm/ \
    --skip-existing
```

Output:
- `humos_proto_interm/HUMOS/{keyid}_v{idx:02d}_{gender}_{hex4_key}.npz` — one file per shape variant
- `humos_proto_interm/humos_131072.yaml` — motion manifest listing all 131,072 NPZ paths
  (1,024 selected clips × 128 shapes = 131,072; not all 20,951 clips are used for training)
- Intermediate only — safe to delete after downstream `.pt` files are verified

---

## Phase 11 — Convert NPZ → MotionLib `.pt` Chunks

Converts the 131,072 NPZs into ProtoMotions' binary MotionLib format.
Processes in batches to stay within GPU/RAM limits.

```bash
python tools/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/humos_proto_interm/ \
    /home/hlz/datasets/humos_proto/ \
    --motion-config /home/hlz/datasets/humos_proto_interm/humos_131072.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cuda \
    --batch-size 8192 \
    --force-remake
```

Output: `humos_proto/humos_131072_{chunk_idx:04d}.pt` — 16 shards, ~54 GB total.
Each shard contains tensors: `gts`, `grs`, `gvs`, `gavs`, `dps`, `dvs`, `lrs`, `contacts`,
`motion_betas`, `motion_gender_ids`, `motion_asset_ids`, `motion_clip_ids`, etc.

Note: output fps is **30** (upsampled from HUMOS's 20 fps) to match ProtoMotions training convention.

---

## Phase 12 — Frame-0 Grounding Offset (IsaacGym)

Each motion clip's first frame may spawn the character partially underground (HUMOS does not
guarantee floor alignment). This step shifts `gts[:, :, 2]` so the lowest collision point at
frame 0 is exactly 5 mm above the ground plane.

```bash
# Runs all 16 shards via IsaacGym (isaacgym conda env)
tools/run_frame0_offsets.sh
```

Each shard call looks like:
```bash
conda run -n isaacgym python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/humos_proto/humos_131072_{N:04d}.pt \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --out-motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_{N:04d}_offset.pt \
    --limit -1 --overwrite
```

Algorithm: creates one IsaacGym env per unique SMPL shape (~128 envs), runs FK at frame 0,
finds the lowest collision geometry point, computes a rigid Z-offset per clip, applies it to all frames.

Output: `humos_proto/offset/humos_131072_{N:04d}_offset.pt` — 16 offset shards, ~54 GB.

---

## Phase 13 — Merge Shards and Upload to RunPod

On RunPod (GPU cloud, for training), merge the 16 offset shards into 4 large files
to reduce file-open overhead during training:

```bash
# On RunPod
python tools/merge_motion_shards.py
# → merged4/humos_{0-3}.pt   (~57 GB total)

# Create slurmrank pointer file
# → humos_slurmrank.pt   (maps each data-parallel rank to its shard)

# Upload merged shards to R2 for persistence
rclone copy /workspace/merged4/ r2:proto-data/merged4/ --progress
```

Final training data on R2: `r2:proto-data/merged4/humos_{0-3}.pt` (57 GB) ✓

---

## Phase 14 — Stage 1 Neutral Dataset (Parallel Track)

To enable warm-starting Stage 2 training, a **neutral-shape** dataset was built from the same
HumanML3D clips, using zeroed betas (mean SMPL body) and all three genders.

### B1 — Generate neutral XMLs

```bash
cd /home/hlz/repos/SMPLSim
conda run -n smplsim python run_neutral.py
# → protomotions/data/assets/mjcf/smpl_mor_neutral/male_neutral_smpl.xml
# → protomotions/data/assets/mjcf/smpl_mor_neutral/female_neutral_smpl.xml
# neutral_neutral_smpl.xml added manually (betas=0, gender=neutral)

cd /home/hlz/repos/ProtoMotions
python tools/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor_neutral \
    --betas-file protomotions/data/assets/all_betas_neutral.pt \
    --out protomotions/data/assets/mjcf/smpl_mor_neutral/assets.yaml
```

### B2 — Export neutral NPZ

Reads the same `humos3dfeats/*.tensor` files as Phase 5, but zeroes out betas:

```bash
python tools/export_tensor_to_amass_npz.py \
    --out-root /home/hlz/datasets/amass_neutral \
    --skip-existing
```

Output: `amass_neutral/HML3D/{keyid}_v00_{gender}_neutral.npz` (20,951 files) +
`humanml3d_neutral_20951.yaml`

### B3 — Convert NPZ → MotionLib `.pt`

```bash
# batch-size ≤ 4096 avoids ZIP64 Python bug
python tools/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/amass_neutral \
    /home/hlz/datasets/humos_proto_neutral \
    --motion-config /home/hlz/datasets/amass_neutral/humanml3d_neutral_20951.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cpu \
    --batch-size 4096
```

Output: `humos_proto_neutral/humanml3d_neutral_20951_000X.pt` (6 chunks)

### B4 — Frame-0 grounding offset

```bash
tools/run_frame0_offsets_neutral.sh
```

Output: `humos_proto_neutral/humanml3d_neutral_20951_000X_offset.pt` (6 chunks)

### B5 — Equalize shards (drop failed clips, rebalance)

```bash
python tools/equalize_slurmrank_files.py \
    --input-dir /home/hlz/datasets/humos_proto_neutral \
    --output-dir /home/hlz/datasets/humos_proto_neutral/offset \
    --base-name humanml3d_neutral_20951
```

5 clips failed grounding → 20,951 → **20,946** clips.
Output: `humos_proto_neutral/offset/humanml3d_neutral_20946_000X.pt`

### B6 — Fix asset IDs

The 6 output shards had mixed `female_neutral`/`male_neutral` asset labels.
Overwrite all to `neutral_neutral`:

```bash
python tools/fix_neutral_asset_ids.py \
    --dir /home/hlz/datasets/humos_proto_neutral/offset \
    --base-name humanml3d_neutral_20946
```

### B7 — Upload to R2

```bash
rclone copy /home/hlz/datasets/humos_proto_neutral/offset/ \
    r2:proto-data/20946_neutral_offset/ \
    --transfers=2 --s3-upload-concurrency=4 --s3-chunk-size=64M \
    --retries=10 --retries-sleep=30s --low-level-retries=20 --progress
```

Final on R2: `r2:proto-data/20946_neutral_offset/humanml3d_neutral_20946_000X.pt` (6 shards, 16 GB) ✓

---

## Phase 15 — Held-out Beta Evaluation Sets (E7 Generalization)

Two sets of 16 body shapes were generated for evaluating generalization to **unseen** betas:
- **interp** — 16 betas, seed=99, range `[-3, 3]` (within training distribution)
- **extrap** — 16 betas, seed=99, range `[-5, 5]` (outside training distribution)

Assets already committed to repo:
- `protomotions/data/assets/all_betas_interp.pt` / `all_betas_extrap.pt`
- `protomotions/data/assets/mjcf/smpl_mor_interp/` / `smpl_mor_extrap/` + `assets.yaml`

### C1 — HUMOS inference for held-out betas

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

### C2 — Export to NPZ

```bash
python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/interp \
    --out-root /home/hlz/datasets/amass_heldout/interp \
    --genders male female --apply-offset-height --skip-existing --fps 30.0

python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/extrap \
    --out-root /home/hlz/datasets/amass_heldout/extrap \
    --genders male female --apply-offset-height --skip-existing --fps 30.0
```

### C3 — Convert to MotionLib `.pt`

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

---

## Summary: Data Flow

```
AMASS website (SMPL+H G, 18 sub-datasets)
        ↓  raw_pose_processing_humos.ipynb    (resampled 20fps, Y-up)
datasets/pose_data/   (~14,055 clips)
        ↓  clean_amass_data.py                (remove treadmill, skating)
datasets/pose_data/   (cleaned)
        ↓  compute_3dfeats.py                 (HumanML3D 263-dim features)
datasets/humos3dfeats/   (2.7 GB, 20,951 valid keyids)
        ↓  infer.py  ×  128 body shapes       (HUMOS diffusion model, q6zbv2tu ckpt)
datasets/humos_output/   (~36 GB, one .pt per clip)
        ↓  export_humos_to_amass_npz.py       (1024 clips × 128 shapes)
datasets/humos_proto_interm/HUMOS/   (131,072 .npz + manifest .yaml, 63 GB)
        ↓  convert_amass_to_motionlib_with_morphology.py   (30fps, cuda, batch 8192)
datasets/humos_proto/   (16 .pt shards, 54 GB)
        ↓  compute_humos_frame0_offsets.py    (IsaacGym, per-shape grounding)
datasets/humos_proto/offset/   (16 offset shards, 54 GB)
        ↓  merge_motion_shards.py  [on RunPod]
merged4/humos_{0-3}.pt   (57 GB, on R2 and RunPod)
```

---

## R2 Storage Status

| R2 path | Size | Contents | Status |
|---|---|---|---|
| `merged4/humos_{0-3}.pt` | 57 GB | Stage 2 training data (1024 clips × 128 shapes) | ✓ uploaded |
| `20946_neutral_offset/humanml3d_neutral_20946_000X.pt` | 16 GB | Stage 1 training data (20,946 neutral motions) | ✓ uploaded |
| `difficult-motions/failed_clips.pt` | 10 GB | Hard-clip fine-tune set (192 clips × 128 shapes) | ✓ uploaded |
| `ckpt/hhi_1024_transfer.zip` | 932 MB | Stage 2 transfer checkpoint | ✓ uploaded |
| `ckpt/hhi_1024_phy_transfer.zip` | 1.1 GB | Physics-feature transfer checkpoint | ✓ uploaded |
| `humos_output/` | 36 GB | HUMOS inference output (128 shapes × 1024 clips) | **NOT uploaded** |
| `humos_output/interp/` | ~7 GB | HUMOS inference for 16 interp held-out betas | **NOT uploaded** |
