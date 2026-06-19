# Held-out Beta Pipeline: Commands to Run

Steps to produce the held-out interp/extrap MotionLib `.pt` files from HUMOS inference
through to first-frame grounding alignment. Everything here runs **locally**.
IsaacGym is only needed for Step 5b (first-frame offset).

Steps 1 & 2 (XML asset generation and assets.yaml) are already done.

---

## Step 3 — HUMOS inference

**What it does:** Runs the trained HUMOS diffusion model on all HumanML3D clips,
generating motion sequences for each of the 16 held-out beta shapes × 2 genders.
Produces one `.pt` file per clip, each containing
`{male: {beta_key: {root_orient, pose_body, trans, betas, offset_height, ...}}, female: {...}}`.
Output goes alongside existing training outputs at `/home/hlz/datasets/humos_output/`.

```bash
cd /home/hlz/repos/humos

# interpolation shapes (16 betas × 2 genders — same range as training, new seed)
python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas_interp.pt \
    --local-out-dir /home/hlz/datasets/humos_output/interp

# extrapolation shapes (16 betas × 2 genders — wider range ±5, out-of-distribution)
python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /home/hlz/repos/ProtoMotions/protomotions/data/assets/all_betas_extrap.pt \
    --local-out-dir /home/hlz/datasets/humos_output/extrap
```

**Status:** interp DONE (717 files, all 16 beta keys verified). extrap: TODO.

---

## Step 4 — Export to AMASS NPZ

**What it does:** Converts each `{keyid}.pt` from HUMOS inference into per-variant
AMASS-format `.npz` files (one per gender×beta_key combination) plus a YAML index.
`--apply-offset-height` uses the `offset_height` field saved by HUMOS to shift the
root trajectory so the motion starts at the correct floor height.

```bash
cd /home/hlz/repos/ProtoMotions

python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/interp \
    --out-root /home/hlz/datasets/amass_heldout/interp \
    --genders male female \
    --apply-offset-height \
    --skip-existing \
    --fps 30.0

python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/extrap \
    --out-root /home/hlz/datasets/amass_heldout/extrap \
    --genders male female \
    --apply-offset-height \
    --skip-existing \
    --fps 30.0
```

---

## Step 5 — Convert to MotionLib .pt

**What it does:** Packs all the per-variant AMASS `.npz` files into a single
MotionLib `.pt` file — the format ProtoMotions training and evaluation scripts
consume. The `--assets-yaml` provides the per-shape beta vectors and root heights
so each motion is paired with the correct morphology metadata.

```bash
cd /home/hlz/repos/ProtoMotions

python tools/convert_amass_to_motionlib_with_morphology.py \
    --motion-yaml /home/hlz/datasets/amass_heldout/interp/humos_*.yaml \
    --assets-yaml protomotions/data/assets/mjcf/smpl_mor_interp/assets.yaml \
    --out /home/hlz/datasets/heldout_interp.pt

python tools/convert_amass_to_motionlib_with_morphology.py \
    --motion-yaml /home/hlz/datasets/amass_heldout/extrap/humos_*.yaml \
    --assets-yaml protomotions/data/assets/mjcf/smpl_mor_extrap/assets.yaml \
    --out /home/hlz/datasets/heldout_extrap.pt
```

---

## Step 5b — First-frame grounding offset

**What it does:** Uses IsaacGym FK to pose each humanoid at frame 0 and compute
the lowest collision point. Shifts `gts[:,:,2]` (all joint z-coordinates) so the
humanoid starts just above the floor (`target_z=0.005 m`). Without this step,
many motions start partially underground or floating, causing immediate physics
failures. Outputs `*_offset.pt` alongside the input files.

Requires IsaacGym — runs locally on your machine.

```bash
cd /home/hlz/repos/ProtoMotions

python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/heldout_interp.pt \
    --asset-root protomotions/data/assets/mjcf/smpl_mor_interp \
    --out-motion-file /home/hlz/datasets/heldout_interp_offset.pt \
    --overwrite

python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/heldout_extrap.pt \
    --asset-root protomotions/data/assets/mjcf/smpl_mor_extrap \
    --out-motion-file /home/hlz/datasets/heldout_extrap_offset.pt \
    --overwrite
```

---

## What comes next (RunPod)

After Step 5b, upload the two offset files to RunPod and run evaluation:

```bash
rsync -avz /home/hlz/datasets/heldout_interp_offset.pt runpod:/workspace/
rsync -avz /home/hlz/datasets/heldout_extrap_offset.pt runpod:/workspace/
```

Then see `note/README.eval-transfer.md` Part 1b Step 6 for the IsaacGym eval commands.
