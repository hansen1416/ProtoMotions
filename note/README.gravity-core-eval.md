# Gravity-Core Clip Evaluation: Physics Features vs Raw Betas

Goal: determine whether `hhi_phy_1024_transfer` (physics features, 15-dim) or
`hhi_1024_transfer` (raw betas, 11-dim) handles COM-change motions better, to inform
the morphology representation choice for Stage 2.

---

## Step 1 — Identify gravity-core clips from pilot training

Analyzed `results/hhi_1024_motion/persistent_failures.txt` (1024 clips, failure data from
epochs 8000–12000) by matching motion descriptions against COM-change keywords.

```bash
python3 /tmp/analyze_com_motions.py  # interactive analysis, see script below
```

**Result:** 88 clips identified as "gravity-core" — all requiring large COM displacement.

| Category | Clips | Persistent failure (≥15/21 epochs) | Avg betas failing/ep |
|---|---|---|---|
| floor_contact (crawl, all-fours, on knees/ground) | 44 | **100%** | **6.70** |
| sit_stand (sit/stand up, get up, lie down) | 16 | **100%** | **6.02** |
| kneel | 9 | **100%** | **5.82** |
| squat_crouch | 32 | **100%** | **4.29** |

All 88 clips fail persistently across all epochs and all body shapes — the clearest
structural bottleneck in the pilot training.

**Key insight:** The `failed_clips.pt` fine-tune set (192 clips, avg_betas ≥ 5.0)
contains only 49 of the 88 gravity-core clips. The other 39 were below the 5.0β threshold
and excluded. Creating a dedicated gravity-core file gives a cleaner, complete signal.

---

## Step 2 — Extract gravity-core clips into a new motion file

Script: `tools/extract_gravity_core_clips.py`

```bash
python tools/extract_gravity_core_clips.py \
    --failures-file results/hhi_1024_motion/persistent_failures.txt \
    --shard-dir /home/hlz/datasets/humos_proto/offset \
    --output /home/hlz/datasets/humos_proto/gravity_core_offset.pt
```

**Output:** `/home/hlz/datasets/humos_proto/gravity_core_offset.pt`
- 88 clips × 128 shapes = **11,264 motions**
- 2,252,800 total frames
- **4,747 MB** (~4.7 GB) — fits within RTX 4060 8 GB VRAM on RunPod

The script scans all 16 local offset shards (`humos_131072_NNNN_offset.pt`) matching by
`motion_clip_ids`. Found clips distributed across all 16 shards (2–14 clips per shard).
`motion_weights` reset to 1.0 (fresh curriculum).

Note: local RTX 4060 cannot run this evaluation — loading 128 different SMPL asset
templates simultaneously exceeds local VRAM. Must run on RunPod.

---

## Step 3 — Upload to R2, pull on RunPod

```bash
# Local → R2
rclone copy /home/hlz/datasets/humos_proto/gravity_core_offset.pt \
    r2:proto-data/gravity_core_offset.pt \
    --transfers=2 --s3-upload-concurrency=4 --s3-chunk-size=64M --progress

# RunPod: pull file + checkpoints
rclone copy r2:proto-data/gravity_core_offset.pt /workspace/ --progress
rclone copy r2:proto-data/ckpt/hhi_1024_phy_transfer.zip /workspace/ --progress
rclone copy r2:proto-data/ckpt/hhi_1024_transfer.zip /workspace/ --progress
unzip hhi_1024_phy_transfer.zip && unzip hhi_1024_transfer.zip
```

---

## Step 4 — Run evaluations (one at a time, same GPU)

```bash
# Run 1: physics features checkpoint
nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_phy_1024_transfer/last.ckpt \
    --simulator isaacgym \
    --motion-file /workspace/gravity_core_offset.pt \
    --num-envs 128 --headless \
    --output evaluation/phy_gravity_core.csv \
    > /tmp/eval_phy.log 2>&1 &

# Run 2: raw betas checkpoint (after Run 1 finishes)
nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_transfer/last.ckpt \
    --simulator isaacgym \
    --motion-file /workspace/gravity_core_offset.pt \
    --num-envs 128 --headless \
    --output evaluation/raw_gravity_core.csv \
    > /tmp/eval_raw.log 2>&1 &
```

---

## Results

CSVs stored at: `evaluation/evaluation_phy_gravity_core.csv` and `evaluation/evaluation_raw_gravity_core.csv`.
Both files: 1 header + 11,264 data rows = complete (88 clips × 128 shapes).

### Run 1 — `hhi_phy_1024_transfer` (physics features, 15-dim)

| Metric | Value |
|---|---|
| mean_body_dist — mean | **0.6730 m** |
| mean_body_dist — median | 0.6472 m |
| mean_body_dist — 90th pct | 1.0377 m |
| mean_body_dist — max | 2.8827 m |
| max_body_dist — max | 299.7 m |
| **Success** (root>0.3 m & dist<0.5 m) | **3323 / 11264 (29.5%)** |
| Exploded (max_dist>100 m) | 5821 / 11264 (51.7%) |
| Fell (root≤0.3 m, no explode) | 529 / 11264 (4.7%) |
| Drift (root ok, dist≥0.5 m) | 1591 / 11264 (14.1%) |

### Run 2 — `hhi_1024_transfer` (raw betas, 11-dim)

| Metric | Value |
|---|---|
| mean_body_dist — mean | **0.6909 m** |
| mean_body_dist — median | 0.6666 m |
| mean_body_dist — 90th pct | 1.0568 m |
| mean_body_dist — max | 3.3431 m |
| max_body_dist — max | 296.4 m |
| **Success** (root>0.3 m & dist<0.5 m) | **3060 / 11264 (27.2%)** |
| Exploded (max_dist>100 m) | 5993 / 11264 (53.2%) |
| Fell (root≤0.3 m, no explode) | 700 / 11264 (6.2%) |
| Drift (root ok, dist≥0.5 m) | 1511 / 11264 (13.4%) |

---

## Analysis

### Paired comparison (per-motion delta)

| Metric | Value |
|---|---|
| mean delta (phy − raw) | **−0.0179 m** (negative = physics better) |
| median delta | −0.0140 m |
| Physics features better | 5805 / 11264 motions (51.5%) |
| Raw betas better | 5459 / 11264 motions (48.5%) |

### Interpretation

Physics features edge out raw betas on this hardest subset:
- mean_body_dist lower by **0.018 m** (physics)
- success rate higher by **+2.3 pp** (29.5% vs 27.2%)
- explosion rate lower by **−1.5 pp** (51.7% vs 53.2%)

**However, the gap is 0.018 m — well below the 0.05 m threshold set as the decision criterion.**

Both checkpoints are dominated by the same structural failure mode: ~52% explosion rate on
gravity-core clips. This is a physics solver instability driven by the contact forces required for
floor-contact, crawl, and kneel motions — not a morphology conditioning problem. Neither
representation improves this.

### Decision: **Raw betas (11-dim) for Stage 2**

Gap too small (0.018 m < 0.05 m threshold) to justify architectural changes. Physics features would
require:
- Rebuilding Stage 1 with 15-dim morphology obs
- Discarding the running `hhi_20946_neutral` checkpoint
- Recomputing z-score statistics across all 128 shapes (requires full-scale MJCF loading)

Raw betas (11-dim) enables a clean Stage 1 → Stage 2 transfer:
- Same obs vector dimension (no `strict=False` loading)
- Stage 1 morphology dims are all-zero (neutral) and will activate cleanly in Stage 2
- The `reset_morphology_normalizer.py` step is still required to clear near-zero variances

The 0.018 m advantage of physics features could re-emerge at full scale (20,946 clips) or with
better contact handling — but this cannot be determined from pilot data alone, and the architectural
cost is too high to commit without a larger signal.
