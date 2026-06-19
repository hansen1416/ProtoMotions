# Evaluation Plan: Transfer Learning Checkpoints

Covers `hhi_1024_transfer` and `hhi_phy_1024_transfer`.  
Two-part pipeline: **Part 1** generates per-motion CSV files; **Part 2** analyses them.

---

## Checkpoints to Evaluate

| Name | Checkpoint path | Architecture | Base checkpoint |
|---|---|---|---|
| `hhi_1024_motion` | `results/hhi_1024_motion/last.ckpt` | MLP + beta concat | — (baseline, already converged) |
| `hhi_1024_transfer` | `results/hhi_1024_transfer/last.ckpt` | MLP + beta concat | `hhi_1024_motion` |
| `hhi_phy_1024_transfer` | `results/hhi_phy_1024_transfer/last.ckpt` | MLP + physics features | `hhi_phy_1024_motion` |

All three use `--robot-name smpl_mor --simulator isaacgym`.  
`resolved_configs_inference.pt` exists for the two base checkpoints already.  
For the transfer checkpoints it will be created on the first evaluation run (training writes it).

---

## Motion files

| Alias | Path | Size | Use |
|---|---|---|---|
| `full` | `/workspace/merged4/humos_slurmrank.pt` | 1024 clips × 128 shapes = 131,072 motions | Primary eval (RunPod only) |
| `smoke` | `/home/hlz/datasets/humos_proto/humos_128_offset.pt` | 1 clip × 128 shapes | Quick pipeline test (local) |

For a subset run, create a small shard (e.g., 32 clips × 128 = 4096 motions) from the offset shards at  
`/home/hlz/datasets/humos_proto/offset/humos_131072_*_offset.pt`.

---

## Part 1 — Generate CSV files

### 1a. Prerequisites

- Checkpoint exists and training has saved at least one `last.ckpt`
- `resolved_configs_inference.pt` present in checkpoint folder  
  (created automatically once training runs past the first epoch; copy from base checkpoint if needed)
- GPU available (IsaacGym required)
- `--num-envs 128` gives exactly one env per shape → each batch evaluates 128 motions in parallel

### 1b. Commands

Run on RunPod (all paths relative to `/workspace/ProtoMotions`).  
Adjust `--num-envs` upward (256, 512) to run more batches in parallel — must be a multiple of 128.

**Baseline (hhi_1024_motion)**
```bash
nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion/last.ckpt \
    --simulator isaacgym \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 128 \
    --output evaluation/hhi_1024_motion_full.csv \
    --progress-every 50 \
    > /tmp/eval_baseline.log 2>&1 &
```

**Transfer — no physics features (hhi_1024_transfer)**
```bash
nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_transfer/last.ckpt \
    --simulator isaacgym \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 128 \
    --output evaluation/hhi_1024_transfer_full.csv \
    --progress-every 50 \
    > /tmp/eval_transfer.log 2>&1 &
```

**Transfer — physics features (hhi_phy_1024_transfer)**
```bash
nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_phy_1024_transfer/last.ckpt \
    --simulator isaacgym \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 128 \
    --output evaluation/hhi_phy_1024_transfer_full.csv \
    --progress-every 50 \
    > /tmp/eval_phy_transfer.log 2>&1 &
```

**Quick smoke test (local, 1 clip × 128 shapes)**
```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_transfer/last.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --num-envs 128 \
    --output evaluation/smoke_transfer.csv
```

### 1c. CSV output columns

| Column | Description |
|---|---|
| `motion_id` | Index into the motion file |
| `gender` | `male` / `female` |
| `beta_key` | Unique body shape identifier |
| `mean_body_dist` | Mean Euclidean distance (all bodies, all frames) |
| `max_body_dist` | Peak distance — values >100m indicate physics explosion |
| `mean_root_dist` | Mean root joint distance |
| `max_root_dist` | Peak root distance |
| `min_root_height` | Minimum root height above ground (m) — falls < 0.3m |
| `steps_seen` | Frames evaluated |

### 1d. Estimated runtime

With `--num-envs 128` and 1024-clip full dataset:  
- 1024 batches × average motion length ~200 frames → ~200K simulator steps  
- Rough estimate: **2–4 hours** on a single A40/A100  
- Use `--num-envs 256` to halve runtime (requires 256 unique shape-matched envs — works only if dataset has ≥2 clips per shape)

---

## Part 2 — Analyse CSV files

All analysis is pure Python / pandas from the CSV output. No simulator needed.  
Target script: `tools/analyse_eval_csv.py` (to be created).

### 2a. Success rate (E2)

**Success criterion:** `min_root_height > 0.3 m AND mean_body_dist < 0.5 m`  
(Adjust thresholds after inspecting distributions.)

```python
df["success"] = (df["min_root_height"] > 0.3) & (df["mean_body_dist"] < 0.5)
df["exploded"] = df["max_body_dist"] > 100.0

print(f"Success rate: {df['success'].mean():.1%}")
print(f"Explosion rate: {df['exploded'].mean():.1%}")
```

Compare across all three checkpoints in one table.

### 2b. Per-shape success rate distribution (E4)

For each `(gender, beta_key)` aggregate success across all clips:

```python
per_shape = df.groupby(["gender", "beta_key"])["success"].mean()
per_shape.hist(bins=20)  # should be roughly uniform; tail = problem shapes
print(f"5th-percentile worst shape: {per_shape.quantile(0.05):.1%}")
```

Output: histogram figure + worst-10 shape table.

### 2c. Cross-shape variance per clip (E5)

For each clip (i.e., grouping by `motion_id // 128`), compute std of `mean_body_dist` across 128 betas:

```python
df["clip_id"] = df["motion_id"] // 128
cross_shape_std = df.groupby("clip_id")["mean_body_dist"].std()
print(f"Mean cross-shape std: {cross_shape_std.mean():.4f}")
print(f"Max cross-shape std:  {cross_shape_std.max():.4f}")
```

Low std = policy adapted per-shape; high std = some shapes consistently fail on certain clips.

### 2d. Shape extremity correlation (E6)

Requires beta vectors. Load from motion file:

```python
import torch
ml = torch.load("/workspace/merged4/humos_slurmrank.pt", weights_only=False)
# ml should expose motion_betas [N, 10] or similar
beta_l2 = ...  # per-motion ||beta||_2, join on (gender, beta_key)
```

Then scatter `beta_l2` vs `mean_body_dist` and fit linear regression.  
Small slope = uniform generalization across extremity.

### 2e. Checkpoint comparison table (paper Table 1)

| Metric | hhi_1024_motion | hhi_1024_transfer | hhi_phy_1024_transfer |
|---|---|---|---|
| Overall success rate | | | |
| Explosion rate | | | |
| Mean body dist | | | |
| 5th-pct per-shape success | | | |
| Mean cross-shape std | | | |

Fill in after all three CSVs are generated.

### 2f. Motion category × shape heatmap (E10)

Requires `data-processing/motion_id_text.json` (clip text labels).  
Categorise clips: locomotion / dynamic / manipulation / static by keyword.  
Compute mean body dist per (motion category, beta-L2 bucket).  
Output 2D heatmap.

---

## Part 3 — Augment evaluator with missing metrics

### 3a. What is currently missing

**Smoothness** — the transfer runs explicitly added a smoothness reward (`-0.05`) and a contact-force-change penalty.
Claiming "transfer improves smoothness" requires recording it. The infrastructure already exists
(`SmoothnessCalculator`, `MotionMetrics`) but `HHIFaultEvaluator` does not call it.

Missing CSV columns: `mean_normalized_jerk`, `high_jerk_frame_pct`

**Structured failure metadata** — currently you can only infer explosions from `max_body_dist > 100`.
It is more useful to know *when* and *how completely* an episode failed.

Missing CSV columns:
- `completed` (bool) — all frames ran without explosion
- `explosion_frame` (int) — first frame where `body_dist > threshold` (−1 if no explosion)
- `beta_l2` (float) — `‖β‖₂` for the shape; needed for E6 scatter plot; computable from
  `motion_lib.motion_betas[motion_id]` but not currently written to CSV

**What does NOT need adding now:** per-body joint breakdown and contact forces — useful for E8/E9
but those are Tier 3 deferred items and the infrastructure already exists to add them later.

### 3b. Required changes to `HHIFaultEvaluator`

1. Allocate a `MotionMetrics` buffer for per-frame rigid-body positions in `initialize_eval`
   (same pattern as `MimicEvaluator._add_robot_state_metrics`).
2. Fill the buffer each frame in `_record_distance` — already has `pred_pos`; write it to the buffer.
3. Track `explosion_frame` per motion: on first frame where `body_dist > EXPLOSION_THRESHOLD`
   (e.g. 5 m, distinct from the 100 m "already exploded" heuristic) record the frame index.
4. In `process_eval_results`, call `SmoothnessCalculator.compute_smoothness_metrics` on the buffer
   to get `normalized_jerk_mean` and `high_jerk_frame_percentage_mean`.
5. Load `motion_lib.motion_betas` and compute per-motion `‖β‖₂`.
6. Add `completed`, `explosion_frame`, `beta_l2`, `mean_normalized_jerk`, `high_jerk_frame_pct`
   to the CSV fieldnames and row dicts in `_build_rows` / `_save_rows`.

### 3c. Updated CSV columns (target schema)

| Column | Type | Description |
|---|---|---|
| `motion_id` | int | Index into motion file |
| `gender` | str | `male` / `female` |
| `beta_key` | str | Unique shape identifier |
| `beta_l2` | float | `‖β‖₂` — shape extremity |
| `mean_body_dist` | float | Mean Euclidean body distance (m) |
| `max_body_dist` | float | Peak body distance — >100 m = explosion |
| `mean_root_dist` | float | Mean root distance (m) |
| `max_root_dist` | float | Peak root distance (m) |
| `min_root_height` | float | Minimum root height (m) — <0.3 m = fall |
| `completed` | bool | All frames ran without explosion |
| `explosion_frame` | int | First frame body_dist >5 m; −1 if none |
| `mean_normalized_jerk` | float | Windowed normalized jerk (lower = smoother) |
| `high_jerk_frame_pct` | float | % frames with jerk above threshold |
| `steps_seen` | int | Total frames evaluated |

---

## Deferred items (need extra data or augmented rollout)

- **E7** Held-out shape generalization — needs new HUMOS betas (interpolation ±3, extrapolation ±5)
- **E8** Joint torque recording — needs augmented `evaluate_hhi_faults.py` loop
- **E9** Contact timing adaptation — needs per-step foot contact state recording
- **E11** Failure mode taxonomy — fall / COM drift / joint-limit / contact failure classification
- **B2** Embodiment probe — actor hidden activations → linear regression to mass/COM/limb lengths
