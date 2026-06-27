# Metrics & Experiments Plan — Preliminary Paper

## Primary evaluation targets

| Checkpoint | Arch | Status |
|---|---|---|
| `hhi_20946_neutral` | MLP + raw betas (neutral, β=0) | **Stage 1 — running** |
| `hhi_stage2_transfer` | MLP + TBD morphology repr | **Stage 2 — blocked on data** |

Pilot checkpoints are available on R2 for ablation comparison:

| Checkpoint | Arch | Epochs | Reward | Inference (8 envs) |
|---|---|---|---|---|
| `hhi_1024_motion` | MLP + raw betas | 12,021 | 0.84 | — |
| `hhi_1024_transfer` | MLP + raw betas | 21,400 | 0.84 | **0/8** |
| `hhi_phy_1024_transfer` | MLP + physics features | 17,000 | 0.84 | **5/8** |

The evaluation methodology below (E1–E7) targets the Stage 2 checkpoint. Run the same pipeline
against pilot checkpoints if ablation comparison is needed.

---

## Key finding from pilot (T1, 2026-06-22)

Pilot transfer runs had nearly identical binary failure rates (177 vs 167 clips). Binary pass/fail
does NOT explain the 0/8 vs 5/8 visual inference gap between physics features and raw betas.
**Continuous metrics are the primary differentiator.** Consequently:
- Implement E3 (smoothness columns) before running E1 so jerk is captured in the same RunPod session.
- Report jerk separately for stable episodes (`max_body_dist < 100m`) vs all episodes, since
  physics explosions dominate the jerk average and confound the comparison.

---

## Tier 1 — Core (must-have for paper, RunPod required)

### E1 — Full CSV evaluation (RunPod, 2–4 hrs, implement E3 augmentation first)

Run `evaluate_hhi_faults.py` against the full Stage 2 dataset. This single step unlocks
E2–E6, S1, S2 as pure pandas post-processing — no further simulation needed.

**Primary target (Stage 2 checkpoint):**
```bash
nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_stage2_transfer/last.ckpt \
    --simulator isaacgym \
    --motion-file /workspace/stage2_data/stage2_slurmrank.pt \
    --num-envs 128 \
    --output evaluation/hhi_stage2_transfer_full.csv \
    --progress-every 50 \
    > /tmp/eval_stage2.log 2>&1 &
```

**Pilot ablation (optional, run on 1024×128 dataset):**
```bash
# run in parallel on RunPod (separate tmux panes)
nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_transfer/last.ckpt \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 128 --output evaluation/hhi_1024_transfer_full.csv &

nohup python -u protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_phy_1024_transfer/last.ckpt \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 128 --output evaluation/hhi_phy_1024_transfer_full.csv &
```

### E2 — Overall success/failure table (pandas, local, 5 min)

Three-way comparison table (Table 1 of paper). Thresholds: `success = min_root_height > 0.3m AND mean_body_dist < 0.5m`, `explosion = max_body_dist > 100m`.

Columns: overall success rate, explosion rate, mean body dist, median body dist, 90th-pct body dist.

### E4 — Per-shape success rate distribution (pandas, local)

Group by `(gender, beta_key)` → 128 per-shape success rates per checkpoint. Plot histogram and CDF overlay for all three. Key question: does `hhi_phy_1024_transfer` have a higher floor across all shapes, or does it help specific shapes only?

### E5 — Cross-shape tracking variance per clip (pandas, local)

For each of the 1024 clips, compute `std(mean_body_dist)` across 128 betas. Mean and max std per checkpoint. Low std = policy adapted per-shape uniformly; high std = some shapes consistently fail on certain clips.

### E6 — Shape extremity correlation (pandas, local)

Scatter `‖β‖₂` vs `mean_body_dist` (one point = one motion, 131k points total). Fit OLS line. Compare slope across checkpoints. Expected: `hhi_phy_1024_transfer` has shallower slope (less degradation at extreme betas).

### E3 — Smoothness + shape extremity columns (augment `evaluate_hhi_faults.py`, implement before E1)

**What it measures:** normalized jerk on rigid body world positions — the standard biomechanics smoothness metric `NJ = (T⁵ × ∫|jerk|² dt) / path_length²`. High NJ = jerky body movement. This is distinct from action smoothness (consecutive action delta), but correlated: a policy that outputs large oscillatory actions will produce jerky body trajectories.

**New CSV columns to add:**
- `mean_normalized_jerk` — per-motion mean NJ across all 24 body joints and all frames (exclude motions with `max_body_dist > 100m` from averages, since explosions dominate and confound)
- `high_jerk_frame_pct` — % of 0.4s windows where any body exceeds NJ threshold 6500
- `beta_l2` — `‖β‖₂` of the SMPL beta vector for this motion's body shape; needed for E6 scatter plot; one line to extract from `motion_lib.motion_betas`

**Infrastructure:** `SmoothnessCalculator` and `MotionMetrics` already exist in `protomotions/agents/evaluators/`. `HHIFaultEvaluator` needs to:
1. Allocate a `MotionMetrics` buffer for per-frame rigid body positions in `initialize_eval`
2. Fill the buffer each frame in `_record_distance` (already has `pred_pos`)
3. Call `SmoothnessCalculator.compute_normalized_jerk_from_pos` in `process_eval_results`
4. Extract `beta_l2` from `motion_lib.motion_betas[motion_id].norm()`
5. Add the three new columns to the CSV fieldnames and rows

**Caveat:** report jerk in two ways — (a) all episodes, (b) stable-only (`max_body_dist < 100m`). This separates explosion-driven jerk from genuine smoothness differences.

---

## Tier 2 — Important (post-E1 analysis, no GPU needed)

### S2 — Failure mode taxonomy (pandas, post-E1)

Classify each episode: **explosion** (`max_body_dist > 100m`), **fall** (`min_root_height < 0.3m`, no explosion), **COM drift** (root ok but `mean_body_dist > 0.5m`, no fall/explosion), **success**. Plot failure-type stacked bar chart per checkpoint, and per shape-extremity bucket (`‖β‖₂` quartiles).

### S1 — Motion type × shape extremity heatmap (pandas, post-E1)

Categorize 1024 clips into 4 types (locomotion, dynamic/jumping, manipulation, static/pose) from `data-processing/motion_id_text.json` using keyword matching. Compute mean body dist per `(motion_type, beta_L2_quartile)` cell → 4×4 heatmap, three panels (one per checkpoint). Shows *which* motion types are shape-sensitive.

---

## Tier 3 — Generalization proof (key paper contribution, needs data pipeline)

### E7 — Held-out beta generalization (RunPod eval, after data pipeline completes)

16 interpolation betas (±3 range, different seed) + 16 extrapolation betas (±5 range). Pipeline status: interp HUMOS inference done locally, needs upload → NPZ export → MotionLib → grounding offset → RunPod eval. Extrap not yet started.

This is the cleanest experiment: same 1024 clips, totally unseen body shapes. Measures genuine cross-shape generalization vs memorization. The paper's Contribution 3 (cross-shape evaluation protocol) depends on this.

### S3 — Retargeting behavior (pandas + root position extraction, post-E1)

For a locomotion clip, extract stride length and stride frequency from root position trajectory across all 128 shapes. Plot stride length vs body height. If positive correlation exists, the policy implicitly scales motion to body proportions — this is the "physically meaningful" argument, not just reference tracking.

---

## Training diagnostics (local, from failed_motion logs)

### T1 — Failed motion overlap analysis (local, already have data)

Extract which clip IDs appear in `hhi_1024_transfer` vs `hhi_phy_1024_transfer` failed motion logs at convergence. Overlap = motions that are hard regardless of conditioning. Non-overlap = where physics features help specifically. Compare to the original `hhi_1024_motion` persistent failures.

**Results:** see `results/analysis/T1_failed_clip_overlap.md`

### T2 — Training curve extraction (from wandb or tensorboard)

Pull `unnormalized_task_rewards`, `eval/success_rate`, `eval/normalized_jerk_mean` over epochs for all three checkpoints → single plot showing convergence and smoothness trajectory. Needed for any training curve figure.

---

## Priority order for preliminary paper

| Priority | Experiment | Blocker | Effort |
|---|---|---|---|
| 1 | **E1** — full CSV (all 3 checkpoints) | RunPod session | 6–12 hrs machine time |
| 2 | **E2** — success/failure table | E1 | 30 min |
| 3 | **E4** — per-shape distribution | E1 | 1 hr |
| 4 | **E6** — shape extremity scatter | E1 | 1 hr |
| 5 | **E3 augmentation** — add smoothness to evaluator | code change (~2 hrs) + re-run | 4 hrs + RunPod |
| 6 | **S2** — failure taxonomy | E1 | 1 hr |
| 7 | **T1** — failed clip overlap | local, have data now | 1 hr |
| 8 | **S1** — motion type heatmap | E1 + text labels | 2 hrs |
| 9 | **E7** — held-out generalization | data pipeline (weeks) | weeks |
| 10 | **S3** — retargeting behavior | E1 + root extraction | 2 hrs |
