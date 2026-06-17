# TODO

## Training

- ~~**A1 (HIGH)** Residual PD: change `q_target = q_neutral + scale * action` → `q_target = q_ref + scale * action` using `EnvContext.mimic.ref_state.dof_pos`~~ *(ProtoMotions does not use residual PD and achieves its results without it; PHC claim does not transfer)*
- **A2 (HIGH)** Contact reward: extend `contact_bodies` to include `L_Knee`, `R_Knee`, `L_Wrist`, `R_Wrist` for crawl/kneel clips
- **A3 (MED)** Phase obs: add `φ = frame_idx / total_frames ∈ [0,1]` as observation key
- **A4 (MED)** Per-shape RunningMeanStd: 128 separate buffers keyed by `asset_id`
- **A5 (MED)** PopArt per-shape return normalization for critic value head
- **A6 (LOW)** TVS difficulty re-scoring: replace current difficulty score with torque variation score (arXiv 2512.07248)

## Analysis (no new training)

- **B1 (CRITICAL)** Held-out eval: generate 16–32 interpolation betas + extrapolation betas via HUMOS; eval `hhi_1024_motion` and `hhi_1024_motion_tune` checkpoints; plot tracking error vs beta L2 distance
- **B2 (HIGH)** Embodiment probe: record actor hidden activations for 128 shapes; fit linear regression → `[mass, com_height, limb_lengths]`; report R² per property
- **B3 (MED)** Stride analysis: measure stride length/frequency across 128 shapes for a walking clip

================================================================================

## Evaluation — `hhi_1024_motion_tune` (systematic)

### Tier 1 — Standard motion imitation metrics (no new code needed)

- **E1 (CRITICAL)** Run `evaluate_hhi_faults.py` on the current checkpoint — produces per-(gender, beta_key) CSV with mean/max body distance, root distance, min root height
- **E2 (HIGH)** Post-process CSV: apply thresholds (`min_root_height > 0.5 m` AND `mean_body_dist < 0.5 m`) to compute **success rate** per shape and overall — primary headline number for paper comparison
- **E3 (MED)** Confirm motion smoothness is logged (already in `MimicEvaluator._register_smoothness_plugin`); extract action jitter / body jerk from eval run

### Tier 2 — Multi-shape specific metrics (novel contribution, ~50 lines pandas each)

- **E4 (HIGH)** **Per-shape success rate distribution**: histogram of success rate across 128 (gender, beta_key) pairs; report 5th-percentile worst shape → core Table 1 / Figure for section 4.1
- **E5 (HIGH)** **Cross-shape variance**: for each clip `c`, compute `std_β(mean_body_dist(c, β))` across all 128 shapes; report mean and max — low variance = policy adapted to morphology, not averaging
- **E6 (HIGH)** **Shape extremity correlation**: scatter `‖β‖₂` vs `mean_body_dist` per shape; fit linear regression; small slope = uniform generalization → section 4.3d
- **E7 (HIGH)** **Held-out shape generalization**: run same evaluation on 16–32 interpolation betas (same range, new seed) + extrapolation betas (±5 range); compare success rate vs training shapes → section 4.2 *(see B1)*

### Tier 3 — Physical analysis (section 4.3, high paper value, needs augmented rollout)

- **E8 (MED)** **Joint torque × body mass** (4.3a): augment evaluator to record per-step joint torques; for the same clip, compare torques across lightest/heaviest shapes; heavier → higher torques validates shape-awareness
- **E9 (MED)** **Contact timing adaptation** (4.3c): record per-step foot contact states; measure contact onset timing across shapes for same locomotion clip; does foot contact shift with body height?
- **E10 (MED)** **Motion type × shape heatmap** (S1): categorise 1024 clips by keyword (locomotion / dynamic / manipulation / static) using `data-processing/motion_id_text.json`; compute mean body distance per (motion category, beta-L2 bucket) cell; output 2D heatmap
- **E11 (MED)** **Failure mode taxonomy** (S3): from rollout data classify failures into fall / COM drift / joint-limit violation / contact failure; plot distribution per shape extremity bucket and motion category
- **E12 (LOW)** **Stride length vs body height** (S4): locomotion clips only; measure stride length/frequency per shape from root XY trajectory; positive correlation with body height = implicit retargeting *(see B3)*

### Execution order

1. E1 → E2 → E4 → E5 → E6 (all from one CSV, can be done now)
2. E7 / B1 (needs held-out beta generation via HUMOS first)
3. E8 → E9 → E10 → E11 (needs augmented rollout loop)
4. E12 / B3, B2 (lowest priority, do if time allows)

## Scale-up (conditional on A gains)

- **C** Full-data run: 20,951 motions from `valid_motions.txt`, `num_envs=8192`, `batch_size=32768`
