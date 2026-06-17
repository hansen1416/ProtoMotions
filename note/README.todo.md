# TODO

## Training

- **A1 (HIGH)** Residual PD: change `q_target = q_neutral + scale * action` → `q_target = q_ref + scale * action` using `EnvContext.mimic.ref_state.dof_pos`
- **A2 (HIGH)** Contact reward: extend `contact_bodies` to include `L_Knee`, `R_Knee`, `L_Wrist`, `R_Wrist` for crawl/kneel clips
- **A3 (MED)** Phase obs: add `φ = frame_idx / total_frames ∈ [0,1]` as observation key
- **A4 (MED)** Per-shape RunningMeanStd: 128 separate buffers keyed by `asset_id`
- **A5 (MED)** PopArt per-shape return normalization for critic value head
- **A6 (LOW)** TVS difficulty re-scoring: replace current difficulty score with torque variation score (arXiv 2512.07248)

## Analysis (no new training)

- **B1 (CRITICAL)** Held-out eval: generate 16–32 interpolation betas + extrapolation betas via HUMOS; eval `hhi_1024_motion` and `hhi_1024_motion_tune` checkpoints; plot tracking error vs beta L2 distance
- **B2 (HIGH)** Embodiment probe: record actor hidden activations for 128 shapes; fit linear regression → `[mass, com_height, limb_lengths]`; report R² per property
- **B3 (MED)** Stride analysis: measure stride length/frequency across 128 shapes for a walking clip

## Scale-up (conditional on A gains)

- **C** Full-data run: 20,951 motions from `valid_motions.txt`, `num_envs=8192`, `batch_size=32768`
