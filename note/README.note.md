# We only append to this file, everything is on chronological order.

# Implementation Notes — HHI Morphology Project

================================================================================

## 1. Research Context

### [Research] Literature Gap

No prior work uses continuous SMPL body shape variation for physics-based motion imitation.

PULSE (ICLR 2024) and PHC (NeurIPS 2023) both explicitly use only the mean SMPL shape. XHugWBC, H-Zero, MetaMorph, and ManyQuadrupeds all work with discrete different robot designs — not continuous beta variation. The combination of (1) continuous SMPL beta variation, (2) physics-based motion imitation across diverse clips, and (3) a single shared policy is our novel contribution.

---

### [Analysis] Why All Architectures Converge to the Same Reward (~0.84)

All three runs (mlp, shape_embed, physics_feat) hit the same ceiling despite different morphology encodings. Two explanations:

**1. Shape variance << motion variance in gradients.**
With 1024 diverse motion clips, gradient signal is dominated by motion content variation. The 128-shape signal is a much smaller component of the loss landscape — the network learns motion-invariant features first; shape-specific adaptation is a small correction.

**2. Implicit shape information from proprioception.**
Some shape information (body height, segment length ratios) is implicitly available in proprioceptive states. The 11-dim beta vector offers an explicit shortcut, but the network may not need it for seen shapes when the implicit signal is sufficient.

**Note on MorFiC (arXiv 2603.14554):** The paper attributes multi-morphology plateaus to shared critic value miscalibration. Our critic already receives `morphology_obs` in its `in_keys` — it is not blind to morphology. MorFiC's fix does not apply to our setup.

---

### [Research] Key Literature Findings

| Finding | Source | Implication |
|---|---|---|
| Residual PD (`q_ref + scale*action`) dramatically helps non-standing motions | PHC (NeurIPS 2023) | Highest priority training fix; also addresses jerk in fine-tune |
| Contact reward needed for floor-contact clips (knees, hands) | PHC, SkillMimic | Extend `contact_bodies` beyond feet |
| Phase variable φ resolves temporal aliasing in squat/kneel | PULSE, Bi-Level | One-line obs addition |
| TVS (Torque Variation Score) correctly rates squats/crawls as hard; kinematic metrics don't | arXiv 2512.07248 | Better difficulty curriculum |
| FiLM is unstable in RL with low-dim conditioning; zero-init gamma is the fix if retried | FiLM-Ensemble | Don't retry FiLM without zero-init gamma |
| Linear probing on activations measures embodiment encoding | Standard RL analysis | "AI learned physics" paper figure |

================================================================================

## 2. Data Pipeline

### [Pipeline] Step 1: Generate SMPL Robot Assets

Use SMPLSim `run.py` to generate `all_betas.pt` and `.xml` files for smpl and smplx.

Use `scripts/generate_smpl_mor_asset_info.py` to generate the asset information `.yaml` files:
- `protomotions/data/assets/mjcf/smpl_mor/assets.yaml`
- `protomotions/data/assets/mjcf/smplx_mor/assets.yaml`

These are used in `protomotions/robot_configs/smpl_mor.py`:
```python
asset: RobotAssetConfig = field(
    default_factory=lambda: RobotAssetConfig(
        asset_folder_name="mjcf/smpl_mor/",
        asset_info_file="mjcf/smpl_mor/assets.yaml",
        ...
    )
)
```

All SMPL `.xml` templates are in `protomotions/data/assets/mjcf/smpl_mor/*.xml`.

---

### [Pipeline] Step 2: Export HUMOS Output → AMASS-style `.npz`

**Batch mode — all 1024 clips, all 128 variants each:**
```bash
python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/ \
    --out-root /home/hlz/datasets/humos_proto_interm/ \
    --skip-existing
```
Produces `humos_proto_interm/HUMOS/*.npz` and `humos_proto_interm/humos_131072.yaml` (1024 clips × 128 variants). `--skip-existing` makes re-runs safe after interruption.

**Small test (8 variants from one clip):**
```bash
python tools/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto_interm_8/ --num 8
```

---

### [Pipeline] Step 3: Convert `.npz` → MotionLib `.pt`

**Full dataset (131,072 motions, batched to stay within RAM):**
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
Produces `humos_proto/humos_131072_{chunk_idx:04d}.pt` per chunk. Each chunk ~3.6 GB. No merge step — chunks are the final output.

---

### [Pipeline] Step 4: Align Frame 0 with Ground

```bash
python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/humos_proto/humos_131072_0000.pt \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --out-motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt \
    --limit -1 \
    --overwrite
```

Run for all shards:
```bash
tools/run_frame0_offsets.sh
```

Merge shards into fewer files:
```bash
tools/merge_motion_shards.py
```

#### [Logic] Frame-0 Grounding Algorithm (`compute_humos_frame0_offsets.py`)

**Goal**: shift each motion's `gts[:, :, 2]` so the lowest collision point at frame 0 sits at `target_z = 0.005 m` above the ground plane. This prevents characters from spawning partially underground.

**Key design**: creates one IsaacGym env per **unique SMPL shape** (~128 envs) and reuses them across all ~131k motions in rounds — ~64× cheaper than one env per motion.

**Steps:**
1. Parse collision geometry from each MJCF XML (capsules, spheres, boxes, cylinders → `LocalShape` objects with body-local points and radii). Pre-batches these into `[N_shapes, P, 3]` GPU tensors once.
2. For each round, set frame-0 root position/rotation and DOF state into the matching env using IsaacGym's indexed set-tensor API (only touched envs disturbed).
3. Run one simulation step (gravity disabled) → refresh `rigid_body_states` to get FK'd world poses.
4. For each geom slot: rotate local points by the body's world quaternion, add body world position → world-space collision points. Lowest Z across all geoms = `lowest_z[env]`.
5. `offset = target_z - lowest_z`. Apply as a constant Z shift to all frames: `gts[start:end, :, 2] += offset`. Velocities (`gvs`) are unchanged — a rigid vertical shift does not affect velocity.
6. Save corrected `gts` back into the motion file (all other metadata fields preserved).

**Output**: `*_offset.pt` file with identical structure to input; only `gts` changes.

---

### [Pipeline] Step 5: Visualize

```bash
python examples/motion_libs_visualizer_mor.py \
    --motion_files ~/datasets/humos_proto/offset/humos_131072_0015_offset.pt \
    --robot smpl_mor \
    --simulator isaacgym \
    --start 360 --batch-size 16
```

---

### [Reference] Motion File Format (`humos_8_offset.pt`)

```
gts:              tensor [n_frames, 24, 3]   — global translations
grs:              tensor [n_frames, 24, 4]   — global rotations (xyzw)
gvs:              tensor [n_frames, 24, 3]   — global velocities
gavs:             tensor [n_frames, 24, 3]   — global angular velocities
dvs:              tensor [n_frames, 69]      — DOF velocities
dps:              tensor [n_frames, 69]      — DOF positions
lrs:              tensor [n_frames, 24, 4]   — local rotations
contacts:         tensor [n_frames, 24]      — contact flags per body
length_starts:    tensor [n_envs]
motion_lengths:   tensor [n_envs]
motion_dt:        tensor [n_envs]
motion_num_frames: tensor [n_envs]
motion_weights:   tensor [n_envs]
motion_betas:     tensor [n_envs, 10]
motion_gender_ids: tensor [n_envs]           — -1 (female) or 1 (male)
motion_genders:   tuple [n_envs]             — 'male' / 'female'
motion_beta_keys: tuple [n_envs]             — e.g. '1e5a1c90'
motion_asset_ids: tuple [n_envs]             — e.g. 'male_0e26b88d'
motion_clip_ids:  tuple [n_envs]             — e.g. '000005'
motion_npz_files: tuple [n_envs]             — *.npz source files
```

================================================================================

## 3. Morphology Code Changes

### [Reference] Key Modified Files

| File | Change |
|---|---|
| `protomotions/robot_configs/smpl_mor.py` | `SmplMorRobotConfig`, points to `smpl_mor` assets.yaml |
| `protomotions/components/motion_lib.py` | `build_asset_id_to_motion_ids()`, `sample_motions_for_asset_ids()` |
| `protomotions/simulator/isaacgym/simulator.py` | Multi-shape XML loading, per-env asset assignment, physics features |
| `protomotions/simulator/base_simulator/simulator.py` | Base changes for morphology support |
| `protomotions/envs/base_env/env.py` | `ctx.env_morphology [num_envs, 11]` in `_build_global_context()` |
| `examples/motion_libs_visualizer_mor.py` | Visualizer: one env per unique `asset_id`, shape-matched motions |
| `protomotions/inference_agent_mor.py` | `--gender-beta` flag, `--max-motions` flag |
| `protomotions/evaluate_hhi_faults.py` | Per-(gender, beta_key) CSV evaluator |
| `protomotions/agents/common/film_mlp.py` | `FiLMMLPConfig` + `FiLMMLPWithCond` (Run 2) |
| `protomotions/agents/common/shape_embed_mlp.py` | `ShapeEmbedMLPConfig` + `ShapeEmbedMLP` (Run 3) |
| `tools/extract_smpl_physics_features.py` | Reads all 128 XMLs, computes 15 features, z-scores, saves `.pt` (Run 4) |
| `protomotions/data/assets/mjcf/smpl_mor/physics_features.pt` | 128×15 feature matrix keyed by `asset_id` (Run 4) |

---

### [Code] Code Path Overview

**Training asset load:**
- `selected_asset_ids` auto-populated from motion library's unique `asset_ids` before simulator init.
- Code path: `env.py:initialize_simulator → simulator.py:_load_humanoid_assets`

**Per-env asset assignment:**
- Environments assigned assets by round-robin over filtered asset set in `IsaacGymSimulator._build_humanoid_asset_assignment`.
- Produces `env_id_to_asset_idx`, `env_id_to_asset_name`, `env_morphology`, etc.
- Code path: `env.py:initialize_simulator → simulator.py → env.py:initialize_simulator`

**Motion sampling:**
- `sample_motions_for_asset_ids` only samples motions from the bucket matching the env's `asset_id`.
- Code path: `mimic_motion_manager.py:sample_motions → motion_lib.py:sample_motions_for_asset_ids`
- `asset_id_to_motion_ids` mapping built in `motion_lib.py:build_asset_id_to_motion_ids`

**Morphology observation pipeline:**
```
simulator._build_humanoid_asset_assignment()
  → self.env_morphology = [gender_id, betas / 3.0]   # [num_envs, 11], built once at startup

_build_global_context()   ← called every step
  → ctx.env_morphology = self.simulator.env_morphology   # same tensor, no copy

ComponentManager.execute_all(observation_components)
  → compute_morphology_obs(morphology=tensor)
  → _observation_buffer["morphology_obs"]   # [num_envs, 11]

get_obs() → network reads it by key
```
Code path: `simulator.py:558-585 → env.py:972 → component_factories.py:1265-1279 → obs/humanoid.py:351`

**Reset pose:**
- `motion_lib.get_motion_state` fetches reference position, rotation, DOF state for env's current `motion_id`.
- Code path: `env.py:compute_ref_reset_state → env.py:reset`

**Betas XML ↔ motion file consistency:**
- Trusted via shared `beta_key` hash. Runtime check enabled with `PROTOMOTIONS_DEBUG=1`.
- Code path: `simulator.py:444-456 → motion_lib.py:136-140 → env.py:326-350`

---

### [Data] Motion Difficulty Curriculum

Full dataset: 20,951 motions across 128 beta variants (64 shapes × 2 genders), listed in `/home/hlz/repos/hhi/data-processing/valid_motions.txt`.

Pilot training: ~1,024 motions sampled from the 5th–55th difficulty percentile. Skips pure static poses (bottom) and motions too hard to converge within budget (top).

================================================================================

## 4. Training Commands

### [Command] Local Smoke Tests (8 envs)

```bash
# Baseline
python protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 --batch-size 16

# FiLM
python protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_film.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 --batch-size 16

# Shape Embedding
python protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_shape_embed.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 --batch-size 16

# Physics Features
python protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_physics.py \
    --experiment-name hhi_physics_feat_1024 \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 --batch-size 16
```

### [Command] Full-Scale RunPod Commands (4096 envs)

```bash
# Run 1 — Baseline
python protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 4096 --batch-size 16384

# Run 5 — Hard Clip Fine-Tune (from converged baseline)
python protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion_tune \
    --motion-file /home/hlz/datasets/humos_proto/failed_clips.pt \
    --num-envs 4096 --batch-size 16384 \
    --overrides agent.config.init_from=results/hhi_1024_motion/last.ckpt
```

### [Command] Inference Commands

```bash
# Single-shape inference
python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_1024_motion/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --compact-spawn-spacing 1.5 --num-envs 16

# Large motion file (use --max-motions to limit GPU memory)
python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_1024_motion/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt \
    --compact-spawn-spacing 1.5 --num-envs 16 --max-motions 1024

# Specific gender-beta selection
python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_single_motion_multi_shape/score_based.ckpt \
    --simulator isaacgym --num-envs 16 \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --gender-beta male:bfd4619b male:c1d2c0ef male:ca12d763 male:cf7925fd \
    --compact-spawn-spacing 1.2
```

All 128 gender-beta keys (female then male, 64 each):
```
female:093098f0 female:09a0fcbd female:0e26b88d female:0f05fd5a female:10900e9a female:10c258c2 female:1658f5d3 female:1e5a1c90 female:2286da8c female:25247499 female:2e949ac0 female:30f6048e female:312bf810 female:324b2d00 female:36baeba5 female:371b5e94
female:3b4a94c2 female:3c2cfe86 female:3faff413 female:42909c1b female:443d6b3e female:4dd55cac female:4de6c13b female:52d9e1de female:546170ba female:653185e6 female:71fbbe41 female:724d4ad2 female:770f9e2c female:78613653 female:7b3c6576 female:7d706ded
female:7e492dfc female:7f246a41 female:82266732 female:944474c9 female:97b473d4 female:9b4a6dda female:9d418743 female:a0720cb2 female:a2c978d0 female:a9143d09 female:abbf826b female:ad5728e1 female:b3fd6d6b female:b8e5fb4e female:b928198f female:bd3137aa
female:bfd4619b female:c1d2c0ef female:ca12d763 female:cf7925fd female:d1dc53df female:d495801e female:d4c80970 female:d6f908ec female:d9dbd795 female:da7b9ae1 female:df1b853d female:dfd2d9cf female:e57f26a5 female:e5c9712a female:f0de7631 female:fb454239
male:093098f0 male:09a0fcbd male:0e26b88d male:0f05fd5a male:10900e9a male:10c258c2 male:1658f5d3 male:1e5a1c90 male:2286da8c male:25247499 male:2e949ac0 male:30f6048e male:312bf810 male:324b2d00 male:36baeba5 male:371b5e94
male:3b4a94c2 male:3c2cfe86 male:3faff413 male:42909c1b male:443d6b3e male:4dd55cac male:4de6c13b male:52d9e1de male:546170ba male:653185e6 male:71fbbe41 male:724d4ad2 male:770f9e2c male:78613653 male:7b3c6576 male:7d706ded
male:7e492dfc male:7f246a41 male:82266732 male:944474c9 male:97b473d4 male:9b4a6dda male:9d418743 male:a0720cb2 male:a2c978d0 male:a9143d09 male:abbf826b male:ad5728e1 male:b3fd6d6b male:b8e5fb4e male:b928198f male:bd3137aa
male:bfd4619b male:c1d2c0ef male:ca12d763 male:cf7925fd male:d1dc53df male:d495801e male:d4c80970 male:d6f908ec male:d9dbd795 male:da7b9ae1 male:df1b853d male:dfd2d9cf male:e57f26a5 male:e5c9712a male:f0de7631 male:fb454239
```

### [Command] Evaluator Command

```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion/last.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --num-envs 64 \
    --output results/eval_mlp_training_betas.csv
```

================================================================================

## 5. Training Results

All runs use `--robot-name smpl_mor --simulator isaacgym`.
The 128 SMPL body shapes (64 β-vectors × 2 genders) span total_mass 26–144 kg and total_height 1.13–1.67 m.

### [Results] Summary Table

| Run | Experiment name | Input dim | Intervention | Reward | Outcome |
|---|---|---|---|---|---|
| 1 | `hhi_1024_motion` | 11 | Baseline — raw beta concat | ≈ 0.84 | **Baseline** |
| 2 | `hhi_film_1024_motion` | 11 | FiLM multiplicative conditioning | ≈ 0.40–0.45 | Failed — fanout + instability |
| 3 | `hhi_se_1024_motion` | 11 → 64 embed | Shape embedding + concat | ≈ 0.84 | Neutral — no gain |
| 4 | `hhi_physics_feat_1024` | 15 | Physics features (z-scored, input swap) | ≈ 0.84 | Neutral — no gain |
| 5 | `hhi_1024_motion_tune` | 11 | Fine-tune on 192 hard clips | > 0.90 | Success rate +15 pp; jerk 3–4× |

---

### [Experiment] Run 1: Baseline — Direct Beta Concatenation (`hhi_1024_motion`)

**Morphology input**: `morphology_obs` = `[gender_id, beta_1/3, …, beta_10/3]` — 11-dim, appended directly to the flat observation vector before the MLP trunk.

**Architecture**: Standard 6-layer 1024-unit MLP (`MLPWithConcat`). No special conditioning — the 11-dim morphology vector is just another group of floats in the input. Actor and critic both receive `morphology_obs` in their `in_keys`.

**Result**: Converged to reward ≈ 0.84. This is the **baseline** all other runs are compared against.

**Known failure modes**: 65 hard clips involving floor-contact motions (crawl, kneel, squat, backward-walk) fail persistently. The root hypothesis is that raw PCA betas give the policy no explicit signal about the physical constraints governing these motions (torso mass, leg length, COM height relative to floor).

**Status**: Converged. Reference point for all ablations.

---

### [Experiment] Run 2: FiLM Conditioning (`hhi_film_1024_motion`)

**Motivation**: FiLM (Feature-wise Linear Modulation) conditions the trunk by predicting per-layer scale (γ) and shift (β) from the morphology input, rather than concatenating morphology into the obs.

**Architecture**: Conditioner MLP (64→64 hidden units) produces `2 × num_layers × hidden_dim` values. Trunk activations modulated as `h_l = h_l × γ_l + β_l`.

**Files**: `protomotions/agents/common/film_mlp.py` (`FiLMMLPConfig`, `FiLMMLPWithCond`), `examples/experiments/mimic/mlp_film.py`.
- Trunk is a `ModuleList` of blocks (not fused) so FiLM scale/shift can be applied between layers.
- `_split_film_params` handles both `[B, D]` and `[T, N, D]` input shapes.
- `_MAIN_OBS_KEYS = ["max_coords_obs", "mimic_target_poses", "previous_actions"]` — morphology not in trunk concat.

**Why it failed**:

1. **Fanout bottleneck**: For a 6-layer × 1024-unit actor, the conditioner must produce `2 × 6 × 1024 = 12,288` outputs from a 64-unit network. Severe compression-to-expansion mismatch dilutes gradients across all conditioner outputs.

2. **Multiplicative instability**: Trunk gradients at layer `l` are scaled by `γ_l`. If `γ_l` drifts from 1.0 early in training, the effective learning rate becomes shape-dependent and unstable. Noisy gamma estimates per minibatch amplify this instability.

**Result**: Reward ≈ 0.40–0.45. The trunk could not converge to a stable feature representation under multiplicative noise from the conditioner.

**Status**: Stopped at 1d 17h. No path to recovery without architectural change.

---

### [Experiment] Run 3: Shape Embedding + Concat (`hhi_se_1024_motion`)

**Motivation**: Replace multiplicative FiLM conditioning with a simple learned projection — encode the 11-dim morphology into a 64-dim embedding via a shallow MLP, then concatenate with the observation before the trunk.

**Architecture**:
```
morphology_obs (11-dim: gender + betas/3)
    → Linear(→ 64) → SiLU
    → shape_embed (64-dim)
                        │
[main obs (400–600+ dim)] ──cat──→ standard 6-layer 1024-unit trunk → output
```

| Property | FiLM | Shape Embed + Concat |
|---|---|---|
| Conditioner output size | 2 × 6 × 1024 = 12,288 | 64 |
| Trunk coupling | Multiplicative (γ × h + β) | Additive (concat) |
| Gradient stability | Trunk grads scaled by γ | Trunk grads unaffected |

**Files**: `protomotions/agents/common/shape_embed_mlp.py`, `examples/experiments/mimic/mlp_shape_embed.py`.
Config knobs: `cond_hidden_units` (e.g. `[64]`), `cond_activation` (default `silu`), `beta_norm_scale` (default `3.0`).

**Result**: Performance almost identical to the baseline (reward ≈ 0.84). The nonlinear projection did not improve over raw concat; the trunk learns an equivalent representation either way.

**Status**: Stopped at 1d 19h. Performance parity confirmed; no further upside expected.

---

### [Experiment] Run 4: Physics Features (`hhi_physics_feat_1024`)

**Motivation**: Raw betas are PCA coefficients in appearance space with no direct physical interpretation. Replace `morphology_obs` (11-dim betas) with `physics_obs` (15-dim z-scored features extracted from each body's MJCF). Gender is implicitly encoded in the physics features.

**The 15 physics features** (z-scored across 128 training bodies):

| Feature | Mean | Std | Range | Units |
|---|---|---|---|---|
| `total_mass` | 73.4 | 25.7 | 26.4 – 144.4 | kg |
| `l_thigh_length` | 0.379 | 0.036 | 0.298 – 0.457 | m |
| `l_shin_length` | 0.409 | 0.045 | 0.303 – 0.502 | m |
| `r_thigh_length` | 0.381 | 0.035 | 0.302 – 0.461 | m |
| `r_shin_length` | 0.405 | 0.043 | 0.304 – 0.495 | m |
| `l_upper_arm_len` | 0.256 | 0.023 | 0.212 – 0.304 | m |
| `l_forearm_len` | 0.253 | 0.025 | 0.201 – 0.302 | m |
| `r_upper_arm_len` | 0.256 | 0.024 | 0.209 – 0.305 | m |
| `r_forearm_len` | 0.258 | 0.025 | 0.210 – 0.306 | m |
| `torso_height` | 0.307 | 0.035 | 0.234 – 0.394 | m |
| `neck_head_height` | 0.304 | 0.029 | 0.247 – 0.379 | m |
| `hip_width` | 0.126 | 0.022 | 0.085 – 0.193 | m |
| `shoulder_width` | 0.358 | 0.048 | 0.251 – 0.472 | m |
| `leg_length` | 0.787 | 0.079 | 0.603 – 0.956 | m |
| `total_height` | 1.399 | 0.125 | 1.132 – 1.666 | m |

**New files**: `tools/extract_smpl_physics_features.py`, `protomotions/data/assets/mjcf/smpl_mor/physics_features.pt`, `examples/experiments/mimic/mlp_physics.py`.

**Modified files**: `simulator.py` (`_build_physics_features()`), `context_views.py` (`env_physics_features` field), `env.py` (`_build_global_context()`), `obs/humanoid.py` (`compute_physics_obs()`), `component_factories.py` (`physics_obs_factory()`).

**Status**: **Converged.** Reward ≈ 0.84 — identical to baseline. Physics features provide no advantage over raw beta concat. Floor-contact motions remain the persistent failure class regardless of morphology encoding mechanism.

#### [Reference] Physics Feature Derivations

**Limb lengths** — from body `pos` vectors (relative offset from parent joint in T-pose):
```
limb_length = norm(body.pos)
```
Body hierarchy: `Pelvis → L_Hip → L_Knee (thigh) → L_Ankle (shin)`, and for arms: `Torso → Spine → Chest → L_Thorax → L_Shoulder → L_Elbow (upper arm) → L_Wrist (forearm)`.

**Torso/head heights** — summed segment lengths:
```
torso_height     = norm(Torso.pos) + norm(Spine.pos) + norm(Chest.pos)
neck_head_height = norm(Neck.pos)  + norm(Head.pos)
```

**Hip/shoulder widths** — lateral global positions (Y axis = lateral):
```
hip_width      = |L_Hip.pos_y - R_Hip.pos_y|
shoulder_width = |global_y(L_Shoulder) - global_y(R_Shoulder)|
```

**Total mass** — from geom density × volume. Three geom types: capsule (`π r² L + (4/3) π r³`), box (`8 × sx × sy × sz`), sphere (`(4/3) π r³`). Total spans 26–144 kg (5.5× range).

**Biomechanically-grounded derived features (not yet implemented — see README.todo.md Track A6):**

| Feature | Formula | Predicts |
|---|---|---|
| `v_preferred_walk` | `sqrt(0.5 × g × l_leg)` | Natural walking speed (Froude, Alexander 1984) |
| `T_step_natural` | `2π × sqrt(l_leg / g)` | Natural step timing (inverted pendulum) |
| `f_upper_mass` | `m_upper / m_total` | Squat/kneel torque demand |
| `tau_knee_proxy` | `m_upper × l_thigh` | Required knee torque (interaction term) |
| `I_swing_leg` | `m_thigh×(0.37 l_thigh)² + m_shank×(l_thigh + 0.28 l_shin)²` | Leg repositioning speed |
| `cormic_index` | `torso_height / total_height` | COM fraction, squat form |

---

### [Train] Run 5: Hard Clip Fine-Tune (`hhi_1024_motion_tune`)

**Motivation**: Fine-tune the converged baseline checkpoint exclusively on the 192 hard clips (crawl/kneel/squat/backward, `--min-avg-betas 5.0`). These clips fail consistently across all shape variants.

**Motion file**: `failed_clips.pt` — 192 clips × 128 body shapes = 24,576 motions (~10 GB). Hard clips selected by average body distance > 5.0 across betas at baseline convergence.

**Results** (as of 2026-06-16):
- `eval/success_rate`: ≈ 0.80 (+15 pp over baseline cluster at ~0.65)
- `unnormalized_task_rewards`: > 0.90 (up from ~0.84)
- `eval/normalized_jerk_mean`: ~2000–2500 (3–4× higher than all baselines at ~500–800)
- `eval/high_jerk_frame_percentage_mean`: ~35–40% (vs ~10–15% for baselines)
- `eval/max_joint_error/max`: slightly elevated vs baselines

**Interpretation**: Success rate improved but motion quality degraded severely. The policy fights to hold difficult configurations against gravity instead of moving through them smoothly — classic narrow-dataset overfitting.

**Root cause of jerk**: Policy must output large actions from neutral posture (`q_neutral`) to reach the reference. Residual PD control (`q_target = q_ref + scale*action`) is the structural fix — at `action=0` the controller already tracks the reference, eliminating large corrective actions.

**Status**: **Converged.** High success rate (+15 pp) but unacceptable jerk (3–4×). Residual PD control is the next training intervention.

---

### [Reference] Training Speed Reference (4× A40)

| Run | Envs / Batch | Step time | Samples/hour |
|---|---|---|---|
| Non-FiLM runs | 4096 / 16384 | ~22 s/step | ~2.7 M |
| FiLM run | 8192 / 32768 | ~34 s/step | ~3.5 M |

Bottleneck is IsaacGym physics sim (30–50% GPU util), not NN compute. A40 ≈ A100 for this workload.

================================================================================

## 6. Bug Fixes

### [Fix] IsaacGym Crash at 4096+ Envs (CUDA Error 700 / Segfault)

**Root cause**: `SimulatorConfig` defaults to 5 projectile cubes per env. Each cube is a dynamic rigid body that PhysX tracks contact patches for against the triangle mesh terrain (wildcard broadphase coverage):

```
5 cubes × 4096 envs × ~4 patches each ≈ 80,000+ patches  ≥  maxRigidPatchCount limit (~80K)  ✗
5 cubes × 1024 envs × ~4 patches each ≈ 20,000 patches                                        ✓
```

`maxRigidPatchCount` is set at PhysX compile time and is not exposed via Python API. (NVlabs/ProtoMotions PR #223)

**Fix 1 — Cap IsaacGym to 1 projectile by default** (`protomotions/simulator/isaacgym/config.py`):
```python
@dataclass
class IsaacGymSimulatorConfig(SimulatorConfig):
    projectile: ProjectileConfig = field(
        default_factory=lambda: ProjectileConfig(num_projectiles=1),
    )
```
To opt back in: `--overrides simulator.projectile.num_projectiles=5`

**Fix 2 — Give each hidden cube a unique z slot** (`protomotions/simulator/base_simulator/config.py`):
```python
@dataclass
class ProjectileConfig:
    hide_z: float = -2.0
    hide_spacing: float = 4.0   # z-gap between hidden slots

    def hidden_z_for_index(self, projectile_index: int) -> float:
        return self.hide_z - self.hide_spacing * projectile_index
        # projectile 0 → z = -2.0, projectile 1 → z = -6.0, etc.
```

**Fix 3 — Spread hidden cubes across x/y by env_id** (`_set_projectile_root_states` in each backend):
```python
positions = positions.clone()
hidden_mask = positions[:, 2] <= self._proj_config.hide_z
if hidden_mask.any():
    hidden_env_offsets = env_ids[hidden_mask].to(positions.dtype)
    positions[hidden_mask, 0] = hidden_env_offsets   # x = env_id
    positions[hidden_mask, 1] = hidden_env_offsets   # y = env_id
```

**Fix 4 — Early-return in `_throw_projectile` when `num_projectiles == 0`** (`base_simulator/simulator.py`):
```python
def _throw_projectile(self) -> None:
    if self._proj_config.num_projectiles == 0:
        return
    ...
```

---

### [Fix] Multi-GPU Deadlock (NCCL P2P Hang)

**Root cause**: NCCL P2P initialization conflicts with IsaacGym's CUDA context. `cudaIpcGetMemHandle()` call silently hangs (30-min timeout) because IsaacGym already holds the CUDA context.

**Fix**: `NCCL_P2P_DISABLE=1` set via `os.environ.setdefault` in `train_agent.py`. Forces NCCL to use PCIe copy-reduce path instead of NVLink P2P. Performance impact negligible (<2 ms/backward pass, well under 1% of epoch time when sim dominates).

**Diagnostic if it recurs**:
```bash
NCCL_DEBUG=WARN python -u protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 1024 --batch-size 16384 --ngpu 2
```
Last print before hang identifies the exact deadlock point.

================================================================================

## 7. Evaluator Improvements

### [Overview] Evaluator Design

The evaluator runs every 200 epochs and serves two purposes:
- **Logging**: records per-frame robot state → computes smoothness, jitter, success rates → logs to wandb
- **Curriculum update**: failed motions get `weight = 1.0`, succeeded motions get discounted (`0.999^200 ≈ 0.82`)

---

### [Optimization] `eval_one_shape_per_motion` — Evaluator Sampling Strategy

**Problem**: With ~131k total motions (1024 clips × 128 shapes), the evaluator iterated every motion each cycle, running for hours and never completing on a single GPU.

**Commit history**:

| Commit | Change | Outcome |
|---|---|---|
| `42b48cf` | `max_eval_motions=2000` — cap randomly sampled motions per eval | Memory-safe but misses 13% of clips each cycle |
| `a08ccd7` | `eval_one_per_shape` — 1 clip per body shape (128 evals) | Wrong direction: covers all shapes but misses most clips |
| `8e74268` | `eval_one_shape_per_motion` — 1 shape per clip (N evals) | Correct: covers every clip, random shape per cycle |

**Final approach** (`eval_one_shape_per_motion=True`, default):
- For each of the N clips, draw one random body shape → N eval motions total
- N = 1024 clips — well within the 8192-env budget
- Each eval cycle samples a fresh random shape assignment → uniform shape coverage over many cycles

**Code** (`protomotions/agents/evaluators/config.py`):
```python
eval_one_shape_per_motion: bool = field(default=True, ...)
max_eval_motions: Optional[int] = field(default=2000, ...)  # ignored when above is True
```

**Code** (`protomotions/agents/evaluators/mimic_evaluator.py`):
```python
def _sample_one_shape_per_motion(self) -> torch.Tensor:
    all_ids = torch.stack(list(asset_to_motion_ids.values()), dim=0)  # [num_shapes, num_clips]
    shape_picks = torch.randint(num_shapes, (num_clips,), device=all_ids.device)
    clip_idx = torch.arange(num_clips, device=all_ids.device)
    return all_ids[shape_picks, clip_idx].sort().values
```

**Key assumption**: `build_asset_id_to_motion_ids()` accumulates motion IDs in consistent clip order across shapes. Position `i` in every shape's list refers to the same underlying clip. This makes column-wise random selection correct.

---

### [Optimization] Clip-Level Curriculum Propagation

**Problem**: When clip X fails under `shape_A`, only motion ID `(clip_X, shape_A)` gets `weight = 1.0`. The other 127 shape variants keep their old (possibly low) weights. Cross-run comparison was also noisy because different random shape draws dominated the failure set.

**Fix**: Expand weight updates to all 128 shape variants of any failed/succeeded clip before applying them.

```python
# In _update_motion_sampling_weights, after mapping local → global IDs:

# Log unexpanded failures (specific clip+shape pairs that actually failed)
self._save_failed_motions(global_failed.tolist(), self.agent.current_epoch)

# Expand to all shape variants before updating weights
if self.motion_lib.has_morphology_metadata():
    global_failed = self._expand_to_clip_variants(global_failed)
    global_success = self._expand_to_clip_variants(global_success)
```

`_expand_to_clip_variants` uses the same `[num_shapes, num_clips]` matrix as `_sample_one_shape_per_motion`: given a set of global motion IDs, finds their clip column indices and returns all shape variants.

**Effect**: After one eval cycle where clip X fails under `shape_A`:
- Before: only `(clip_X, shape_A)` gets `weight = 1.0`
- After: all 128 shape variants of clip X get `weight = 1.0` immediately

**Condition**: only activated when `motion_lib.has_morphology_metadata()` is True — no-op for single-morphology datasets.

---

### [Fix] `max_motions` — Inference with Large Motion Files

**Problem**: `humos_131072_0000_offset.pt` is 3.6 GB. Loading it during inference with 16 envs exceeds GPU memory.

**Fix**: Added `max_motions: Optional[int]` to `MotionLibConfig` (default `None`). When set, `load_from_file` loads the full file to CPU, slices to the first N motions, then moves to GPU.

**Usage**: `--max-motions 1024` — with 128 shapes, set ≥ 128 so morphology-consistent sampling has at least one motion per shape. 1024 is comfortable (~30 MB GPU vs 3.6 GB).

================================================================================

## 8. Evaluation on hhi_1024_motion_tune

### [Command] Inference (visual)

```bash
python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_1024_motion_tune/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt \
    --compact-spawn-spacing 1.5 --num-envs 16 --max-motions 128

python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_1024_motion_tune/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/failed_clips.pt \
    --compact-spawn-spacing 1.5 --num-envs 16 --max-motions 128
```

---

### [Command] E1 — Systematic Fault Evaluation

**Motion file choice**: MotionLib's slurmrank loader requires distributed (multi-GPU) setup.
For single-machine evaluation the motion file must be a single `.pt` file.
Two options depending on scope:

**Local (hard clips only — 192 clips × 128 shapes = 24,576 motions):**
Evaluates the exact set fine-tuned on. Most informative for measuring tune improvement.
```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion_tune/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/failed_clips.pt \
    --num-envs 256 \
    --headless \
    --output results/hhi_1024_motion_tune/eval_failed_clips/fault_report.csv
```
Expected runtime: ~256 envs × 96 batches × ~150 frames ≈ 30–40 min on one A40/4090.

**RunPod (full 1024 clips × 128 shapes = 131,072 motions):**
Complete evaluation for the paper. Requires the merged training file on `/workspace`.
```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion_tune/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 256 \
    --headless \
    --output results/hhi_1024_motion_tune/eval_full_1024/fault_report.csv
```

---

### [Reference] E1 CSV Output Schema

One row per motion. Columns:

| Column | Used by |
|---|---|
| `gender`, `beta_key` | E4 per-shape grouping |
| `mean_body_dist` | E2 success threshold, E5 cross-shape variance, E6 shape extremity scatter |
| `min_root_height` | E2 fall detection (threshold 0.5 m) |
| `max_body_dist`, `mean_root_dist`, `max_root_dist` | supplementary |
| `motion_id`, `asset_id`, `motion_clip_id` | join back to clip metadata |
| `steps_seen` | sanity check (should equal motion frames) |

---

### [Plan] E2–E6 Post-Processing (pandas, no re-simulation)

All derived from the E1 CSV + the motion file's `motion_betas` tensor:

- **E2 Success rate**: `(min_root_height > 0.5) & (mean_body_dist < 0.5)` per row → overall %
- **E4 Per-shape histogram**: group by `(gender, beta_key)`, compute per-shape success rate → histogram over 128 shapes
- **E5 Cross-shape variance**: group by `motion_clip_id`, compute `std(mean_body_dist)` across 128 shapes → mean and max std
- **E6 Shape extremity scatter**: join beta L2 norms from `motion_betas` → scatter `‖β‖₂` vs `mean_body_dist`, fit OLS

---

### [Results] Smoke Test — shard 0, 2 clips × 128 shapes (256 motions)

**Command:**
```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion_tune/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt \
    --num-envs 256 \
    --headless \
    --output evaluation/smoke_test_shard0.csv \
    --overrides motion_lib.max_motions=256
```

**Output:** `evaluation/smoke_test_shard0.csv`

| Metric | Value |
|---|---|
| Motions evaluated | 256 (2 clips × 128 shapes) |
| mean_body_dist mean | 1.027 m |
| mean_body_dist max | 1.854 m |
| max_body_dist max | 355.7 m |
| Success rate (root>0.5m & body_dist<0.5m) | 2.3% (6/256) |
| Physics explosions (max_body_dist>100m) | 243/256 (95%) |

**Interpretation:**
- `max_body_dist` values of 50–355m are physics instability: individual body parts (hands/feet) reaching degenerate constraint states while the root pelvis stays grounded (mean `min_root_height` ≈ 0.9m)
- 2.3% success rate and 1m mean body dist indicate near-total failure on these 2 shard-0 clips
- **Expected behaviour**: the fine-tune was trained exclusively on `failed_clips.pt` (192 hard clips). Shard-0 clips are different motions the fine-tune never saw — high error here is **catastrophic forgetting on easy clips**
- This is a key paper finding: fine-tuning on hard clips improves those clips at the cost of general tracking quality
- Next: run on `failed_clips.pt` to measure actual improvement on the clips the model was trained on; run baseline on shard-0 to confirm forgetting quantitatively

---

### [Results] Partial Failed-Clips Eval — tune checkpoint, 8 clips × 128 shapes (1024 motions)

**Memory constraint**: `failed_clips.pt` (24,576 motions) is 10.86 GB — too large for local RTX 4060 (7.72 GB).
Max safe `max_motions` with `num_envs=256`: ~1024. Full 192-clip eval must run on RunPod.

**Command:**
```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion_tune/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/failed/failed_clips.pt \
    --num-envs 256 \
    --headless \
    --output evaluation/tune_failed_clips_1k.csv \
    --overrides motion_lib.max_motions=1024
```

**Output:** `evaluation/tune_failed_clips_1k.csv`

| Metric | Tune (failed clips) | Tune (shard-0 easy clips) |
|---|---|---|
| Motions evaluated | 1024 (8×128) | 256 (2×128) |
| mean_body_dist mean | **0.750 m** | 1.027 m |
| Success rate | **14.6%** | 2.3% |
| Falls (root ≤ 0.5m) | 33.0% | ~0% |
| Drifts (root ok, bad tracking) | 52.4% | ~97% |

**Failure mode breakdown (key paper finding):**
- **Falls** (33%): root height drops below 0.5m — complete balance loss. Mean body dist 0.68m
- **Drifts** (52%): root stays up (~0.9m) but body tracking fails — consistent with the 3–4× jerk reported in training. Mean body dist 0.91m
- **Successes** (14.6%): body_dist < 0.5m AND root > 0.5m — balanced across genders (female 14.6%, male 14.5%)

Two visible clusters in worst-performing motions:
1. `min_root_height ≈ 0.1m` → clear fall (floor-contact motions where gravity wins)
2. `min_root_height ≈ 0.9m` → no fall but large drift → jerk/high-frequency oscillations

**Interpretation:** Tune model performs noticeably better on the hard clips it was trained on (0.75m vs 1.03m on easy clips, 14.6% vs 2.3% success). But 14.6% success on crawl/kneel/squat is still low — residual PD (TODO A1) is the structural fix needed.

================================================================================

## 9. Transfer Training Setup (2026-06-17)

### [Analysis] Evaluation Diagnosis

Evaluation of `hhi_1024_motion_tune` on two motion sets reveals:

| Dataset | Physics explosion rate | Stable-run success |
|---|---|---|
| Shard-0 easy clips (2 clips × 128 shapes) | 95% | 46% |
| Hard clips 1k (8 clips × 128 shapes) | 62% | 38% |

**Primary failure mode: physics explosions** (`max_body_dist > 100m`). The tune checkpoint's 3–4× jerk causes IsaacGym's constraint solver to blow up — limbs reach degenerate states while the root stays grounded. Among the 38% of hard-clip runs that don't explode, 38.3% succeed and 44.2% fall. The tracking quality itself is reasonable when stable.

**Shard-0 catastrophic forgetting is almost entirely explosions**, not true forgetting — only 5% of shard-0 runs stay numerically stable, of which 46% succeed.

### [Fix] Reward Changes in `mlp.py` for Transfer Run

**1. Action smoothness weight: `-0.02` → `-0.05`**

`compute_action_smoothness = ‖a_t - a_{t-1}‖₂ × weight`. Increased penalty directly discourages the large consecutive action deltas that destabilize the physics solver.

**2. Contact force change penalty added (new)**

```python
"contact_force_change_rew": contact_force_change_rew_factory(
    weight=-1e-4, threshold=30.0, zero_during_grace_period=True
)
```

`compute_contact_force_change_rew = clamp(|F_t - F_{t-1}| - threshold, min=0).sum()`. Penalizes sudden contact force spikes above 30 N. The infrastructure (`current_contact_force_magnitudes`, `prev_contact_force_magnitudes`) is already populated every step — this is zero-overhead to add.

### [Command] Transfer Training Command

```bash
nohup python -u protomotions/train_agent.py \
      --robot-name smpl_mor \
      --simulator isaacgym \
      --experiment-path examples/experiments/mimic/mlp.py \
      --experiment-name hhi_1024_transfer \
      --motion-file /workspace/merged4/humos_slurmrank.pt \
      --checkpoint results/hhi_1024_motion/last.ckpt \
      --num-envs 4096 \
      --batch-size 16384 \
      --use-wandb \
      --wandb-project hhi-protomotions \
      --wandb-entity yugoamaryl \
      --wandb-group hhi_1024_transfer > /tmp/train_hhi_transfer.log 2>&1 &
```

Motion sampling weights restart fresh (desired — improved `eval_one_shape_per_motion` sampling strategy replaces old curriculum state).

================================================================================

## 10. Transfer Training Fix (2026-06-18)

### [Fix] Revert Smoothness and Contact Force Changes (commit 02c18d6)

The reward changes introduced in the previous section caused `WARNING:tensorboardX.x2num:NaN or Inf found in input tensor` every epoch during `hhi_1024_transfer`. Root cause: `contact_force_change_rew` returns raw unbounded Newton values (`force_changes.sum(dim=-1)`), which overflow float16 in the logging pipeline (`episode_env_tensors` stores as float16, max ~65504). Large SMPL body shapes with diverse motions can produce contact force changes far exceeding this limit, silently producing `inf` in TensorBoard/WandB logs.

Reverted in `mlp.py`:
- `action_smoothness` weight: `-0.05` → `-0.02` (restored original)
- `contact_force_change_rew` reward component: removed entirely

### [Command] Transfer Training Command (with 4 GPUs)

```bash
nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp.py \
  --experiment-name hhi_1024_transfer \
  --motion-file /workspace/merged4/humos_slurmrank.pt \
  --checkpoint results/hhi_1024_motion/last.ckpt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 4 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_1024_transfer > /tmp/train_hhi_transfer.log 2>&1 &
```

**Next to run on RunPod:** Full 192-clip evaluation of both tune and baseline checkpoints on `failed_clips.pt` and at least one full shard, to quantify forgetting vs improvement trade-off.

================================================================================

## 11. New Training Strategy — Two-Stage Curriculum (2026-06-19)

### [Analysis] Why the 717-clip Test Set Was Wrong

The original E7 held-out evaluation ran HUMOS inference on 717 HumanML3D test-split clips to test generalisation to new body shapes. This confounds **motion OOD** with **shape OOD**: if the policy fails, it is impossible to tell whether the cause is the new motion content or the new body shape. E7 is supposed to isolate shape generalisation only.

Root cause of the 717: `infer.py` concatenates train+val+test splits, skips keyids already in the rclone cache (`remote_index.txt`, 21,742 entries from prior training runs), and strips M-prefix from the test split. The 717 = 664 MLD test clips + 53 train/val stragglers — an accidental outcome of rclone cache state, not a deliberate split.

### [Strategy] Two-Stage Curriculum

**Stage 1 — Learn all motion content on a neutral body.**
Train on all 20,951 valid HumanML3D clips (from `valid_sorted.json`, difficulty-ordered) with betas = 0 (neutral SMPL shape). The policy sees the full diversity of human motion without any body-shape variation. Morphology obs is still passed (all zeros) — the policy learns to ignore it.

**Stage 2 — Transfer: introduce body shape variation.**
Fine-tune the Stage 1 checkpoint on the existing 1024-clip × 128-shape dataset (`humos_slurmrank.pt`). The policy already tracks all motion types; Stage 2 only teaches it to adapt per body shape.

**Why this is better:**
- After Stage 1 the policy has seen all 20,951 motions — any held-out evaluation clip is in-distribution for motion content.
- E7 (held-out betas on the same 1024 clips) then tests *only* shape generalisation — the confound is gone.
- Stage 2 is a smaller learning problem: shape adaptation on top of a converged motion prior.

### [Pipeline] Stage 1 Data Preparation — Neutral-Beta npz Export

**Source:** HUMOS `.tensor` files at `/home/hlz/repos/humos/datasets/humos3dfeats/`.
Each `.tensor` file contains the original AMASS motion (`root_orient`, `pose_body`, `trans`) at 20 FPS with the original actor's betas (non-zero, real person). We zero the betas and keep the poses as-is — the joint angles are nearly body-shape-agnostic, and the small kinematic inconsistency is acceptable for Stage 1.

**Motion IDs:** exactly the 20,951 keyids in `valid_sorted.json` (physically plausible, difficulty-ordered). Includes M-prefix (mirrored) clips.

**Script:** `tools/export_tensor_to_amass_npz.py`

```bash
# Run from ProtoMotions root
python tools/export_tensor_to_amass_npz.py \
    --out-root /home/hlz/datasets/amass_neutral \
    --skip-existing
```

**Output (already generated, 2026-06-19):**
```
/home/hlz/datasets/amass_neutral/
    HML3D/
        {keyid}_v00_{gender}_neutral.npz   # 20,951 files
    humanml3d_neutral_20951.yaml           # motion config for downstream pipeline
    humanml3d_neutral_20951_manifest.yaml  # clip metadata with difficulty scores
```

npz fields per clip:
| Field | Shape | Value |
|---|---|---|
| `poses` | (T, 66) | root_orient(3) + body(63), float32 |
| `trans` | (T, 3) | root translation, float32 |
| `betas` | (10,) | all zeros |
| `gender` | scalar str | original actor's gender (`male`/`female`) |
| `clip_id` | scalar str | HumanML3D keyid (e.g. `000005`, `M004501`) |
| `mocap_framerate` | scalar | 20.0 |

**Next steps** (not yet run):

Step 5 — Convert to MotionLib `.pt` (run in ProtoMotions root):
```bash
# First attempt used --batch-size 25000 → single 7.3 GB file → PyTorch miniz ZIP reader fails on >~4 GB.
# Correct command uses --batch-size 4096 → 6 chunks of ~1.4 GB each (safe limit ~3.5 GB).

python tools/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/amass_neutral \
    /home/hlz/datasets/humos_proto_neutral \
    --motion-config /home/hlz/datasets/amass_neutral/humanml3d_neutral_20951.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cpu \
    --batch-size 4096
```

Output (already generated, 2026-06-19):
```
/home/hlz/datasets/humos_proto_neutral/
    humanml3d_neutral_20951_0000.pt   # 4096 motions, ~1.4 GB
    humanml3d_neutral_20951_0001.pt   # 4096 motions
    humanml3d_neutral_20951_0002.pt   # 4096 motions
    humanml3d_neutral_20951_0003.pt   # 4096 motions
    humanml3d_neutral_20951_0004.pt   # 4096 motions
    humanml3d_neutral_20951_0005.pt   # 471 motions
```

**Verified correct** — all 20,951 motions load cleanly with the right fields. Two expected differences from the HUMOS dataset:
- **20 fps** (not 30): `.tensor` files are at 20 fps; 20 has no divisor ≥ 30 so the converter keeps 20 fps (`motion_dt = 0.05`). `motion_dt` is stored per-motion so MotionLib interpolates correctly at training time.
- **Variable clip length 0.3s – 263.6s** (vs fixed 6.6s in HUMOS): `valid_sorted.json` applies no duration filter. Long clips are fine — MotionLib samples random start frames within each clip during training.

Requires a `neutral` entry in an `assets.yaml` pointing to a single neutral-body SMPL MJCF (one per gender). Create `male_neutral` and `female_neutral` assets from the existing smpl_mor XMLs with zero betas.

Also added `smpl_mor_neutral` to `protomotions/robot_configs/factory.py` — same as `smpl_mor` but with `asset_folder_name` and `asset_info_file` pointing to `mjcf/smpl_mor_neutral/`. Use `--robot-name smpl_mor_neutral` for Stage 1 training and visualization.

Step 5a — Create neutral SMPL MJCF assets (run from SMPLSim repo, `smplsim` conda env):

Generates `male_neutral_smpl.xml`, `female_neutral_smpl.xml` (betas = 0) and
`all_betas_neutral.pt = {"neutral": zeros(10)}`.

```bash
cd /home/hlz/repos/SMPLSim
conda run -n smplsim python run_neutral.py
```

Output:
```
protomotions/data/assets/mjcf/smpl_mor_neutral/male_neutral_smpl.xml
protomotions/data/assets/mjcf/smpl_mor_neutral/female_neutral_smpl.xml
protomotions/data/assets/all_betas_neutral.pt
```

Step 5b — Generate `assets.yaml` for the neutral MJCF folder (run from ProtoMotions root, any env):

```bash
cd /home/hlz/repos/ProtoMotions
python tools/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor_neutral \
    --betas-file protomotions/data/assets/all_betas_neutral.pt \
    --out protomotions/data/assets/mjcf/smpl_mor_neutral/assets.yaml
```

This produces 2 entries in the YAML (`male_neutral` and `female_neutral`), each with
`betas=[0,0,...,0]` and `root_height=0.95` (default; adjusted after Step 6 grounding).

Step 6 — Frame-0 grounding offset (IsaacGym):
```bash
bash tools/run_frame0_offsets_neutral.sh
```
Runs `compute_humos_frame0_offsets.py` on all 6 chunks against `mjcf/smpl_mor_neutral` assets.
Output: `humos_proto_neutral/humanml3d_neutral_20951_{0000-0005}_offset.pt`

Step 6b — Visualize motion (kinematic playback, no trained policy needed):
```bash
# Before Step 6 (raw, may float/clip through floor):
python examples/env_kinematic_playback.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --num-envs 4 \
    --motion-file /home/hlz/datasets/humos_proto_neutral/humanml3d_neutral_20951_0000.pt \
    --experiment-path examples/experiments/mimic/mlp.py

# After Step 6 (grounded):
python examples/env_kinematic_playback.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --num-envs 4 \
    --motion-file /home/hlz/datasets/humos_proto_neutral/humanml3d_neutral_20951_0000_offset.pt \
    --experiment-path examples/experiments/mimic/mlp.py
```

Step 7 — Stage 1 training (RunPod):

Upload all 6 offset chunks first:
```bash
rsync -avz /home/hlz/datasets/humos_proto_neutral/*_offset.pt runpod:/workspace/humos_proto_neutral/
rsync -avz protomotions/data/assets/mjcf/smpl_mor_neutral/ runpod:/workspace/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor_neutral/
```

MotionLib loads one `.pt` per GPU. For single-GPU RunPod, pass one chunk (4096 motions — sufficient for Stage 1 warm-up). For multi-GPU, use the slurmrank file-per-rank mechanism (see existing humos training setup for reference).

```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_stage1_neutral \
    --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
    --num-envs 4096 \
    --batch-size 16384
```

Step 8 — Stage 2 transfer (RunPod, from Stage 1 checkpoint):
```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_stage2_transfer \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --checkpoint results/hhi_stage1_neutral/last.ckpt \
    --num-envs 4096 \
    --batch-size 16384
```

------

Humos neutral dataset, used for baseline training

```bash
rclone copy /home/hlz/datasets/humos_proto_neutral/offset/ \
      r2:proto-data/20946_neutral_offset/ \
      --transfers=2 \
      --s3-upload-concurrency=4 \
      --s3-chunk-size=64M \
      --retries=10 \
      --retries-sleep=30s \
      --low-level-retries=20 \
      --progress
```

------

## [Design Question] Two-Stage Transfer Learning

**Context:** Previous runs trained directly on 1024 motions × 128 body shapes with beta + physical params concatenated into obs. Rewards/unnormalized_task_rewards >0.9, eval/success_rate ~0.7–0.8.

New approach: train Stage 1 on 20k neutral motions (all beta=0), then transfer to Stage 2 on 128-shape motion data.

### Q1 — How do we design the NN for Stage 1 (neutral, beta=0)?

Since all bodies are neutral (beta=0), the morphology input is constant and uninformative. Design options:
- No morphology input at all — pure motion-tracking policy on neutral SMPL
- Include a zeroed beta slot (padded) so the Stage 2 fine-tune can activate it without architecture change
- Include physical params only (body proportions/masses derived from beta=0) to keep the interface consistent

Key tension: a simpler Stage 1 (no morphology branch) learns faster but may need architectural surgery before Stage 2. A beta-aware Stage 1 (with frozen zero-betas) is ready to fine-tune but adds dead parameters during Stage 1.

### Q2 — How do we design the transfer learning process for Stage 2 (128-shape, large data)?

The full 128-shape dataset is too large to hold in memory at once — need a rolling/streaming strategy. Design considerations:

- **Data rolling:** load shards incrementally (e.g. one shard at a time), cycle through all shards across epochs rather than loading all 128 shapes up front
- **Freezing strategy:** freeze the motion-tracking backbone, fine-tune only the morphology-conditioning layers first; then unfreeze and joint-fine-tune
- **Learning rate:** Stage 2 LR should be 5–10× lower than Stage 1 to preserve learned locomotion priors
- **Curriculum:** start Stage 2 with small beta magnitudes, gradually increase to full ±2σ variation
- **Checkpoint:** Stage 2 resumes from Stage 1 `last.ckpt` with `--checkpoint` flag; morphology obs (beta / physical params) must be re-enabled in the experiment config

------

## [Design Decision] Q1 Answer — Stage 1 NN Architecture

**Decision: use `smpl_mor_neutral` + `mlp.py` (flat concat), with `morphology_obs` always present.**

> **Note:** This section originally proposed `mlp_film.py`. That was superseded by empirical results:
> `hhi_film_1024_motion` (FiLM from scratch) stalled at reward ~0.40–0.45 vs ~0.84 for flat concat,
> due to fanout bottleneck and multiplicative instability. All non-FiLM alternatives (ShapeEmbed,
> physics features) matched flat concat but added no gain. `mlp.py` is the architecture winner
> for both Stage 1 and Stage 2.

### Key facts from codebase research

**Obs vector composition (mlp.py):**
| Key | Approx dim | Content |
|-----|-----------|---------|
| `max_coords_obs` | ~362 | body pos/rot/vel/angvel + root_h + contacts |
| `mimic_target_poses` | ~similar | future pose targets with velocities |
| `previous_actions` | 63 | last action (SMPL DOFs) |
| `morphology_obs` | **11** | `[gender_id, betas/3.0]` (1 + 10) |

`morphology_obs` is built in `simulator.py` as `cat([gender_id ∈ {-1,+1}, betas/3.0])`. For `smpl_mor_neutral`, betas are all-zero → morphology_obs = `[±1, 0, 0, …, 0]`.

**`smpl_mor_neutral` already populates `morphology_obs`** — same code path as `smpl_mor`, just different assets. No code change needed for Stage 1.

**Checkpoint loading is strict** — `load_state_dict()` with no `strict=False`. Obs dims must be identical between Stage 1 and Stage 2 for the checkpoint to load directly.

### Why `morphology_obs` must be included in Stage 1 (even though betas=0)

- Identical obs dims → direct `load_state_dict()` at Stage 2, no architecture surgery
- Running mean/std normalizer accumulates morphology stats during Stage 1 (near-zero); reset with `tools/reset_morphology_normalizer.py` before Stage 2 to avoid saturated beta inputs
- Network learns the gender signal during Stage 1; betas are just an uninformative constant the network learns to ignore — exactly the right prior going into transfer

### Stage 1 training command (RunPod)

```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_stage1_neutral \
    --motion-file /workspace/humos_proto_neutral/humanml3d_neutral_20946_0000.pt \
    --num-envs 4096 \
    --batch-size 16384
```

*(shard 0000 only for single-GPU; for multi-GPU use the slurmrank mechanism with all 6 shards)*

> **Note on `ckpt/20951_neutral.zip` on R2:** This is a 199-epoch checkpoint trained on the
> raw 20951-motion shards (mixed female_neutral/male_neutral asset IDs, before the fix).
> It does NOT correspond to the corrected 20946-motion offset shards used for training now.
> Do not use it as a warm-start for Stage 1.

### Stage 2 transfer command (RunPod, from Stage 1 ckpt)

```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_stage2_transfer \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --checkpoint results/hhi_stage1_neutral/last.ckpt \
    --num-envs 4096 \
    --batch-size 16384
```

**Before Stage 2:** run `tools/reset_morphology_normalizer.py` on the Stage 1 checkpoint to reset beta normalizer dims (var[-10:] → 1.0) before transferring.

------

## [Implementation] Obs Normalizer Reset Before Stage 2

### Why this is necessary

`mlp.py` uses a single `RunningMeanStd` over the full concatenated obs vector (shape `(1014,)`).
The obs concat order is: `max_coords_obs | mimic_target_poses | previous_actions | morphology_obs`.
`morphology_obs` (11 dims) = `[gender_id, betas/3.0]` is always **last**.

After Stage 1 (neutral, beta=0):
- `var[-10:]` (the 10 beta dims) ≈ **0** — input was a constant zero throughout training
- `var[-11]` (gender) ≈ **1.0** — fine, gender varied normally

When Stage 2 introduces real betas, normalization divides by `sqrt(~0 + 1e-5) ≈ 0.003`,
pushing all beta values to `±167` before the clamp truncates them to `±5`.
The network sees saturated, uninformative beta inputs for thousands of steps until the
running stats catch up (count ≈ 6e9 after Stage 1 → very slow adaptation).

### Fix: `tools/reset_morphology_normalizer.py`

Resets `mean[-11:]` = 0.0 and `var[-11:]` = 1.0 in both actor and critic normalizers.
`count` is left untouched so locomotion dim stats keep their momentum.
Saves to a new file — does not overwrite the original checkpoint.

### Commands

```bash
# 1. Dry-run: inspect the dead beta dims after Stage 1 completes
python tools/reset_morphology_normalizer.py \
    --checkpoint results/hhi_stage1_neutral/last.ckpt \
    --dry-run

# Expect to see var[-10:] ≈ 0 on all beta dims — that confirms the problem.
# var[-11] (gender) should still be ≈ 1.0.

# 2. Apply the reset (saves alongside original, does NOT overwrite)
python tools/reset_morphology_normalizer.py \
    --checkpoint results/hhi_stage1_neutral/last.ckpt \
    --output results/hhi_stage1_neutral/last_morph_reset.ckpt

# 3. Stage 2 uses the reset checkpoint
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_stage2_transfer \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --checkpoint results/hhi_stage1_neutral/last_morph_reset.ckpt \
    --num-envs 4096 \
    --batch-size 16384 \
    --overrides agent.config.actor_optimizer.lr=2e-6 agent.config.critic_optimizer.lr=1e-5
```

Stage 2 LR is 10× lower than Stage 1 defaults (`actor: 2e-5 → 2e-6`, `critic: 1e-4 → 1e-5`).
No freezing — `mlp.py` has no clean structural boundary to freeze at.

---

## Physics Features vs Raw Betas — Transfer Inference Finding (2026-06-21)

Both `hhi_1024_transfer` (raw betas, 11-dim) and `hhi_phy_1024_transfer` (physics features, 15-dim) converged to the same training reward (~0.84). However, on inference:

- `hhi_phy_1024_transfer`: **5/8 envs successfully followed**
- `hhi_1024_transfer`: **0/8 envs successfully followed**

Training reward parity is a poor proxy for transfer robustness. Physics features (mass, limb lengths, hip/shoulder/leg width, height) provide causally meaningful conditioning; the policy can learn actual control laws from them. Raw betas are abstract SMPL latents — the policy implicitly decodes them and apparently fails under transfer pressure.

Reproduce with:

```bash
# physics features (5/8)
python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_phy_1024_transfer/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0001_offset.pt \
    --compact-spawn-spacing 1.5 --num-envs 8 --max-motions 128

# raw betas (0/8)
python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_1024_transfer/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0001_offset.pt \
    --compact-spawn-spacing 1.5 --num-envs 8 --max-motions 128
```

> TODO: confirm at larger scale (32–64 envs) and check if `humos_131072_0001_offset.pt` shard is in-distribution for transfer training.


------

nohup rclone copy /media/hlz/R/humos_output_interp gdrive:humos_output_interp \
      --progress \
      --transfers 4 \
      --checkers 8 \
      --drive-chunk-size 64M \
      --log-file rclone_upload_interp.log \
      --log-level INFO > rclone_upload_interp_stdout.log 2>&1 &



================================================================================

## Video Recording — 64 Clips (2026-06-23)

32 motions sampled across 16 offset files (2 per file, varied small indices),
recorded with both checkpoints → 64 videos total.

Outputs: `/home/hlz/Videos/hhi_phy/` and `/home/hlz/Videos/hhi/`

### Example (full command)

```bash
python protomotions/record_video_mor.py \
  --checkpoint /home/hlz/repos/ProtoMotions/results/hhi_phy_1024_transfer/score_based.ckpt \
  --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt \
  --simulator isaacgym \
  --num-envs 16 \
  --compact-spawn-spacing 2.0 \
  --motion-index 41 \
  --output /home/hlz/Videos/hhi_phy/f0000_m041.mp4
```

### Remaining 63 commands (ckpt / motion-file / motion-index / output filename)

Checkpoint paths:
- `hhi_phy` → `results/hhi_phy_1024_transfer/score_based.ckpt`
- `hhi`     → `results/hhi_1024_transfer/score_based.ckpt`

| ckpt    | file index | motion-index | output                  |
|---------|------------|--------------|-------------------------|
| hhi_phy | 0000       | 168          | f0000_m168.mp4          |
| hhi_phy | 0001       | 50           | f0001_m050.mp4          |
| hhi_phy | 0001       | 142          | f0001_m142.mp4          |
| hhi_phy | 0002       | 9            | f0002_m009.mp4          |
| hhi_phy | 0002       | 267          | f0002_m267.mp4          |
| hhi_phy | 0003       | 12           | f0003_m012.mp4          |
| hhi_phy | 0003       | 223          | f0003_m223.mp4          |
| hhi_phy | 0004       | 74           | f0004_m074.mp4          |
| hhi_phy | 0004       | 144          | f0004_m144.mp4          |
| hhi_phy | 0005       | 116          | f0005_m116.mp4          |
| hhi_phy | 0005       | 259          | f0005_m259.mp4          |
| hhi_phy | 0006       | 27           | f0006_m027.mp4          |
| hhi_phy | 0006       | 139          | f0006_m139.mp4          |
| hhi_phy | 0007       | 11           | f0007_m011.mp4          |
| hhi_phy | 0007       | 241          | f0007_m241.mp4          |
| hhi_phy | 0008       | 53           | f0008_m053.mp4          |
| hhi_phy | 0008       | 147          | f0008_m147.mp4          |
| hhi_phy | 0009       | 30           | f0009_m030.mp4          |
| hhi_phy | 0009       | 153          | f0009_m153.mp4          |
| hhi_phy | 0010       | 70           | f0010_m070.mp4          |
| hhi_phy | 0010       | 238          | f0010_m238.mp4          |
| hhi_phy | 0011       | 7            | f0011_m007.mp4          |
| hhi_phy | 0011       | 274          | f0011_m274.mp4          |
| hhi_phy | 0012       | 15           | f0012_m015.mp4          |
| hhi_phy | 0012       | 187          | f0012_m187.mp4          |
| hhi_phy | 0013       | 80           | f0013_m080.mp4          |
| hhi_phy | 0013       | 279          | f0013_m279.mp4          |
| hhi_phy | 0014       | 7            | f0014_m007.mp4          |
| hhi_phy | 0014       | 277          | f0014_m277.mp4          |
| hhi_phy | 0015       | 74           | f0015_m074.mp4          |
| hhi_phy | 0015       | 231          | f0015_m231.mp4          |
| hhi     | 0000       | 41           | f0000_m041.mp4          |
| hhi     | 0000       | 168          | f0000_m168.mp4          |
| hhi     | 0001       | 50           | f0001_m050.mp4          |
| hhi     | 0001       | 142          | f0001_m142.mp4          |
| hhi     | 0002       | 9            | f0002_m009.mp4          |
| hhi     | 0002       | 267          | f0002_m267.mp4          |
| hhi     | 0003       | 12           | f0003_m012.mp4          |
| hhi     | 0003       | 223          | f0003_m223.mp4          |
| hhi     | 0004       | 74           | f0004_m074.mp4          |
| hhi     | 0004       | 144          | f0004_m144.mp4          |
| hhi     | 0005       | 116          | f0005_m116.mp4          |
| hhi     | 0005       | 259          | f0005_m259.mp4          |
| hhi     | 0006       | 27           | f0006_m027.mp4          |
| hhi     | 0006       | 139          | f0006_m139.mp4          |
| hhi     | 0007       | 11           | f0007_m011.mp4          |
| hhi     | 0007       | 241          | f0007_m241.mp4          |
| hhi     | 0008       | 53           | f0008_m053.mp4          |
| hhi     | 0008       | 147          | f0008_m147.mp4          |
| hhi     | 0009       | 30           | f0009_m030.mp4          |
| hhi     | 0009       | 153          | f0009_m153.mp4          |
| hhi     | 0010       | 70           | f0010_m070.mp4          |
| hhi     | 0010       | 238          | f0010_m238.mp4          |
| hhi     | 0011       | 7            | f0011_m007.mp4          |
| hhi     | 0011       | 274          | f0011_m274.mp4          |
| hhi     | 0012       | 15           | f0012_m015.mp4          |
| hhi     | 0012       | 187          | f0012_m187.mp4          |
| hhi     | 0013       | 80           | f0013_m080.mp4          |
| hhi     | 0013       | 279          | f0013_m279.mp4          |
| hhi     | 0014       | 7            | f0014_m007.mp4          |
| hhi     | 0014       | 277          | f0014_m277.mp4          |
| hhi     | 0015       | 74           | f0015_m074.mp4          |
| hhi     | 0015       | 231          | f0015_m231.mp4          |

================================================================================

## 13. Residual PD Control (commit 5ac91ae6b3afd1eabb0c062f76b6f5e4e059251b)

**Motivation:** Standard PD uses `q_target = q_neutral_mid + scale * tanh(action)`. For floor-contact poses (crawl/kneel/squat), joints are far from neutral so the policy must sustain large outputs every step → jerk and instability. Residual PD sets `q_target = q_ref(t) + residual_scale * tanh(action)`, so `action=0` already tracks the reference exactly and the policy only corrects for dynamics.

### Code changes

| File | Change |
|---|---|
| `protomotions/envs/action/action_functions.py` | Added `make_residual_pd_action_config(robot_config, residual_scale=0.3)` — builds action config with uniform 0.3 rad scale and `use_residual_pd=True` flag |
| `protomotions/envs/action/__init__.py` | Exported `make_residual_pd_action_config` |
| `protomotions/envs/base_env/env.py` | Modified `_process_action` to inject `context.mimic.ref_state.dof_pos` as `pd_action_offset` when `use_residual_pd=True` |
| `examples/experiments/mimic/mlp_residual_pd.py` | New experiment config identical to `mlp.py` but using `make_residual_pd_action_config` |

### Smoke test command (8 envs, local)

```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_residual_pd.py \
    --experiment-name hhi_pd_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 --batch-size 16
```

### Fine-tune command (RunPod, from Stage 1 checkpoint)

```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_residual_pd.py \
    --experiment-name hhi_stage1_residual_pd \
    --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
    --checkpoint results/hhi_20946_neutral/last.ckpt \
    --num-envs 4096 --batch-size 16384
```

**Watch at epoch 0:** `env/terminate_mean` should stay ~0.001 and `info/episode_length` ~200. A spike to termination rate >0.3 means epoch-0 action overshoot — reduce `residual_scale` to 0.1 or rescale the loaded actor's final-layer weights by `old_pd_scale / residual_scale`.

================================================================================

## 12. Stage 1 Neutral Training — `hhi_20946_neutral` (2026-06-28)

### [Results] Training Progress at Epoch ~20,400 (~174 hours)

**Result dir:** `results/hhi_20946_neutral/`
**Dataset:** 20,946 neutral-body HumanML3D motions (betas=0, `smpl_mor_neutral`)
**Hardware:** 6× GPU (6 ranks), 4096 envs/rank, batch 16384

#### Success Rate Trajectory

| Epoch | Success Rate |
|-------|-------------|
| 200   | 0.4%        |
| 1000  | 29.1%       |
| 4200  | 50.8%       |
| 7400  | 65.9%       |
| 10400 | 71.4% (mid-plateau) |
| 15400 | 80.1%       |
| 18600 | 83.5%       |
| 20200 | **84.9%** (peak) |
| 20400 | 84.2%       |

Rapid gains through ~epoch 5000, then slow climb through 70–80%, now oscillating 82–85%. Confirmed plateau in last ~5000 steps (+3–4 pp gain).

#### Key Metrics (epoch 20400)

| Metric | Start | Current |
|--------|-------|---------|
| `eval/success_rate` | 0.4% | **84.2%** |
| `rewards/unnormalized_task_rewards` | 0.659 | **0.845** (flat) |
| `eval/gt_error/mean` (translation) | 1.32 m | **0.195 m** |
| `eval/gr_error/mean` (rotation) | 1.51 | **0.287** |
| `eval/gt_error/failure_rate` | 99.6% | **15.8%** |
| `info/episode_length` | ~5 steps | **~205 steps** |
| `env/terminate_mean` | 20.4% | **0.1%** |
| `times/training_hours` | — | **174 h** |

#### Reward Component Breakdown (raw, unnormalized)

| Reward | Start | Current | Status |
|--------|-------|---------|--------|
| `gt_rew` (translation) | 0.791 | 0.838 | Nearly flat |
| `gr_rew` (rotation) | 0.325 | **0.682** | Biggest remaining gap |
| `gv_rew` (velocity) | 0.456 | 0.910 | Well converged |
| `gav_rew` (angular velocity) | 0.250 | 0.732 | Converged |
| `rh_rew` (root height) | 0.805 | 0.924 | Converged |
| `contact_match_rew` | 0.411 | 1.545 | Converged |
| `pow_rew` (power) | 1134 | **196** | Good reduction |

#### Failure Analysis

- Unique failed motions at epoch 20400: **2,271** (from `failed_motions/`)
- Persistent failures (fail in all 5 most recent evals): **1,834** = **8.8%** of 20,946 dataset
- Inconsistent / borderline failures: ~437

#### Assessment

Plateau is real. Rotation tracking (`gr_rew`, `gr_error`) is the primary remaining bottleneck — converged more slowly than translation and velocity components. The ~16% failure rate is concentrated in ~1,800 hard motions (likely floor-contact poses: crawl/kneel/squat/backward, consistent with prior `hhi_1024_motion` findings). Diminishing returns suggest this is near the practical ceiling for Stage 1. Recommend using this checkpoint (or `epoch_20000.ckpt` / `score_based.ckpt`) as the Stage 2 warm-start.

================================================================================

## 14. Residual PD Transfer — `hhi_20946_neutral_rpd` (2026-06-29)

Fine-tune `hhi_20946_neutral/score_based.ckpt` (84.9% success) with residual PD on the same neutral dataset.
Adapts the policy to the new action mode before Stage 2 multi-shape transfer.

### [Command] Download checkpoint from R2 (RunPod)

```bash
rclone copy r2:proto-data/ckpt/20946_neutral.zip /workspace/ProtoMotions/ \
    --transfers=1 \
    --multi-thread-streams=16 \
    --multi-thread-chunk-size=128M \
    --progress
```

### [Command] Training (RunPod)

```bash
nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_residual_pd.py \
    --experiment-name hhi_20946_neutral_rpd \
    --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
    --checkpoint results/hhi_20946_neutral/score_based.ckpt \
    --num-envs 6144 --batch-size 24576 \
    --ngpu 6 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_neutral_rpd > /tmp/train_neutral_rpd.log 2>&1 &
```

================================================================================

## 15. Full RL Training Process — Deep Dive (2026-06-30)

### 1. The Training Loop (`BaseAgent.fit()`)

Each epoch is two phases: **rollout** (data collection, `torch.no_grad()`) then **optimization** (gradient steps).

```
for epoch in range(max_epochs):
    # Phase 1: Collect num_steps timesteps from num_envs parallel envs
    for step in range(num_steps):
        obs_td = env.reset(done_indices)          # reset terminated envs
        actor_output = model(obs_td)              # policy forward: action, mean_action, neglogp, value
        next_obs, rewards, dones, ... = env.step(actor_output["action"])
        experience_buffer.store(obs, action, reward, done, value, next_value)

    # Phase 2: Optimize on the collected batch
    normalize_rewards_in_buffer()
    pre_process_dataset()   # compute GAE advantages + returns
    optimize_model()        # multiple mini-epochs of PPO updates
```

**Scale in training runs:** 4096 envs × 32 steps × 6 GPUs = 786,432 samples/epoch. `max_epochs` is computed from `training_max_steps // world_size // num_envs // num_steps`.

---

### 2. Observation Vector

The actor sees a flat concatenation of 4 key groups:

| Key | ~Dim | Content |
|---|---|---|
| `max_coords_obs` | ~362 | global body pos/rot/vel/angvel, root height, contact flags |
| `mimic_target_poses` | ~similar | reference body poses + velocities (future targets) |
| `previous_actions` | 63 | last action output (SMPL DOFs) |
| `morphology_obs` | 11 | `[gender_id, beta_1/3, …, beta_10/3]` |

Total ~1014 dims. One `RunningMeanStd` normalizes the whole vector. **This is why the normalizer reset tool is critical before Stage 2** — beta dims have variance ≈ 0 after Stage 1, which would saturate them to ±167 before the clamp.

The critic gets the same keys (including `morphology_obs`), so it is not blind to body shape.

---

### 3. Action Pipeline — Standard PD vs Residual PD

**Standard PD** (`make_pd_action_config`, all runs before `hhi_20946_neutral_rpd`):

```python
# build_pd_action_offset_scale() computes from joint limits:
pd_action_offset = 0.5 * (lim_high + lim_low)   # joint midpoint ("neutral")
pd_action_scale  = 0.5 * (lim_high - lim_low)   # × action_scale

# At inference:
q_target = pd_action_offset + pd_action_scale * tanh(action)
```

For a crawling pose, `q_ref` is far from `pd_action_offset`. The policy must output a large sustained `action` every step just to fight the offset — hence jerk and instability.

**Residual PD** (`make_residual_pd_action_config`, `mlp_residual_pd.py`):

```python
# pd_action_scale = uniform 0.3 rad across all DOFs
# pd_action_offset = REPLACED at runtime by context.mimic.ref_state.dof_pos

# At inference (env.py:_process_action):
if use_residual_pd:
    params["pd_action_offset"] = context.mimic.ref_state.dof_pos   # q_ref(t)

q_target = q_ref(t) + 0.3 * tanh(action)
```

When `action = 0`, the joint target is exactly the reference pose. The policy only needs to output small corrections for dynamics, contacts, and physics error. This structurally eliminates large sustained outputs for floor-contact poses.

**The `pd_action_offset` in the config dict is a placeholder `zeros` tensor** — it gets replaced by `context.mimic.ref_state.dof_pos` every step via `_process_action()` (`env.py:516-518`).

---

### 4. Reward Components (`mlp.py`)

| Component | Signal | Notes |
|---|---|---|
| `gt_rew` | Global translation matching | Position of all 24 SMPL bodies |
| `gr_rew` | Global rotation matching | Orientation of all 24 bodies (biggest gap in Stage 1) |
| `gv_rew` | Global velocity matching | Body velocities — well converged |
| `gav_rew` | Global angular velocity matching | Body angular velocities — converged |
| `rh_rew` | Root height matching | Pelvis height |
| `contact_match_rew` | Contact flag matching | Foot/hand contact alignment with reference |
| `pow_rew` | Power penalty | Energy efficiency (reduced 1134 → 196 in Stage 1) |
| `action_smoothness` | `‖a_t - a_{t-1}‖₂ × -0.02` | Jerk suppression |

Reward normalization (`normalize_rewards=True`) uses a running mean/std over discounted returns. Values are stored un-normalized in the buffer; normalization is applied before advantage computation.

---

### 5. PPO Update — What Happens in `optimize_model()`

**Step A: `pre_process_dataset()` — GAE advantages**

```python
# discount_values() implements Generalized Advantage Estimation (Schulman 2016)
delta[t] = reward[t] + gamma * V(s[t+1]) * (1 - done[t]) - V(s[t])
advantages[t] = delta[t] + (gamma * tau) * delta[t+1] + ...

returns = advantages + values
```

Parameters: `gamma=0.99`, `tau=0.95` (GAE λ — controls bias/variance trade-off).

Advantages are EMA-normalized (`alpha=0.05`, clamp ±4σ) before the policy update to stabilize gradients.

**Step B: multiple mini-epochs over minibatches**

```
for mini_epoch in range(num_mini_epochs):
    for batch in shuffle(experience_buffer):
        actor_step(batch)   # PPO clipped surrogate loss
        critic_step(batch)  # value MSE loss
```

**Actor loss (PPO clipped surrogate, `agent.py:actor_step`):**

```python
ratio = exp(old_neglogp - current_neglogp)   # importance weight
surr1 = advantages * ratio
surr2 = advantages * clamp(ratio, 1 - e_clip, 1 + e_clip)
ppo_loss = max(-surr1, -surr2).mean()        # conservative: use the worse bound

actor_loss = ppo_loss + bounds_loss + extra_loss
# bounds_loss penalizes |mean_action| > 1 (keeps outputs in valid range)
```

`e_clip = 0.2`. If `clip_frac > 0.6` in a batch, the entire remaining mini-epoch skips actor updates — the policy has already moved too far from the rollout distribution.

**Critic loss (clipped value, `agent.py:critic_step`):**

```python
critic_loss_unclipped = (V_new - returns)²
V_clipped = V_old + clamp(V_new - V_old, -e_clip, e_clip)
critic_loss_clipped = (V_clipped - returns)²
critic_loss = max(critic_loss_unclipped, critic_loss_clipped).mean()
```

**Learning rates:** Actor `2e-5`, Critic `1e-4` (defaults). Adaptive KL scheduling is available but disabled in current runs. Stage 2 transfer uses 10× lower: actor `2e-6`, critic `1e-5`.

---

### 6. Two-Stage Curriculum — Current Position

```
Stage 1: hhi_20946_neutral  (DONE, epoch ~20,400, 174 h)
  ├── 20,946 HumanML3D clips, betas=0, smpl_mor_neutral
  ├── 84.9% success rate, reward 0.845, plateau confirmed
  ├── Failure: ~1,834 hard clips (crawl/kneel/squat/backward) = 8.8%
  └── score_based.ckpt → uploaded to R2 as ckpt/20946_neutral.zip

Stage 1.5: hhi_20946_neutral_rpd  (IN PROGRESS on RunPod)
  ├── Fine-tune Stage 1 ckpt with residual PD, same neutral dataset
  ├── Action space: q_ref(t) + 0.3·tanh(a)  vs old: q_neutral + scale·tanh(a)
  ├── Goal: adapt policy to new action mode before multi-shape transfer
  └── Watch epoch-0: terminate_mean should stay ~0.001; if >0.3 reduce scale to 0.1

Stage 2: hhi_stage2_transfer  (PLANNED)
  ├── Start from Stage 1.5 checkpoint
  ├── Multi-shape dataset: humos_slurmrank.pt (1024 clips × 128 shapes)
  ├── --robot-name smpl_mor, LR 10× lower (actor 2e-6, critic 1e-5)
  └── Run reset_morphology_normalizer.py on Stage 1.5 ckpt first
```

**Why Stage 1.5 is the right bridge:** Switching directly to residual PD + multi-shape in Stage 2 would give the policy two simultaneous shocks — new action semantics AND new body shapes. Stage 1.5 on the familiar neutral dataset isolates the action-space change, letting the policy recalibrate before body-shape variation is introduced.

---

### 7. Key Design Decisions — Summary

| Decision | Choice | Reason |
|---|---|---|
| Architecture | `mlp.py` flat concat | FiLM fanout bottleneck (12,288 conditioner outputs); ShapeEmbed/physics matched baseline |
| Morphology input | Physics features (15-dim) over raw betas (11-dim) | Transfer inference: 5/8 vs 0/8 success — physics features causally meaningful |
| Evaluator strategy | `eval_one_shape_per_motion` | Covers every clip per cycle; `max_eval_motions=2000` missed 13% of clips |
| Curriculum propagation | All 128 shape variants updated when any shape fails a clip | Prevents cross-shape weight inconsistency |
| Action control | Residual PD (`residual_scale=0.3` rad) | Eliminates sustained large outputs for floor-contact poses; root cause of jerk |
| Two-stage curriculum | Stage 1 neutral → Stage 1.5 RPD → Stage 2 multi-shape | Separates motion learning from shape adaptation; clean E7 shape-generalisation test |

---

### 8. Why the Normalizer Reset is Critical Before Stage 2

#### What the normalizer does

`RunningMeanStd` (`protomotions/agents/utils/normalization.py`) maintains two buffers per observation dimension — `mean` and `var` — updated via Welford's algorithm across every rollout step seen during training. At every policy forward pass:

```python
normalized = (obs - mean) / sqrt(var + 1e-5)
# then clamped to [-5, 5]
```

This keeps all input dimensions in roughly the same scale so gradients are not dominated by large-magnitude dims.

#### What happens to the beta dims during Stage 1

In Stage 1, `smpl_mor_neutral` is used — all bodies have betas = 0. So for all 4096 envs × every step of all 174 training hours, the 10 beta dims of `morphology_obs` receive the **exact same constant value: 0.0**.

The running variance of a constant signal is by definition zero:

```
var[beta_1] ≈ 0.0
var[beta_2] ≈ 0.0
...
var[beta_10] ≈ 0.0
```

The sample count after Stage 1 is enormous: `4096 envs × 32 steps × 6 GPUs × ~20,400 epochs ≈ 1.6 × 10¹⁰`. Welford's algorithm gives this accumulated statistic enormous inertia.

#### The saturation problem at Stage 2 start

When Stage 2 starts and real betas (e.g., `beta_3 = 1.8`) flow in, the normalizer divides by the accumulated `sqrt(var + 1e-5)` for that dim:

```python
normalized_beta_3 = (1.8 - 0.0) / sqrt(0.0 + 1e-5)
                  = 1.8 / 0.00316
                  ≈ 569
```

Then the clamp fires:

```python
clamp(569, -5, 5) → 5.0   # always maxed out
```

Every beta value — positive or negative, large or small — gets clamped to ±5. The network sees a binary signal ("max" or "min") with no information about actual beta magnitudes. The policy cannot learn any shape-conditioned behavior.

#### Why the running stats won't fix themselves naturally

Welford's combination formula for adding a new batch of N samples to an existing count C:

```
new_var = (old_var × C + batch_var × N + delta² × C×N / (C+N)) / (C+N)
weight of new data = N / (C + N)
```

With `C = 1.6e10` and `N = 4096 × 32 = 131,072`:

```
weight of new data ≈ 131072 / 1.6e10 ≈ 8 × 10⁻⁶
```

Each Stage 2 epoch moves the beta variance by eight-millionths of the way toward the true value. Going from `var ≈ 0` to `var ≈ 1.0` would require roughly **125,000 Stage 2 epochs** — more than Stage 1 itself took.

#### The fix: `tools/reset_morphology_normalizer.py`

```python
mean[-11:] = 0.0    # reset morphology dims (gender + betas)
var[-11:]  = 1.0    # restore unit variance so normalization is identity
# count left untouched — locomotion dims keep their full momentum
```

After the reset, the beta dims start from a sensible prior (identity transform). Within a few hundred Stage 2 epochs the normalizer converges to true Stage 2 statistics. The first ~1003 locomotion dims are completely unaffected — their `count` and `var` are preserved, so the hard-won normalizer state from 174 hours of Stage 1 is not disturbed.

================================================================================

## 16. Implementation — Phase Variable φ and Contact Bodies Extension (2026-06-30)

### A2. Phase Variable φ

**Motivation:** Periodic and quasi-periodic motions (squat, kneel, walk) exhibit temporal aliasing — the policy sees the same proprioceptive state at two different points in the clip (e.g., descending into a squat and ascending out of it look identical in joint-space). Without a phase signal, the policy cannot disambiguate and may output contradictory actions. Adding φ = `motion_time / clip_length ∈ [0, 1]` as a 1-dim obs resolves this.

**Files changed:**

| File | Change |
|---|---|
| `protomotions/envs/context_views.py` | Added `motion_phase: Tensor = FieldPath()` descriptor to `MimicContext`; added `motion_phase` param to `__init__` |
| `protomotions/envs/control/mimic_control.py` | Computed `motion_phase` in `populate_context` after `motion_lengths` is already available; passed to `MimicContext` |
| `protomotions/envs/obs/humanoid.py` | Added `compute_motion_phase_obs(motion_phase)` pass-through |
| `protomotions/envs/component_factories.py` | Added `motion_phase_obs_factory()` binding `EnvContext.mimic.motion_phase`; exported in `__all__` |
| `examples/experiments/mimic/mlp_residual_pd.py` | Imported factory; added `"motion_phase_obs"` to `observation_components` and to all four `in_keys` lists (actor, actor mu_model, critic, model) |

**Key implementation detail — computation in `mimic_control.py`:**

```python
motion_lengths = self.env.motion_lib.get_motion_length(motion_ids)
future_times = torch.minimum(future_times, motion_lengths.unsqueeze(-1))

# Phase: clamp denominator to avoid div-by-zero on zero-length clips
motion_phase = (motion_times / motion_lengths.clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(-1)  # [num_envs, 1]
```

`motion_lengths` was already computed on that line for future-time clamping — the phase adds no extra motion lib query. The `unsqueeze(-1)` gives shape `[num_envs, 1]` so it concatenates cleanly with the other obs groups. `LazyLinear` absorbs the extra dim on the first forward pass with no architecture change.

---

### A3. Contact Bodies Extension

**Motivation:** `contact_match_rew` penalises mismatches between the policy's contact flags and the reference motion's contact flags. With only feet in `contact_bodies`, the reward has no signal for floor-contact poses — the policy gets zero guidance toward putting knees on the floor during a kneel or hands on the floor during a crawl. Adding knees and wrists to `contact_bodies` gives direct reward signal for those contact events.

**Design decisions:**

- Changes are confined to `mlp_residual_pd.py` — not baked into the robot config — so standard walking experiments are unaffected.
- `non_termination_contact_bodies` must also be extended; otherwise the termination checker fires the moment a knee or wrist touches the ground, which would immediately reset the env and prevent the policy from ever learning these poses.
- `non_termination_contact_bodies` is set directly (not via `update_fields`) because `update_fields` only reprocesses `contact_bodies` through `abstract_names_to_body_names`. The literal SMPL body names work as-is.
- The cached property `non_termination_contact_body_ids` in `env.py` is computed lazily after env creation, so setting `non_termination_contact_bodies` in `configure_robot_and_simulator` (which runs before env creation) is safe.

**File changed — `examples/experiments/mimic/mlp_residual_pd.py`:**

```python
def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    robot_cfg.update_fields(
        contact_bodies=[
            "all_left_foot_bodies",   # L_Ankle, L_Toe
            "all_right_foot_bodies",  # R_Ankle, R_Toe
            "L_Knee", "R_Knee",       # kneel / squat floor contact
            "L_Wrist", "R_Wrist",     # crawl floor contact
        ]
    )
    robot_cfg.non_termination_contact_bodies = [
        "R_Ankle", "L_Ankle", "R_Toe", "L_Toe",
        "L_Knee", "R_Knee", "L_Wrist", "R_Wrist",
    ]
```

**Effect on reward:** `contact_match_rew` now fires on 8 bodies instead of 4. For a crawl clip where the reference has both wrists and both ankles contacting the floor, the reward will be non-zero as soon as the policy achieves any of those contacts — providing a gradient toward the correct configuration from the very start of an episode.
