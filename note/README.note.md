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


================================================================================

Second transfer residual pd attempt

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

At the very first eval point (right after load, before any real training):
- With action=0 → q_target=q_ref exactly, the character starts by directly tracking the reference pose with the pretrained
PD gains — no learned dynamic corrections yet, but also no wrong/overshot targets.
- Expect eval/gt_error/mean and eval/success_rate to open in a moderate, non-collapsed range — clearly worse than the mlp.py
baseline's 0.84 success / 0.195 error (since it hasn't learned the contact/balance corrections residual PD exists to add),
but nowhere near the 0.001 success / 1.29 error catastrophic floor from the checkpoint-mismatch run.
- eval/action_delta_mean_deg should start very low (actions near zero → very smooth), rising gradually as the policy learns
real corrections — unlike before, where it was already climbing from step 1 because the actor was fighting its own bad
starting point.
- info/episode_reward should open noticeably above the ~12 we saw last time, likely closer to the mid-to-high range, since
action_smoothness reward is near-max and tracking is roughly plausible.


Over training:
  - Because the policy only needs to learn small corrections (not relearn the whole task), convergence should be much faster
  than the ~5,200 epochs it took last time just to claw back to 0.64. I'd expect it to pass the mlp.py baseline's 0.84 success
  rate well before that, if the approach is working as intended — that's the actual point of residual PD.
  - One thing not fixed by our change: the critic wasn't reset, so its value estimates are still calibrated to the standard-PD
  reward/dynamics landscape. Expect possibly noisier losses/critic_loss, adv_norm/*, or actor/clip_frac in the first handful
  of epochs while it recalibrates — this is minor and self-correcting, not a red flag on its own.

  Red flag to check immediately after launch: pull up the very first eval/success_rate / eval/gt_error/mean point in wandb. If
  it's back near 0.001 / 1.29 (i.e., looks like the collapsed run), the fix didn't actually engage — worth double-checking
  the run picked warm_start mode (log line should say WARM START: Using checkpoint for initialization: ...) rather than
  silently resuming.

It was failed, the results was the same as residual pd transfer, no improvement

================================================================================

## 17. Failed-Motion Analysis — `hhi_20946_neutral` (2026-07-02)

**Context:** After the residual-PD transfer (`hhi_20946_neutral_rpd`) failed, examined the base
`hhi_20946_neutral` checkpoint it was fine-tuned from more closely to determine which motions
persistently fail and whether they are the physically difficult ones.

### [Method]

612 failure logs at `results/hhi_20946_neutral/failed_motions/` (epochs 1000–20400, 6 ranks).
Single neutral body shape (β=0) — no beta multiplication, so `motion_id` indexes directly into
each rank's 3,491-motion shard (`/home/hlz/datasets/humos_proto_neutral/offset/humanml3d_neutral_20946_000{0-5}.pt`).
Analyzed the 18 most recent epochs (≥17000), joined against
`/home/hlz/repos/hhi/data-processing/motion_id_text.json` for clip descriptions.

Ranked output: `results/hhi_20946_neutral/persistent_failures.txt` (6,772 ranked clips)
Full note: `note/README.failed-motions-20946-neutral.md`

### [Results] Overall Failure Scale

`eval/success_rate` plateaus ≈82–85% at late epochs (consistent with §12's 84.9% peak).
- **1,818 / 20,946 clips (8.7%)** fail in **all** 18 analyzed epochs
- **3,502 clips (16.7%)** fail in ≥50% of analyzed epochs

### [Results] Category Breakdown (top 100 worst clips, all 18/18 epochs failed)

| Category | Count |
|---|---|
| single-leg balance / kick / leg-swing / stretch | 40 |
| crawl / all-fours | 20 |
| balance on object / beam / climb | 14 |
| squat / crouch | 8 |
| sit | 7 |
| backward motion | 6 |
| kneel | 5 |
| lie down / get up / push-up | 3 |
| unclassified (fast walk/run, crab-walk outliers) | ~7 |

### [Analysis] Are the failures the physically difficult ones? — Yes

~93% of the worst clips are physically hard motions. Crawl/kneel/squat/sit/backward — the same
classes found in the earlier `hhi_1024_motion` pilot (§1, `note/README.failed-motions.md`) — are
still severe. But a **new dominant category** emerges at full 20,946-clip scale that was barely
present in the smaller 1024-clip pilot subset: **single-leg dynamic balance** (standing on one
foot, leg kicks/swings/circles, knee-to-chest) — 40% of worst clips.

Root cause interpretation: both failure families share a narrow/unstable base-of-support
problem — COM-drop clips (crawl/kneel/squat) vs. narrowed-support-polygon clips (single-leg).

### [Relevance] Connection to the `hhi_20946_neutral_rpd` Transfer Failure

The RPD transfer was fine-tuned from this checkpoint at epoch 20000, when 8.7–16.7% of clips
were already persistently failing, concentrated in exactly the categories residual PD targets
(crawl/kneel/squat — see §13/§16 A3 contact-body extension) plus the newly-found
single-leg-balance class residual PD was not designed for. Residual PD's `q_ref(t) + scale·tanh(action)`
fixes the "policy must fight a distant PD offset" problem for floor-contact poses, but does
nothing structurally for single-leg balance, where the difficulty is dynamic COM control over a
narrow support polygon rather than distance from the PD neutral pose.

**Next step (not yet done):** re-run the same failure analysis against
`results/hhi_20946_neutral_rpd/failed_motions/` for a direct before/after comparison against
this baseline.

================================================================================

## 18. Architecture Decision — GMT-Style MoE (motion axis only)

**Context:** Literature survey (GMT/PMCP/FARM/HyperDistill — full detail in memory
`architecture_research.md`) led to training **from scratch**, no checkpoint reuse — this retires
FARM (frozen-base advantage moot with nothing to protect) and PMCP (train→mine-failures→freeze→
retrain→compose pipeline too much operational complexity for a first attempt). **GMT-style MoE**
is the motion-axis structure. HyperDistill (body-shape axis) was designed and implemented, then
reverted for scope (2026-07-03) — **no code remains in this repo**; full design preserved in
memory (`architecture_research.md`) if revisited later.

### [Decision] Motion axis — GMT-style MoE + load-balancing loss

K parallel expert trunks + a gate, blended `a = Σᵢ pᵢaᵢ`, trained jointly in one PPO pass. Added a
Switch-Transformer/GShard-style load-balancing loss to fix GMT's RL-specific collapse risk (an
unconstrained gate can collapse onto 1-2 experts under noisy RL gradients):
```
f_i = fraction of minibatch routed to expert i (argmax of gate)
P_i = mean gate probability assigned to expert i over the minibatch
L_lb = K · Σᵢ f_i · P_i                         # minimized → uniform utilization
L_actor_total = L_ppo_clip + λ_lb · L_lb        # λ_lb ≈ 0.01
```
This only prevents degenerate collapse — it doesn't guarantee experts specialize along the axis
we care about (crawl/kneel/squat/single-leg-balance vs. everything else, §17); check post-hoc via
`expert_selection` against the known failure taxonomy.

**Why this should work — interference, not capacity:** `hhi_1024_motion` (1024 clips × 128
shapes = 131k instances, monolithic 6×1024 trunk) converged to reward ≈0.84; `hhi_20946_neutral`
(same trunk, 20,946 clips, shape count *reduced* to 1) plateaus at 82-85% with 8.7% persistent
failures. Shape diversity went *down* in that comparison, so the degradation isolates to
motion-count scaling, not capacity — gradient interference between incompatible motion types
(majority "easy" motions dominate shared-weight updates), not insufficient parameters. Separate
weights per behavior cluster is MoE's actual justification.

**K=8**, chosen without the scaling-curve diagnostic (proposed to calibrate K empirically by
training unmodified `mlp.py` on clip-count subsets — skipped for time). Only two data points
exist — 1024 clips works, 20,946 clips fails, nothing in between — so `20946/1024≈21` is a
conservative upper bound (1024 is a confirmed floor, not a measured ceiling), not a calibrated
target. K=8 is a middle-point bet: enough to meaningfully stress-test the load-balancing loss
beyond a trivial K=2 (easy to hit a 50/50 split without the loss doing real work), without paying
~21× per-step actor compute (`moe_mlp.py` evaluates all K experts every forward pass — compute
scales with K directly, not just parameter count). Driven by a new target run — 20,946 clips ×
{2,4} shapes, generating on `r2:proto-data/hhi_stage2/` (38/41 batches as of 2026-07-03) — which
sits in the same clip-count regime as the failing `hhi_20946_neutral` run, unlike the 1024-clip
pilot, so it's the real stress test of whether MoE helps.

### [Decision] Shape axis — flat concat (unchanged from baseline)

`morphology_obs` stays a plain concatenated input, same as `mlp.py`. Known not to be sufficient
on its own — `hhi_1024_motion` (flat-concat): 1028 persistent failures; `hhi_se_1024_motion`
(nonlinear-encode-then-concat): 1026, no real improvement — but fixing it is out of scope for
this round. HyperDistill design (mechanism, math, LazyLinear workaround) preserved in memory
(`architecture_research.md`) if this needs revisiting.

**Also out of scope:** Residual PD (confirmed orthogonal, `action_functions.py:416` — separate
ablation once the capacity story is validated), critic restructuring (stays flat-concat), PMCP
(fallback only if MoE fails to move the persistent-failure cluster).

### [Plan] Implementation — what's built (2026-07-03)

1. **`protomotions/agents/common/moe_mlp.py`** — `MoEMLPConfig` + `MoEMLP`. K expert trunks +
   gate, blended. `gate_mode: "learned"|"hard"` (hard mode routes via a precomputed per-env
   assignment instead of a gate net — all K experts still evaluated either way; skipping
   unselected experts' compute in hard mode is a possible follow-up, not implemented). Writes
   `gate_probs`/`expert_selection`. Config lives alongside the module (matches
   `film_mlp.py`/`shape_embed_mlp.py` precedent, not centralized in `agents/common/config.py`).
2. **`protomotions/agents/ppo/config.py`** — `MoELoadBalanceConfig` (mirrors `L2C2Config`),
   `moe_load_balance` field on `PPOAgentConfig`.
3. **`protomotions/agents/ppo/agent.py`** — `calculate_extra_actor_loss` extended with the
   load-balancing term (same extension point already used for L2C2).
4. **`examples/experiments/mimic/mlp_moe.py`** — `mlp.py` + `MoEMLPConfig(num_experts=8,
   expert_layers=[1024]×6)`, `moe_load_balance.enabled=True`. Morphology flat-concat, critic
   unchanged, standard PD, trained from scratch.
5. **`examples/experiments/mimic/mlp_wide.py`** — capacity-matched ablation control, see below.
   (Renamed from `mlp_moe_wide.py`, 2026-07-03 — it has no MoE structure at all, plain
   `MLPWithConcatConfig` same as `mlp.py` just wider, and the old name actively implied
   otherwise. Repo convention names files after their own mechanism, not what they're a
   control for — that relationship belongs in the docstring/note, not the filename.)

Verified on CPU (dummy TensorDict): forward/backward through experts and gate, both `gate_mode`
variants, load-balancing loss numerics, full `PPOActor` integration at the real `num_envs=4096`
batch size. Not yet run on GPU/IsaacGym.

### [Ablation] Capacity vs. routing

K=8 experts have ~8× the baseline trunk's parameter count in the part of the network that
matters — so a plain MoE-vs-baseline comparison can't tell "the routing structure helped" apart
from "the network just got bigger." `mlp_wide.py` is the control: same extra parameter
budget, poured into **one** trunk instead of eight (`layers=[2896]×6`, `w=1024·√8`, no gate, no
load-balancing loss, no dependency on `moe_mlp.py`). Verified empirically, not just
analytically: MoE (K=8) expert-stack params = 45,326,520; widened trunk = 43,130,151 —
**95% match** (gap is the gate's own 166K params). Critic identical in both files (3,544,065
params either way) — MoE total 49,036,993 vs. wide total 46,674,216, a 5.1% gap overall.

| Run | Config | Isolates |
|---|---|---|
| `hhi_moe_1024_motion` | `mlp_moe.py`, K=8 | MoE, the actual proposal |
| `hhi_wide_1024_motion` | `mlp_wide.py`, width 2896, no MoE | raw capacity, no routing |

Baselines on record (same 1024-motion subset): `hhi_1024_motion` (flat-concat) — 1028 persistent
failures; `hhi_se_1024_motion` (shape-embed) — 1026. Reading order: widened-trunk vs. baselines
tests whether capacity alone beats flat-concat; MoE vs. widened-trunk tests whether routing adds
anything beyond matched capacity — that's the actual question. Check the
**crawl/kneel/squat/backward category** specifically (`note/README.failed-motions.md`), not just
the aggregate count. Same pair should also run on the Stage 1 v2 data (20,946 clips × 2 shapes,
§19) once it lands — that run, not the 1024-motion pilot, sits in the actual failure regime this
design targets.

================================================================================

## 19. Stage 1 v2 — Filtering 20,946×128 Down to 20,946×2 (2026-07-03)

**Why:** `hhi_20946_neutral` (the original "Stage 1," single neutral shape) didn't meet
expectations (§17 — 8.7% persistent failures). `r2:proto-data/hhi_stage2/` (20,946×128, full
Stage 2 data) is generating now. Rather than wait for the full 128-shape set to test the MoE
work from §18, filter it down to 2 shapes now — **this becomes the new Stage 1**, reusing the
name deliberately since the old one is being superseded, not run alongside it.

**Script: `tools/extract_stage1_shapes.py` (2026-07-03, done).** Same schema/extraction pattern
as `tools/extract_gravity_core_clips.py` (`FRAME_KEYS`/`PER_MOTION_TENSOR_KEYS`/
`PER_MOTION_TUPLE_KEYS` are identical), but filtering by `motion_asset_ids` (shape) instead of
`motion_clip_ids` — keep all 20,946 clips, keep only `--asset-ids` (default
`male_71fbbe41 female_71fbbe41`) per shard. Adds R2 I/O the reference script doesn't have:
per-shard `rclone copy` down from `--r2-source` (default `r2:proto-data/hhi_stage2/`) → filter in
memory → save → `rclone copy` up to `--r2-dest` (default `r2:proto-data/hhi_stage1/`) → delete
local copy (same bounded-disk pattern as `prepare_stage2_data.py`, ~3.4GB resident at a time),
with the same `{workspace}/filter_log.txt` resume convention. Sequential, no shard-processing
parallelism (bandwidth-bound anyway, matches `prepare_stage2_data.py`'s own approach — flagged in
the script docstring as a place to speed up later if needed, not implemented now).

**Verified against a real Stage 2 shard** (`/media/hlz/R/stage2_data/batch_0000_0000_offset.pt`,
64 clips × 128 shapes = 8192 motions): filtered to exactly 128 motions (64 clips × 2 shapes, as
expected), frame data spot-checked identical after re-slicing (`gts` tensor byte-for-byte match
against the original for a sampled motion), `length_starts` recomputed consistently,
`motion_weights` reset to fresh 1.0. Not yet run against R2 (needs the remote server).

Output stays shard-per-shard; consolidation into a slurmrank pointer is a separate step reusing
`tools/merge_motion_shards.py`, not built into this script. **Bandwidth note:** download cost is
unavoidably the full ~1.1TB regardless of N —
each shard bundles all 128 shapes of its clips together, so getting all 20,946 clips means
pulling every shard either way. Savings are on storage/upload/everything downstream, not download.

**Shapes chosen: `male_71fbbe41` + `female_71fbbe41`** (same beta_key, opposite genders). Not
extremes — deliberately close to the population center (`physics_features.pt`, 128 shapes):

| | mass | height | mass percentile (within gender) |
|---|---|---|---|
| population median | 68.3 kg | 1.394 m | — |
| `male_71fbbe41` | 68.8 kg | 1.409 m | 39th |
| `female_71fbbe41` | 72.6 kg | 1.439 m | 64th |

Reasoning: the shape axis is explicitly out of scope for the current round (§18 — flat-concat,
no HyperDistill). Picking extreme bodies (e.g. lightest/heaviest, 26–144kg range) would introduce
a confound unrelated to what this run is testing — an extreme body may be inherently harder to
control with PD gains tuned for average bodies, making any failure ambiguous between "motion-count
interference" (the actual question) and "this particular body is hard." Same-beta-key,
opposite-gender keeps gender as the one clearly isolated shape variable while staying physically
close to the population `hhi_1024_motion`/`hhi_20946_neutral` were already evaluated against.


nohup python -u tools/extract_stage1_shapes.py \
      --r2-source r2:proto-data/hhi_stage2/ \
      --r2-dest r2:proto-data/hhi_stage1/ \
      --workspace /workspace/stage1_prep > /tmp/stage1data.log 2>&1 &

================================================================================

## 20. Key-Joint Tracking Idea — Per-Joint Error Breakdown (2026-07-05)

**Theoretical foundation (verified against source papers, 2026-07-06):**

- **DeepMimic** (Peng et al., SIGGRAPH/TOG 2018) — original precedent for splitting the imitation
  reward into separate weighted terms instead of one blended pose error: pose (0.5), velocity
  (0.05), **end-effector (0.15)**, root (0.1), center-of-mass (0.2). Establishes that end-effector
  position error deserves its own dedicated, separately-weighted reward term — the root idea behind
  "key-joint" reweighting.
- **H2O** (He et al., IROS 2024, arXiv:2403.04436) — tracks a sparse set of **8 key bodies**
  (shoulders, elbows, hands, ankles) as the reward/observation target instead of the full ~20+
  body set. Direct precedent for "a small key-joint set is enough to define the task." Verified
  detail: H2O applies **uniform** weighting across all 8 key bodies — it does not itself argue for
  weighting wrists over ankles.
- **ExBody** (Cheng et al., 2024, arXiv:2402.16796) — the actual source of an upper-body-priority
  split: upper body directly imitates reference pose/keypoints, while the legs are "relaxed" to
  track a velocity command instead of copying joint angles, justified by "the mechanical
  limitations and stability requirements" of a real bipedal robot. This assumption — legs handled
  by a separate stability-oriented control path rather than dense pose tracking — is exactly why
  it doesn't transfer to our setup: we have no separate balance controller, everything is dense
  pose-tracking in sim, and legs are the joints failing *hardest*, not the ones that need relaxing.
- **ExBody2** (2024, arXiv:2412.13196) — reports upper-body vs. lower-body tracking error (MPJPE)
  as separate metrics rather than one blended number. Direct precedent for this section's own
  grouped per-body-region error table methodology.
- **OmniH2O** (He et al., CoRL 2024, arXiv:2406.08858) — same lineage as H2O; adds
  teacher-student distillation and stability-shaping regularization rewards (feet-height/air-time
  curricula). Adjacent context, not additional evidence on per-joint weighting.
- **Classical grounding**: operational-space / task-priority control (Khatib, 1987) — the
  control-theoretic argument that end-effector task-space error is the quantity to prioritize,
  with proximal/redundant joints treated as lower-priority null-space DOFs. DeepMimic's and H2O's
  key-body reward terms are RL-flavored descendants of this idea.

**Our own experiment support:** ran a per-body-joint error breakdown on the 60 worst
persistent-failure clips of `hhi_moe_20946_2shape` (root-relative position error against
`last.ckpt`, epoch 5460) to check whether root/ankle error actually dominates over wrist error.
This 60-clip subset is still only 6.7% success (56/60 fail) at epoch 5460 — genuinely stuck, not
slow-converging. Grouped mean root-relative position error, worst to best:

| group | mean err (m) |
|---|---|
| ankles+toes | 0.214 |
| wrists+hands | 0.139 |
| knees | 0.117 |
| elbows | 0.117 |
| shoulders+thorax | 0.111 |
| spine/torso/neck/head | 0.093 |
| hips | 0.036 |

Ankle/toe error is ~54% larger than wrist/hand error, and both sit well above the rest — i.e. the
key-joint set proposed (root + wrists + ankles) is exactly the two worst-tracked extremity groups,
so the idea is well-supported: prioritizing those and loosening knees/elbows/shoulders/spine/hips
doesn't sacrifice anything that's currently working. This directly contradicts ExBody's
upper-body-priority design (see Theoretical foundation above — earlier draft of this note
misattributed that design to H2O): ExBody's leg-relaxation assumes a separate real-robot balance
controller, which doesn't apply here, so ankles should get equal-or-more weight than wrists, not
less. **Net: the literature supports "track a sparse key-body set" (H2O) and "measure/report error
by body region separately" (ExBody2), but no paper found actually argues wrists should outweigh
ankles — that asymmetry (ExBody) rests on a real-robot-balance-controller assumption that doesn't
hold here. The ankle-over-wrist ordering in this project is evidenced by our own epoch-5460 data,
not by any cited paper.**

**Why:** `hhi_moe_20946_2shape` (§18's MoE stress test) judged plateaued around epoch 5000+
(minimal gain over several hundred epochs), prompting the reward-reweighting idea above.

**Implementation: done (2026-07-06).** Steps 1-5 below were implemented and unit-verified (not
yet trained/launched — that's a separate step). One deviation from the plan as originally written:
step 5 said to edit `mlp_moe.py` directly; instead created a new sibling experiment file
`examples/experiments/mimic/mlp_moe_keyjoint.py` (copy of `mlp_moe.py` + the `gt_body_weights`
wiring) so the baseline config used by the still-referenced `hhi_moe_20946_2shape` run stays
untouched, matching this repo's existing pattern of one experiment file per isolated variant
(`mlp.py`, `mlp_wide.py`, `mlp_film.py`, `mlp_moe.py`, ...). Reward-body aggregation was
unweighted everywhere before this change — confirmed by reading the actual code, not assumed:

1. `protomotions/envs/rewards/base.py` — `mean_squared_error_exp()` (line 42-79) does
   `per_body = diff_sq.mean(dim=-1)` then plain `per_body.mean(dim=-1)` (line 71/73) — an
   unweighted mean over the body axis. This is the actual place per-body weighting has to be
   injected. Add an optional `body_weights: Optional[Tensor] = None` param; when given, replace
   the unweighted mean with a **normalized weighted mean**:
   `(per_body * body_weights).sum(-1) / body_weights.sum()` — normalizing by the weight sum keeps
   the reward on the same scale, so existing `coefficient` values (e.g. `gt_coef=-25.0` in this
   experiment) don't need retuning. `rotation_error_exp()` (line 82-113) has the identical pattern
   (`angle_diff_sq.mean(dim=-1)`, line 111) for a second-order test on rotation tracking — not
   needed for the first test (see scope below).
2. `protomotions/envs/rewards/tracking.py` — `compute_gt_rew()` (line 58-77) is the only kernel
   that needs to change for the first test: add `body_weights: Optional[Tensor] = None` and pass
   it into `mean_squared_error_exp(..., body_weights=body_weights)`.
3. `protomotions/envs/component_factories.py` — `gt_rew_factory()` (line 462-481) add
   `body_weights: Optional[Tensor] = None` and put it in `static_params` alongside `weight`/
   `coefficient` (the `MdpComponent`/`resolve_args` machinery in `mdp_component.py` already passes
   `static_params` straight through as kwargs and auto-moves any `Tensor` value to the right device
   — no framework changes needed, confirmed via `mdp_component.py:167-276`). Then
   `mimic_tracking_rewards_factory()` (line 572+) needs a pass-through `body_weights` param forwarded
   into `gt_rew_factory(...)`.
4. New small helper (put near the other factories in `component_factories.py`, or inline in the
   experiment file since it's a 5-line one-off): `build_key_body_weights(kinematic_info, key_bodies,
   key_weight, other_weight=1.0)` returning a `[num_bodies]` tensor. **Must** resolve indices via
   `kinematic_info.body_names.index(name)` at env-config-build time, not hardcoded indices/order —
   the 24-body order used in `diff_key_joint_errors.py`'s `BODY_NAMES` happens to match this robot's
   MJCF depth-first traversal order (smpl_mor), but a different robot (G1, H1_2) has a different
   body count/order, so hardcoding would silently misindex on any other robot config.
5. Wired up in `examples/experiments/mimic/mlp_moe_keyjoint.py`'s `env_config()`:
   `mimic_tracking_rewards_factory(..., gt_body_weights=build_key_body_weights(robot_cfg.kinematic_info,
   key_bodies=["Pelvis","L_Wrist","R_Wrist","L_Ankle","R_Ankle"], key_weight=3.0))`. `key_weight=3.0`
   is an arbitrary first guess, not derived from theory or data — a hyperparameter to sweep once the
   first test result is in.
6. **Must be a new run, not a resume**: per this repo's resume semantics, `resolved_configs.pt` is
   loaded directly and the experiment file is *not* re-executed on resume, and `body_weights` is a
   Tensor (can't be expressed via scalar `--overrides`) — so this has to launch as a fresh
   `--experiment-name` from the edited `mlp_moe.py`, not a resume of `hhi_moe_20946_2shape`.
7. **First-test scope** (keep it a single isolated change, per earlier agreement to test one thing
   at a time): only `gt_rew` (position tracking); leave `gr_rew`/`gv_rew`/`gav_rew`/`rh_rew`
   untouched, since the per-joint diagnostic above only measured position error. `key_bodies` =
   root + 2 wrists + 2 ankles (5 of 24 bodies) — the originally-proposed set, not the broader
   toe/hand groupings used only for the analysis table above.
8. **Pre-existing dead code found during this investigation, do not reuse for this**:
   `protomotions/envs/base_env/utils.py:combine_rewards()` already accepts a `region_weights`
   param (docstring: "per-body weights based on anatomical regions") and `protomotions/envs/
   base_env/env.py` already computes `self._density_weights` via `compute_body_density_weights()`
   (`protomotions/components/pose_lib.py:204`) and passes it in — but `combine_rewards()` never
   actually applies `region_weights` to anything; it's a fully unused parameter. Don't repurpose it
   for key-joint weighting — its intended semantics (down-weighting anatomically dense regions like
   finger chains) is a different concept from "make root/wrists/ankles matter more." Separate,
   pre-existing dead-code cleanup, unrelated to this task.

**Diagnostic method used above (historical, already run):** built a 60-clip probe motion file from
the worst persistent failures (recent-window, epochs 2600-3600), ran
`inference_agent.py --full-eval` on `last.ckpt` (epoch 5460) to dump
`predicted_motion_lib_epoch_5460.pt`, then diffed per-body position error against the probe ground
truth (`results/hhi_moe_20946_2shape/diff_key_joint_errors.py`). Two gotchas hit along the way: (1)
naive diffing gave nonsense ~180m uniform error because the saved rollout carries each parallel
env's world-grid placement offset — fixed by recentering both trajectories to root-relative
position before diffing; (2) `MimicEvaluator._update_motion_sampling_weights` assumes every shape
has an equal-size clip list and crashes on this unbalanced 29/31-shape probe subset — worked around
with a throwaway no-op monkeypatch for this run only, not a repo change. Ran locally (laptop RTX
4060, 8GB) — IsaacGym works locally for small one-off `--full-eval` checks like this
(`num_envs<=8`, `simulator.sim.physx.default_buffer_size_multiplier=1.0` to avoid OOM), no RunPod
round-trip needed.

**Caveat:** measured on the worst-60 subset, which skews toward crawl/kneel/squat clips by
construction (§17/[[failed_motions_20946_neutral]]), so the gap may be smaller on the full dataset.

**Status:** code is implemented and unit-verified (see Implementation above). **Launching the
actual `hhi_moe_20946_2shape_keyjoint` training run is a separate step, not yet started —
deferred until next session.**

================================================================================

## 21. Future Idea (parked) — Self-Triggered Progressive Expert Growth (2026-07-06)

MoE (§18/[[architecture_research]]) validated the K=8 bet (~8x faster to matched success-rate vs
the monolithic baseline) but still plateaus on the same persistent-failure cluster. [[architecture_research]]
already earmarked PMCP (progressive primitives, mine-failures → freeze → add-primitive → compose)
as the fallback for exactly this trigger — but PMCP's manual staged-run pipeline is cumbersome.

**Idea:** automate PMCP's staging inside one continuous run — detect when an expert/data-subset has
"saturated" (smoothed per-motion success rate plateaus) and only then grow/reallocate capacity
(bias gate toward a fresh or under-used expert for the remaining unsaturated motions), instead of
externally mining failures and launching a new job per primitive.

Open design questions: what saturation signal triggers growth (some variant of the existing
`motion_weights_update_success_discount` machinery, but as a trigger not just a sampling reweight);
how to bias routing so a newly-grown expert actually receives gradient (PMCP solves this by
weight-initializing the new primitive from the previous one, not randomly); freeze vs. keep-training
old experts (forgetting risk vs. losing GMT's joint-training benefit).

**Not started** — no validated precedent found yet for this exact combination (auto-triggered
capacity growth + MoE gate + physics-based motion imitation RL); closest analogs are PMCP itself
(this exact domain, but externally-staged) and continual-learning capacity-expansion work (e.g.
dynamically-expandable-networks-style triggers, but for sequential task boundaries, not a shared
RL router). **Needs a literature pass before design — parked, revisit later.**

### [Literature] Pass Complete (2026-07-07)

**PMCP (Luo et al., ICCV 2023, arXiv:2305.06456) directly answers all three open design
questions** — verified against the paper's Section 3.2 / Appendix B.3 / Algorithm 1, not just
re-cited from memory:
- **Trigger:** automatic, performance-based — convergence defined as "success rate on the current
  hard subset no longer increases," then the primitive is evaluated on the full motion set and
  failures form the next hard subset. Structurally identical to the plateau-on-a-hard-subset idea
  above, and this project's existing infra (`motion_weights_update_success_discount` EMA,
  `failed_motions/` dumps, §17's persistence-counting method) is already most of the way to
  measuring this signal.
- **Init:** confirmed — PMCP tested both random init (PNN-style) and warm-starting the new
  primitive from the previous primitive's weights, and kept the warm-start version.
- **Freeze:** confirmed — the old primitive is fully frozen before the new one is created. Zero
  forgetting, at the cost of GMT's joint-training benefit for old experts.

**Structural mismatch (why this isn't just "go implement PMCP"):** PMCP's composer/gate trains
once, at the end, over fully-assembled frozen primitives — sequential frozen experts stitched
together after the fact, not a live, jointly-trained gate that grows in place the way GMT's MoE
gate does here. Also: PMCP's own GitHub README (not the paper) states training is **not
automated** — a human runs `forward_pmcp.py` to mine hard sequences and manually restarts training
per primitive. This confirms the "cumbersome" framing above is accurate to the real
implementation, not just the paper's clean Algorithm-1 presentation — automating this inside one
continuous run is a genuine, unaddressed gap.

**DynMoE (ICLR 2025, arXiv:2405.14297)** — closest MoE-native precedent for growing expert count
*during* training. Its trigger is **coverage-based**, not performance-based: a token that
activates zero experts (all gate scores below a learned threshold) triggers adding a new expert;
an expert activated by no tokens gets removed. No analog to this in the current codebase (no
per-token routing failure signal exists). Init/freeze policy could not be confirmed (PDF
extraction failed repeatedly) — flagged as unconfirmed, not guessed. Supervised/vision/LM only, no
RL discussion.

**Progressive Neural Networks (Rusu et al. 2016) and Self-Controlled Dynamic Expansion Model
(2025, arXiv:2504.10561)** — both **require externally-supplied discrete task boundaries/IDs**
(SCDEM explicitly instantiates a new expert "for a given task," no drift/saturation detection).
Negative evidence supporting this project's framing: continuous single-curriculum RL with no task
IDs doesn't fit the classical continual-learning template, which is exactly why PMCP's
subset-eval-based trigger (no task ID needed) fits better.

**"Mixture of Experts in a Mixture of RL Settings" (arXiv:2406.18420)** — most on-topic MoE+RL
paper found: studies fixed-K MoE inside actor-critic DRL under non-stationarity (helps with
dormant-neuron issues) but does **not** discuss or propose growing expert count. Its silence on
growth is itself weak evidence that this combination is unaddressed in the RL+MoE literature, not
just overlooked by this project's own search.

**Di-SkilL (arXiv:2403.06966)** — curriculum RL with per-expert context auto-adaptation, found but
not deeply verified (time-boxed); likely fixed-K, not growing K. Follow-up read, not asserted.

**Verdict: no direct precedent for the exact combination** (auto-triggered growth + live MoE gate
+ RL). Recombines pieces from different domains: PMCP has the right trigger/init/freeze answers
but the wrong architecture (sequential frozen + one-shot gate); DynMoE has the live-gate growth
mechanism but the wrong trigger (coverage, not performance) and unconfirmed init/freeze, in a
non-RL setting.

**Recommendation — worth pursuing, framed as porting PMCP's validated choices into GMT's live-gate
architecture, not as implementing an existing method:**
- **Trigger:** PMCP's success-rate-plateau-on-current-hard-subset signal, direct reuse — this
  project's per-motion success EMA and `failed_motions/` logs are already closer to this than to
  DynMoE's per-token coverage signal, which has no analog here.
- **Init:** warm-start the new expert from whichever existing expert is currently most responsible
  for routing the unsaturated motions (PMCP's confirmed choice, adapted from "one primitive" to
  "clone one of K experts"). **Load-bearing, not cosmetic**: under the existing load-balancing loss
  (§18), a freshly randomly-initialized expert competing against 8 already-converged experts would
  likely never get picked at all — clone-init is probably required for the new expert to receive
  any gradient.
- **Freeze:** PMCP's answer (freeze old experts) is the safer starting point, consistent with this
  project's own lesson that RL gradients are noisy and coupled moving parts are risky (§18's
  load-balancing loss exists for exactly this failure class). A partial-freeze (reduced LR instead
  of full freeze) is a plausible middle ground but is **extrapolation, not validated by any
  surveyed paper** — flagged explicitly, not presented as a finding.

Still parked — this is a design sketch, not an implementation plan. Revisit if/when the current
key-joint MoE run (§20) is evaluated and still plateaus on the same cluster.

(AMP/ASE discriminator was also discussed as an alternative — `MimicADD(AMP)` already exists in
`protomotions/agents/mimic/agent_add.py` and is unused by current mimic experiments, which run
plain PPO. Judged not intriguing enough to pursue now: current failures look like precision
failures, not the recovery/off-reference gap AMP is best evidenced for — not parked, just not
pursuing.)

================================================================================

## 22. Deep Research — Additional Solutions for the Persistent-Failure Cluster (2026-07-07)

Deep research pass (3 parallel literature angles) for solutions beyond everything already tried
(§16/§18/§20/§21). Of the findings, two are judged the real candidates — the rest were reward/
curriculum tweaks that optimize within the existing paradigm rather than questioning a premise of
it (full list preserved in memory `hard_motion_solutions_survey.md` if needed later).

**1. Physics-corrected reference distillation** (InterMimic arXiv:2502.20390, 2025; ReActor ACM
TOG July 2026; PhysDiff ICCV 2023 oral; PARC arXiv:2505.04002 — convergent finding across sources).
Imitating a physics-refined rollout instead of the raw kinematic reference: InterMimic reports
23.9%→90.7% (train) / 9.6%→95.5% (test) success on the same motions; ReActor gets 97.45% vs 95.51%
success / 4.22° vs 6.62° joint RMSE over retarget-then-freeze, explicitly fixing motions "otherwise
infeasible." Questions a premise none of our other attempts have: that the raw AMASS/HumanML3D
reference is itself a valid target. Supports the hypothesis that some of our ~1,818
persistent-failure clips are hard because the reference is borderline-infeasible for the tracked
body, not purely a policy-capacity problem — though no source quantifies this fraction for our
dataset (don't oversell as "our data is X% broken").
**Implementation sketch**: re-run the hard-clip fine-tune (already done once, `hhi_1024_motion_tune`),
dump its converged rollouts (reuse the `predicted_motion_lib_epoch_*.pt` pattern from the §20
key-joint diagnostic), splice them back into the motion library as the new reference for those
clips. Existing infra (`extract_failed_motions.py`, rollout-dump pattern) covers most of the
plumbing. **Cost: moderate**, data-pipeline work, no GPU needed until the validation run.

**2. Residual PD, retried differently.** Not a new idea — already failed twice (§13/§14, and the
RPD failure-log comparison in memory `failed_motions_20946_neutral.md` showed a collapse-at-
warm-start pattern both times) — but "What Makes Value Learning Efficient in Residual RL?" (Ma et
al., arXiv:2602.10539, 2026 — the paper's own proposed fix was extracted as paraphrase, not
verbatim; diagnosis treated as solid, prescription as directional) names the likely mechanism:
**critic miscalibration under action-space semantic shift** — the critic's value estimates,
calibrated to standard-PD dynamics, misguide the actor from step one of an abrupt switch to
residual actions. Both our attempts did exactly that abrupt switch with no critic reset. Standard
residual-RL mitigation: reset/pretrain the critic on the new action semantics first, and ramp
`residual_scale` gradually (e.g. 0→0.3 rad over epochs) instead of switching the whole action
config at epoch 0. **Cost: medium** — worth one more attempt with these two changes, kept in the
note for the citation/mechanism even though it's a retry, not a fresh axis.

================================================================================

## 23. Idea (parked, not researched) — Curve-Shape Matching Instead of Pointwise-in-Time Tracking (2026-07-07)

**The idea:** every tracking reward (`gt_rew`, `gr_rew`, `gv_rew`, `gav_rew`) currently compares
policy state at time `t` to the reference at the *same* `t` — strict phase-locked correspondence.
Proposal: reward matching the *shape* of the joint trajectory (curve) rather than the exact
value-at-exact-time — the motion doesn't need to be identical, just similarly-shaped, allowing
local timing slack.

**Why this matters more here than in fixed-body papers (PHC/GMT/H2O/etc.):** how fast a body can
execute a movement is a function of its mass/moment of inertia. We retarget one captured
joint-angle-vs-time curve across 128 SMPL shapes spanning 26-144 kg. A heavier body decelerating
out of a squat needs more time to dissipate the same relative momentum than a lighter one, even
with an equally good policy. Forcing exact-time correspondence onto every body shape may demand a
target that's positionally reachable but not *on that schedule* for that particular mass — a
distinct infeasibility mechanism from §22's dynamics-infeasibility (target unreachable at all).
This is invisible to any fixed-body paper's evaluation, so no amount of citing their results
confirms or refutes it for us specifically.

**Related known techniques (general knowledge, not a fresh literature search — unlike §21/§22):**
- Dynamic Time Warping / Soft-DTW (Cuturi & Blondel, ICML 2017) — differentiable curve-shape loss
  tolerant of local time-warp.
- Motion Matching / phase-based animation (Holden et al.'s PFNN and successors) — game-animation
  characters driven by matching motion *features*/phase rather than exact frame correspondence.
- AMP/ASE — already unused in this repo (`MimicADD(AMP)`, `protomotions/agents/mimic/agent_add.py`)
  — a different mechanism for a related idea: a discriminator judges whether motion snippets look
  like the *distribution* of real motion, no exact-time correspondence to a specific reference at
  all. Previously deprioritized (§21) on grounds that current failures look like precision
  failures, not a recovery/off-reference gap — worth revisiting that call if this idea is pursued,
  since "precision failure" and "wrong governance of timing" aren't obviously the same claim.

**Implementation shapes considered (not built):**
1. Bounded local time-warp / windowed best-match — compare against whichever frame in a small
   window `[t-δ, t+δ]` best matches, instead of exactly `t`. Cheap, bounded (avoids unconstrained
   reward hacking). Natural extension of the existing `motion_phase` observation (§16 A2) from
   "observation only" to "also drives reward alignment."
2. Soft-DTW as a proper differentiable trajectory loss — more correct, more expensive, more new
   code in a vectorized IsaacGym reward pipeline.
3. Feature/shape-based terms (velocity direction, contact order) instead of raw time-aligned
   position/velocity — `gv_rew`/`gav_rew` are already a step toward "shape" over "exact pose" but
   remain exact-time.

**Key risk:** timing is not always dispensable — for genuinely fast/dynamic motions (a kick, a
fast transition) timing *is* the skill. Over-relaxing the warp window risks the policy learning a
shape-similar but qualitatively slower/lazier version of the motion — a reward-hacking mode
specific to this fix. The right warp-window bound is an empirical question, not obvious upfront.

**Relationship to §22:** complementary, not competing. Physics-corrected reference distillation
fixes "target unreachable at all, at any schedule." Curve-shape matching fixes "target reachable,
but not exactly on this schedule for this body." Both could matter for the same failure cluster,
for different frames — not mutually exclusive fixes.

**Status:** parked, not researched or implemented. No literature pass run yet on
"time-warp-tolerant imitation reward for morphology-varying bodies" specifically — revisit with a
proper search before building anything, per this project's own standard for distinguishing
verified findings from first-principles reasoning.

### [Refinement] Continuous Progress Variable Instead of a Fixed Window (2026-07-07)

Follow-up idea: instead of a bounded `[t-δ, t+δ]` window (implementation shape 1 above), let the
reference-alignment pointer be a **continuously-growing state variable** — "grow the curve from
the root" — rather than snapped to wall-clock time or reset each window. Monotonic, starts at the
clip's start, advances based on tracking quality rather than 1:1 with simulation time.

**Named precedent (general knowledge, not freshly searched):**
- **Path-following control** (vs. trajectory tracking) in robotics — a path parameter `s` (e.g.
  Lapierre & Soetanto, ship/AUV path-following) evolves by its own adaptive dynamics rather than
  being locked to wall-clock time; canonical distinction between "hit x_ref(t) at time t" and
  "follow the shape of x_ref(s), advance s as fast as tracking allows."
- **Online score-following** (Dannenberg 1984 onward; Arzt & Widmer) — live music performance
  tracked against a reference score via incremental/online DTW; score position is a state variable
  advanced by current match quality, monotonic, never resets backward.

**Why stronger than the fixed window for this project's actual motivation:** a fixed small window
only absorbs local jitter. A body that needs to be uniformly slower through an *entire* squat
(mass/inertia-driven, not incidental) will drift outside any reasonably small δ well before the
motion ends, since the mismatch accumulates over time. A continuously-growing progress variable
has no such ceiling — it can run persistently slower/faster for as long as needed, which is the
actual shape of the mass-driven timing problem, not just occasional desync.

**Candidate advance rules for `ref_progress` (replacing `motion_manager.py`'s rigid
`motion_times += env_dt`):**
1. **Reactive/error-gated** — advance at nominal rate when tracking error is low, slow or hold
   when error is high, resume once caught up. Pure feedback, no biomechanical assumptions.
2. **Principled/mass-scaled** — scale nominal advance rate by a per-body dynamic-similarity factor
   derived up front, not reactively. Connects directly to an already-flagged-but-unimplemented
   idea in §4's physics-features table: `T_step_natural = 2π√(l_leg/g)` (Froude-style natural
   timing formula). Only covers the mass-driven component, not general "policy currently behind."
   Could combine both: principled baseline pace, reactively adjusted.

**Critical risk — reward hacking via deliberate stalling.** If the policy can influence how fast
its own reference advances, it can learn to track badly on purpose to keep the reference stuck on
an easy early frame indefinitely, harvesting reward without ever finishing the clip — the same
pathology as the well-known CoastRunners boat-racing RL example (agent loops for points instead of
finishing the race, because the reward didn't actually require finishing). Any "reference waits
for you" mechanism needs a countervailing force from day one: a hard cap on total allowable lag
between `ref_progress` and wall-clock `t`, and/or a fixed episode time limit so stalling
accumulates strictly less total reward than progressing. Not an afterthought — has to be in the
initial design.

**Architectural cost, higher than the fixed-window version.** `motion_times` is currently literal
elapsed simulation time and is load-bearing well beyond the reward: termination
(`done_clip = (motion_times + dt) >= end_times`), future-target queries in `mimic_control.py`, the
evaluator's success-window bookkeeping, and the `motion_phase` observation (§16 A2) — which would
itself need to be redefined in terms of `ref_progress` rather than wall-clock time, or it
reintroduces the exact temporal-aliasing problem that feature was built to fix. A bigger lift than
implementation shape 1, touching multiple subsystems, not a drop-in reward change.

**Status:** parked, liked directionally, needs careful design before any implementation attempt —
explicitly flagged (by the user) as one to be very careful with, given the reward-hacking risk
above is not hypothetical-and-unlikely, it's a well-documented RL failure mode this exact
mechanism invites by construction.

### [Refinement 2] Reward Mechanism — Future Window as Observation, History Window as Reward (2026-07-07)

Continues the Refinement above, working out concretely how `ref_progress` (`s_t`: a
time-into-clip value, same unit as `motion_times`, monotonic, always `≤` wall-clock `t`) actually
feeds the reward, split into two separate jobs rather than one mechanism.

**Split:**
- **Future window `[s_t, s_t+w]` → observation only.** Not new machinery — `protomotions/envs/
  obs/target_poses.py`'s `build_max_coords_target_poses_future_rel()` already builds this (future
  reference frames, root-relative, `future_steps` param), currently anchored to `motion_times`.
  Only change needed: re-anchor to `s_t`.
- **History window `[t-k, t]` → reward.** No search: direct pairing `x_τ` vs `y(s_τ)` at each past
  `τ`, using whatever `s_τ` the (still-undecided) advance rule already committed to at that step.
  All time-warp tolerance comes from `s_τ` lagging `t`, not from any search/DTW inside the reward.

**Insufficient history (`t < k`, e.g. right after episode reset):** fixed-size `[num_envs, k]`
window always (needed for GPU batching across envs that reset at different times), zero-padded
when `t<k`. Pad slots must be masked out of any average — a raw 0-vs-0 pad slot reads as a fake
perfect match otherwise (seq2seq padding analogy, but padding alone isn't sufficient without the
mask, same as `ignore_index`/attention-masking in that setting). History buffer must also be
cleared on env reset (mid-episode termination+respawn), or the window would straddle two unrelated
attempts — check existing `HistoricalView` (state-history-buffer infra) for reset-clearing
precedent before building new.

**Two-regime comparison:**
1. `t < k`: plain masked pointwise position error (today's `exp(-error/σ)` via
   `mean_squared_error_exp`), unchanged.
2. `t ≥ k`: switch to genuine **curve-shape** comparison — masked pointwise error, even windowed,
   is still fundamentally a position-difference metric, not a shape metric.

**Shape metric chosen for the `t ≥ k` regime: Lin's Concordance Correlation Coefficient (CCC),
not bare Pearson correlation.**

`CCC = (2·ρ·σ_x·σ_y) / (σ_x² + σ_y² + (μ_x−μ_y)²)`

computed per joint/dimension channel over the window's real (unmasked) frames, then averaged
across channels. Chosen over bare correlation because correlation alone is blind to amplitude/
offset (a joint moving in perfect sync at half amplitude still scores `ρ=1`) — CCC folds the
amplitude/offset penalty into the denominator directly, no separate hand-rolled amplitude term
needed. Discrete Fréchet distance (the rigorous "dog on a leash" curve-distance formulation)
considered as the more-correct alternative; deprioritized as the cheap option to try first.

**Known gaps, not resolved this pass:**
- Per-channel zero-variance guard — a near-static joint gives `σ≈0`, CCC denominator breaks.
  Needs a clamp/skip.
- Minimum real-frame count for CCC to be meaningful (correlation on 1-2 points is noise) — may
  need its own threshold, possibly stricter than `t≥k` alone implies.
- Which joints feed this at all — separate open question, see base idea above (§23's key-joint
  tracking is a different purpose than this progress-gate/reward question; tentatively considered
  root+knee chain for the progress-gate role vs. root+wrists+ankles for the reward-weighting role
  in §20, not settled).
- The advance rule itself (what actually sets `s_τ` each step) — still undecided. Tentatively
  floated: reuse this same history CCC/error signal to gate how fast `s_t` advances, not committed.

**Status:** design in progress, not implemented. **Next open question, paused mid-discussion
(2026-07-07):** the advance rule (what formula actually sets `s_τ`/`s_t` each step, replacing
`motion_manager.py`'s fixed `motion_times += env_dt`) — discussion got confusing distinguishing
"the window `[t-k,t]`" (fixed-length, already settled) from "`s_τ`, a value that must be freshly
computed every step of the whole episode, not just inside the window" (the actual unsolved part),
and separately from "wall-clock sim time `t`" (stays fixed, untouched) vs. "which reference frame
to show" (currently wrongly hard-locked to `t` via `motion_times`, the thing `s_t` is meant to
unlock). Revisit with a clearer/simpler explanation before continuing — possibly a worked numeric
example across a full short episode rather than abstract formulas.

================================================================================

## 24. `mlp_moe_stable.py` — PPO Update-Stability Guard Rails (2026-07-09)

**Context:** `hhi_moe_20946_neutral` (§18's K=8 MoE run) hit a transient dip at epoch ~7540-7600
(`eval/success_rate` 92.5%→88.3%, self-corrected by epoch 7900). Diagnosed as an outsized PPO
update, not the earlier NaN-broadcast bug (`critic/bad_grads_count` stayed 0) — see memory
`moe_20946_neutral_run.md` for full evidence. `examples/experiments/mimic/mlp_moe_stable.py` tests
whether two already-implemented-but-unused PPO safety knobs prevent a repeat.

**What changed vs. `mlp_moe.py`** (architecture identical — same K=8 `MoEMLPConfig`, same critic,
same env/rewards):
1. `adaptive_lr=AdaptiveLRConfig(enabled=True, desired_kl=0.01)` — was fully disabled in
   `mlp_moe.py`. Halves actor/critic LR when post-update KL > 2×`desired_kl`, grows it back ×1.5
   (capped at `max_lr`) when KL is well under target.
2. `actor_clip_frac_threshold` tightened `0.6 → 0.4` — skip remaining actor minibatch updates for
   the epoch earlier, before a large fraction of the batch is already clipped.

Both are guard rails only (no effect on well-behaved epochs), deliberately isolated from any
exploration/entropy change (`learnable_std` was considered, held back for a separate ablation to
avoid confounding — go/1+2 decision).

**Launch mode — warm start, not from scratch:** `--checkpoint results/hhi_moe_20946_neutral/
last.ckpt` + new `--experiment-name hhi_moe_20946_neutral_stable`. `train_agent.py`'s
`detect_checkpoint_mode()` treats new-name+checkpoint as "warm_start" (executes this file fresh,
so the config changes actually apply; only weights are loaded) — different from same-name
"resume" (reloads pickled config, ignores CLI/file changes). Chosen to directly test recovery
from the current ~90%+ state for a fraction of the compute, at the cost of not re-testing whether
this also fixes the early-training clip_frac instability (epochs 247-931).

**Status (2026-07-09):** launched, running, looking good so far. See memory
`moe_20946_neutral_run.md` for live metrics as they come in.

================================================================================

## 25. Persistent-Failure-Cluster Overlap — MoE vs. Baseline (2026-07-09)

**Question:** does GMT-MoE (§18) specifically fix `hhi_20946_neutral`'s known hard cluster
(§17 — crawl/kneel/squat/sit/backward/single-leg-balance, 1,818/20,946 clips failing all 18
analyzed epochs), or just improve everywhere uniformly?

**Method:** same indexing scheme as §17 (`global_clip_idx = rank*3491 + motion_id`, verified to
still apply — both runs use the identical shard scheme). Same 18-analyzed-epochs method, but only
epochs 3000-6400 were locally available for `hhi_moe_20946_neutral` (pod was reassigned to the
`_stable` warm-start job before syncing later epochs) — **numbers below are a lower bound**,
`eval/success_rate` was still rising past epoch 6400. Output: `results/hhi_moe_20946_neutral/
persistent_failures.txt`.

**Result — MoE roughly halves the persistent-failure set, unevenly across categories:**
- 18/18-persistent count: baseline 1,818 (8.7%) → MoE 916 (4.4%), **-49.6%**.
- 861 clips (47.4% of baseline's set) still fail under both — the genuinely hard core.
- Only 55 new persistent failures appeared under MoE that weren't in baseline (6.0% of MoE's
  set) — minimal new regressions, this is a real fix, not a shuffle.
- **Fix rate by category (of baseline's members; corrected after fixing a regex bug that missed
  gerunds/plurals — "crawls," "kneeling," "balancing" — in the first pass):** single-leg-balance
  63% fixed, lie/get-up/push-up 59%, kneel 55%, sit 55%, squat/crouch 53%,
  balance-on-object/beam 43%, backward 39%, **crawl/all-fours only 24% (272→208 clips) — the one
  clear outlier, every other category improved 39-63%**.

**Takeaway:** MoE is not "too weak to learn these" in general — most categories responded well to
more capacity/routing with zero reward changes, which argues against a blanket "too complicated"
explanation. Crawl/all-fours is the exception: capacity alone barely moved it, so it likely needs
a different mechanism, not just more experts. Best next candidate: physics-corrected reference
distillation (hard-motion-solutions-survey rec. 4) — tests directly whether crawl's raw mocap
references are borderline-infeasible for this body (a real possibility for hands-and-knees mocap,
prone to foot-penetration/self-collision on retarget) rather than a policy-capacity problem.

results/hhi_moe_20946_neutral/persistent_failures_18of18.txt


CUDA_VISIBLE_DEVICES=0 python protomotions/record_video_mor.py \
      --checkpoint results/hhi_moe_20946_neutral_stable/score_based.ckpt \
      --motion-file /workspace/hhi_moe_stable_top8_failures.pt \
      --simulator isaacgym \
      --num-envs 8 --output output/videos/hhi_moe_stable_top8_failures.mp4

================================================================================

## 26. `hhi_moe_20946_neutral_stable` Status Pull + Implausible-Motions Triage + Next Fine-Tune Plan (2026-07-10)

**Status pull (epoch 16400/16427):** `eval/success_rate` **95.9%** (up from 93.4% at epoch 6400,
§24's launch point), `eval/gt_error/mean` 0.102, `eval/gr_error/mean` 0.145,
`eval/gt_error/failure_rate` 4.1%. Guard rails holding: `actor/clip_frac` 0.129 (well under the
0.4 threshold), `actor`/`critic` `bad_grads_count` 0, `moe_load_balance_loss` 0.999,
`moe_expert_utilization_std` 0.021 — no sign of a §24-style epoch-7540 dip recurring so far.

**Implausible-motions triage:** of the 5,958 clips in `persistent_failures_final.txt`
(epochs 8461-16400, 41 evals), keyword-scanned descriptions for support-object dependency (sit on
a chair, lean on a table/counter, hand on a wall/railing, actual stairs, vehicle seat, platform) —
objects this run's simulator does not have (`scene_file=None`, no scene objects). 23 clips flagged
and manually spot-checked, written to `results/implausible_motions.json`
(`motion_id`/`global_clip_idx`/`description`/`epochs_failed`). Broader keyword buckets (held
objects like ball/cup/instrument — usually kinematically fine without weight-bearing dependency —
and second-person/partner interactions) bring the union to 190/5,958 (3.2%), but only the 23
support-dependent ones are a real physical-plausibility claim.

**[Decision] Keep all motions, exclude nothing.** Neither the 23 implausible clips nor the rest of
the failed set (including the 438-clip 41/41-persistent core) will be dropped from training or
eval, for Stage 2 or otherwise — consistent with the "full HumanML3D library" scale claim
(`project_overview` memory) and the earlier decision (§25 / `moe_20946_neutral_run` memory) not to
chase the persistent-failure cluster further. `results/implausible_motions.json` is a reference
list only. Explicit ask: see what pose the policy converges to on these 23 clips (e.g. "kneels
with stool" with no stool present) rather than filter them out.

**[Plan] Next fine-tune pass — cheap, warm-startable from `hhi_moe_20946_neutral_stable`'s current
checkpoint, no architecture change, to be run tomorrow:**
1. **`learnable_std=True` + `entropy_coef`** — actor's action log-std is currently frozen at the
   initial `actor_logstd=-2.9` for the whole run (`learnable_std=False`). This is the exploration
   change that was deliberately held back from §24's guard-rail run to avoid confounding "did the
   guard rails help" with "did more exploration help." Now that the guard-rail run has plateaued
   near 96%, this is the natural next single-variable test.
2. **Tighten `adaptive_lr.desired_kl`** (currently 0.01) and/or **`actor_clip_frac_threshold`**
   (already 0.6→0.4 in §24, could go to 0.3) — smaller, gentler late-stage policy updates, aimed at
   squeezing out remaining gains without risking another transient dip.
3. **Anneal `e_clip`** (PPO clip epsilon, flat 0.2 for the whole run so far) — narrowing it late in
   training is a standard fine-tune-the-fine-tune trick; not schedule-able yet, would need a small
   code change to `protomotions/agents/ppo/agent.py`/`config.py` first.
4. **`moe_load_balance.lambda_lb`** sweep (currently 0.01, never tuned) — lowest-priority of the
   four; `moe_load_balance_loss` is already ~0.999 and `moe_expert_utilization_std` only 0.021, so
   there's little imbalance left to fix.

**[Decision] Not touching the expensive options** (`num_experts` K sweep away from 8,
`expert_layers` width/depth) — these require training from scratch (checkpoint won't load across
an architecture change) and reopen the K/capacity sweep the user already deprioritized when
`hhi_moe_20946_neutral_stable` was judged "good enough" (§25 / `moe_20946_neutral_run` memory,
2026-07-09 decision to stop chasing the persistent-failure cluster).

================================================================================

## 27. `hhi_wide_20946_neutral` vs. `hhi_moe_20946_neutral_stable` — Capacity-Matched Control Check (2026-07-11)

Pulled both runs' `eval/success_rate`/`gt_error`/`gr_error` from tfevents (identical `num_envs`/
`batch_size` = 6144/24576, same 20,946-clip dataset, epochs directly comparable), cross-checked
against raw `failed_motions/` counts.

| metric (latest epoch) | wide (ep. 8600) | MoE stable (ep. 16400) |
|---|---|---|
| `success_rate` | 96.21% | 95.90% |
| `gt_error/mean` | 0.1005 | 0.1018 |
| `gr_error/mean` | **0.179** | **0.145** |
| failed clips (raw) | 793/20946 | 859/20946 |

**Not a clean win for wide.** Success rate and `gt_error` are a statistical wash — both runs
oscillate ±1pp success epoch-to-epoch over their last 8 evals, well inside wide's apparent edge.
`gr_error` (rotation) is the one real, consistent gap, and MoE wins it: wide sits 0.179-0.195
across its last 8 evals vs. MoE stable's 0.145-0.153 — a stable ~20-25% difference, not noise.

**Wide's real advantage: training efficiency, not final quality.** It reached this performance
from scratch in 8600 epochs vs. MoE stable's ~16400 (8461 inherited from the `hhi_moe_20946_neutral`
base run + ~7939 more in the `_stable` warm-start) — roughly half the epochs to the same success
rate. Frame as "capacity alone gets most of the way, faster; MoE routing still wins on rotation
tracking," not "wide beats MoE."

Caveats: neither run confirmed converged/plateaued; §26's planned `learnable_std` fine-tune for
MoE stable had not launched as of this check, so the MoE number isn't necessarily final either.

**Persistent-failure overlap (2026-07-11), `persistent_failures_final.txt` vs. wide's own
43/43-persistent set (476 clips, epoch 200-8600, its whole run so far) vs. MoE-stable's 41/41-set
(438 clips, epoch 8461-16400):** 364 clips overlap (83.1% of MoE-stable's set, 76.5% of wide's) —
largely the same hard core (crawl/all-fours, kneel, single-leg-balance) fails under both. Where
they diverge, MoE-stable comes out slightly ahead: 112 clips only wide still fails (MoE-stable
already fixed them) vs. only 74 clips only MoE-stable still fails (wide already fixed them).

**[Decision] Let `hhi_wide_20946_neutral` keep training rather than call this comparison now.**
At epoch 8600 wide is not clearly better than MoE-stable — tied on success_rate/gt_error, behind
on gr_error and on the persistent-failure overlap above — so drawing a "wide wins" conclusion at
this point would be premature. Revisit once wide has logged more epochs.

================================================================================

## 28. Cheap fine-tuning ideas for `hhi_wide_20946_neutral` (2026-07-11)

As of epoch 11000, wide has moved past the §27 snapshot: success_rate 97.29% (was 96.21%),
gt_error/mean 0.0743 (was 0.1005), gr_error/mean 0.157 (was 0.179, closing on MoE-stable's 0.145).
Still improving epoch-to-epoch but decelerating (~1.9pp/1000ep around epoch 5000-6200, down to
~0.45pp/1000ep by epoch 8600-11000) — diminishing returns, not yet flat.

Same frozen-exploration setup as MoE had before §26's fine-tune plan, so the same cheap,
warm-startable, no-architecture-change levers apply directly (pure hyperparameter/schedule
changes on the existing PPO setup — no new reward terms, no new `MdpComponent`):

1. **`learnable_std=True`** — actor's `actor_logstd` is frozen at -2.9 for the whole run
   (`learnable_std=False`), identical to MoE's pre-§26 setup. With success-rate gains decelerating,
   this is the natural next single-variable test: more late-run exploration, paired with
   `entropy_coef` (currently 0.005) so it doesn't collapse back down.
2. **Tighten `adaptive_lr.desired_kl`** (currently 0.01) and/or **`actor_clip_frac_threshold`**
   (currently 0.6, could go to 0.4/0.3) — gentler late-stage policy updates. Matches the observed
   deceleration curve: smaller steps late is the fix for a flattening-but-still-rising curve.
3. **Anneal `e_clip`** (PPO clip epsilon, flat 0.2 for the whole run so far) — needs a small code
   change to `protomotions/agents/ppo/agent.py`/`config.py` to make it schedulable first, but a
   standard fine-tune-the-fine-tune move.

MoE's 4th §26 item (`moe_load_balance.lambda_lb` sweep) doesn't apply here — wide has no gate/
load-balance loss.

**Not included here (reward/env changes, out of scope for this note):** HuB CoM-over-stance-foot
reward, symmetry loss, termination curriculum — see [[hard_motion_solutions_survey]] recs 1-3.
Deliberately left out — those change the task/reward, not just the optimizer schedule, and would
confound a capacity-vs-routing comparison the same way §26 avoided confounding MoE's guard-rail
run.

**Not started** — same as MoE's §26 list, this is a plan, not yet launched.

================================================================================

## 29. `hhi_wide_20946_neutral` status check — plateaued, gr_error gap closed (2026-07-12)

Pulled `eval/success_rate`/`gt_error`/`gr_error` from `lightning_logs/version_0` tfevents, epochs
11000-12800 (last local checkpoint: `last.ckpt`/`epoch_12000.ckpt`, both dated 2026-07-11 ~21:50;
no newer local data as of this check on 2026-07-12 morning — worth confirming the pod is still
running before assuming training has stalled).

| epoch | success_rate | gt_error/mean | gr_error/mean |
|---|---|---|---|
| 11000 | 97.29% | 0.0743 | 0.157 |
| 11800 | 96.68% | 0.0893 | 0.160 |
| 12200 | 97.43% | 0.0732 | 0.147 |
| 12600 | 96.42% | 0.0810 | 0.163 |
| 12800 | 97.20% | 0.0760 | 0.151 |

**Plateaued, not still climbing.** §28 (epoch 11000) framed this as "still improving but
decelerating." That's no longer the right read — success_rate has been oscillating 96.4-97.4%
with no net movement for 1800 epochs (11000-12800), well inside run-to-run noise. Same story for
gt_error/mean (0.072-0.089, no trend) and gr_error/mean (0.145-0.163, no trend). This looks flat,
not decelerating-but-rising.

**gr_error gap vs. MoE-stable has closed.** §27 (epoch 8600) found wide clearly behind MoE-stable
on rotation tracking: 0.179 vs. 0.145, a "stable ~20-25% difference." As of epoch 12800, wide's
gr_error is 0.145-0.163 across its last 5 evals — overlapping MoE-stable's 0.145 (epoch 16400,
see [[moe_20946_neutral_run]]), sometimes matching it exactly (epoch 12200: 0.147). The one
real, consistent MoE advantage from §27 is no longer clearly present at wide's current epoch
count. Combined with wide's known ~2x training-efficiency edge (§27), this now reads closer to
"wide matches MoE-stable on all three metrics, in fewer epochs" than "MoE wins rotation tracking."

**No instability recurrence.** No epoch-7540-style dip (the MoE run's transient PPO instability
from [[moe_20946_neutral_run]]) visible anywhere in 11000-12800 — success_rate never drops more
than ~1pp between consecutive 200-epoch evals.

**Not done yet**: none of the §28 fine-tune ideas (`learnable_std`, tighter `adaptive_lr`/
`actor_clip_frac_threshold`, `e_clip` anneal) have been launched — this is a pure status check on
the existing run. Given the plateau, `learnable_std=True` (§28 item 1) is now a stronger
candidate than before: a flat success-rate curve with frozen `actor_logstd=-2.9` since the start
is exactly the "exploration is likely limiting further gains" signal it was proposed to address.

**How to apply:** when next asked about this run, re-check `lightning_logs/version_0` tfevents
and confirm whether training has resumed past epoch 12800 (and whether the pod is still active)
before reusing these numbers, and check whether the §28 fine-tune has launched before
re-suggesting it.

================================================================================

## 30. §28's 3 fine-tune knobs, explained in plain English (2026-07-12)

User asked for a plain-English explanation of the §28 fine-tune ideas before deciding whether to
launch any of them. Recorded here so the reasoning behind picking item 1 first doesn't have to be
re-derived next time.

**1. `learnable_std` + `entropy_coef` — how much random noise the policy adds to its own actions.**
The actor outputs a mean action plus a spread (std) around it, and samples from that during
training — that's how it explores. `actor_logstd=-2.9` has been a fixed constant for this run's
entire duration, so the exploration budget never changes. `learnable_std=True` lets the network
adjust that spread on its own; `entropy_coef` (a small bonus for staying spread-out) is paired
with it because a learnable std left alone tends to collapse toward zero noise once the policy
feels confident — even if it's actually just stuck. Net effect: "let the policy try more varied
things again."

**2. `adaptive_lr.desired_kl` + `actor_clip_frac_threshold` — how big a step each update takes.**
KL divergence measures how much the policy's behavior actually changed after an update;
`desired_kl=0.01` is the target amount of change per update, and `adaptive_lr` shrinks/grows the
learning rate to hit that target. `clip_frac` is a related signal: PPO caps how far any single
action's probability can move in one update, and `clip_frac` is the fraction of the batch that hit
that cap — high clip_frac means updates are frequently maxing out the safety limit. Tightening
both numbers means smaller, more conservative updates late in training. Net effect: "take gentler
steps."

**3. `e_clip` — the actual size of PPO's safety cap.** This is the number `clip_frac` (above) is
measured against. PPO clips the new/old action-probability ratio to `[1-e_clip, 1+e_clip]` —
currently 0.2 (±20%), flat for the whole run. Annealing it means shrinking that window later in
training (e.g. to ±10% or ±5%) — same spirit as item 2, but a direct hard cap rather than a
reactive KL-based adjustment. Listed as the most work of the three because `e_clip` is currently a
fixed constant in `protomotions/agents/ppo/agent.py`/`config.py`, not schedule-able yet.

**One-line summary:** item 1 pulls toward "more exploration," items 2 and 3 pull toward "more
caution" (two different levers for the same idea — one reactive, one a hard cap). This is also why
item 1 is the natural first thing to try alone on `hhi_wide_20946_neutral`: a flat/plateaued curve
with no clip_frac spikes and no instability dips (§29) reads as an exploration problem, not a
stability problem — so the "more caution" knobs (2, 3) don't have an obvious signal pointing at
them yet.

**How to apply:** reuse this explanation instead of re-deriving it if the user (or a future
session) asks what these three knobs mean again; update if the actual config field names or
defaults in `protomotions/agents/ppo/config.py` change.

================================================================================

## 31. Implemented §30 item 1 as `mlp_wide_explore.py` — single-variable, not bundled (2026-07-12)

User asked whether to implement all 3 of §28's fine-tune ideas together. Decided no, same
single-variable reasoning as §30's summary, plus a concrete precedent already in the repo:
`mlp_moe_stable.py`'s own docstring explicitly considered bundling `learnable_std` into that run
and rejected it for confounding reasons — bundling all 3 here has the same problem, worse, since
item 1 (more exploration) pulls the opposite direction from items 2/3 (more caution).

**Created `examples/experiments/mimic/mlp_wide_explore.py`** — identical to `mlp_wide.py` except
`learnable_std=True` on the actor config. `entropy_coef` left at `PPOAgentConfig`'s default
(0.005), no override needed — it already existed but was inert with `learnable_std=False`. Items
2 and 3 deliberately NOT included. Follows the same warm-start pattern as `mlp_moe_stable.py`:
new experiment file + new `--experiment-name` so `train_agent.py` executes it fresh (picks up
`learnable_std=True`) while only using `--checkpoint results/hhi_wide_20946_neutral/last.ckpt`
for weight initialization. Architecture otherwise unchanged (same widened trunk, same critic), so
the checkpoint loads with no shape mismatches.

Launch command (6x A40, same envs/batch as the parent run) is in the file's docstring:
```
python protomotions/train_agent.py \
    --robot-name smpl_mor_neutral --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide_explore.py \
    --experiment-name hhi_wide_20946_neutral_explore \
    --checkpoint results/hhi_wide_20946_neutral/last.ckpt \
    --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
    --num-envs 6144 --batch-size 24576 --ngpu 6 \
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_20946_neutral_explore
```

**Not yet launched** — file created and syntax-checked (`py_compile`) locally only;
`pre-commit`/`ruff` weren't available in this environment to lint it, so re-run
`pre-commit run --files examples/experiments/mimic/mlp_wide_explore.py` before committing.
Actual training launch needs the RunPod pod and is a user action, not something done from this
session.

**Budget/stop rule (carried over from §29):** re-check after ~2000-3000 epochs; if
`eval/success_rate`/`eval/gr_error` haven't clearly moved, stop and take whichever checkpoint is
better (this one or `hhi_wide_20946_neutral`'s own `last.ckpt`) to Stage 2 rather than continuing
to chase marginal gains. Items 2/3 remain available as follow-ups but are not queued next by
default — only pick them up if there's a specific instability signal (clip_frac spikes, dips)
that this exploration-only change doesn't explain, per §30's reasoning.

**How to apply:** before re-suggesting a fine-tune for this run, check whether
`hhi_wide_20946_neutral_explore` has been launched/finished (`results/hhi_wide_20946_neutral_explore/`
existing + its tfevents) and re-pull current numbers rather than reusing this entry's plan as if
it were a completed result.

================================================================================

## 32. Stage 2 Architecture Plan — Frozen Backbone + Zero-Init LoRA-Style Residual Adapter (2026-07-13)

**Context:** the original Stage 2 plan (§"Q1/Q2 Answer" above) was a full fine-tune of the
identical `mlp.py`/`mlp_wide.py` architecture, loaded via strict `load_state_dict()` — no
architectural change from Stage 1, just a lower LR and a normalizer reset. This section proposes
an alternative: freeze the Stage 1 trunk entirely and add a small trainable adapter conditioned
on `morphology_obs`, so Stage 2 can't catastrophically forget the motion prior learned on all
20,946 clips, and the actual optimization problem at Stage 2 is tiny (adapter params only)
instead of re-touching all ~45M trunk weights across 2.68M motion-instances (20,946 clips x 128
shapes). Not yet implemented — this section is the design, not a result.

**Base checkpoint:** `hhi_wide_20946_neutral` (not the MoE line) — as of this writing
`note/README.rclone.md`/`README.runpod.md` have uncommitted edits pulling that run's checkpoint
zip from R2 and relaunching `mlp_wide_explore.py` from its `score_based.ckpt`. Recommend basing
Stage 2 on the current best non-regressed wide-MLP checkpoint rather than blocking on the
`_explore` learnable_std fine-tune's outcome (that fine-tune already failed once as a straight
regression per `moe_20946_neutral_run.md`/[[architecture_research]] memory, before being retried) —
these are two independent workstreams and Stage 2 planning shouldn't wait on it.

### Why not just full fine-tune (the original Q1/Q2 plan)

One concrete finding motivates freezing over full fine-tune: `morphology_obs` (11-dim,
`[gender_id, betas/3.0]`) is already concatenated into the trunk's input in `mlp_wide.py`
(`morphology_obs_factory()`), but during Stage 1 it was always exactly zero — so gradient w.r.t.
those input columns was exactly zero for the entire run (`dL/dW = dL/dh · x`, `x≡0` identically).
The frozen trunk's only existing morphology pathway is untrained noise from `weight_init`, not a
degenerate-but-real signal. Full fine-tuning has to fix this by letting the whole 6x2896 trunk
drift; a residual adapter re-injects real shape signal without touching it, at the cost of a
smaller-capacity intervention (see escalation ladder below if that proves insufficient).

### Design

New module `LoRAResidualMLPWithConcat` **subclasses `MLPWithConcat` directly** (`protomotions/agents/common/mlp.py:67`) —
does not wrap it — so parameter names (`norm.*`, `mlp.*`) stay identical to `mlp_wide.py`'s. This
is what lets the Stage 1 checkpoint load with zero key-remapping.

```
mu_model forward:
  base_out   = super().forward(obs)        # frozen 6x2896 trunk, unchanged from Stage 1
  bottleneck = adapter_down(all_obs)        # shared LazyLinear, in -> r, TRAINABLE
  W_up_e     = hypernet(morphology_obs)     # per-env (r -> action_dim) matrix, TRAINABLE
  delta      = einsum(bottleneck, W_up_e)   # per-env residual
  output     = base_out + delta
```

- **Freeze-on-first-forward, not in `__init__`.** `base_agent/agent.py:189-196` runs a dummy
  forward pass (`agent.setup()`) to materialize `LazyLinear` layers *before* `agent.load(checkpoint)`
  runs (`train_agent.py:834-836`). So the module sets `requires_grad=False` on `norm`/`mlp`
  params the first time its own `forward()` executes, after materialization — self-contained,
  no changes needed to the training loop's setup/load ordering. Confirmed safe either way:
  `load_state_dict` copies into `param.data` regardless of `requires_grad`, so freeze-before-load
  vs. freeze-after-load doesn't matter for correctness.
- **Zero-init the hypernetwork's last layer** so `delta ≡ 0` at Stage 2 step 1 — Stage 2 starts
  as an *exact* continuation of the Stage 1 policy, not a fresh regression. Important under PPO,
  where a bad initial policy burns rollout compute before recovering. Same convention FARM uses
  for its zero-initialized adapters (`architecture_research.md` point 4).
- `adapter_down`'s input dim is only known after concat, hence `LazyLinear` — the same
  "unknown input dim" wrinkle the reverted HyperDistill build hit (`architecture_research.md`
  point 3), solved the same way (shared projection, not per-env). `hypernet`'s output dim
  (`robot_config.number_of_actions`) is known at config-build time, so no ambiguity there —
  simpler than HyperDistill's per-layer version, which needed the same workaround at every layer.
- **Critic: unchanged, unfrozen, plain `MLPWithConcatConfig`**, fully fine-tuned, no adapter —
  no prior to protect, matches the existing "critic stays flat-concat" precedent from §18.
- Recommended rank `r=16` — one global bottleneck now carries what HyperDistill spread across
  per-layer adapters (r=8-16 there), so erring toward the higher end.

### One required code change

`self.model.load_state_dict(state_dict["model"])` at `base_agent/agent.py:262` is unconditionally
strict — the new adapter/hypernet keys don't exist in the Stage 1 checkpoint, so this raises as-is
(this is the same strict-loading constraint noted in the Q1 Answer section above, "Checkpoint
loading is strict... no `strict=False`"). Add an opt-in field to the base agent config,
`allow_partial_checkpoint_load: bool = False`, defaulting to the current strict behavior
everywhere (including the existing warm-start pattern used for `hhi_moe_20946_neutral_stable`,
which relies on strict matching to catch architecture drift as a safety net):
```python
self.model.load_state_dict(
    state_dict["model"], strict=not self.config.allow_partial_checkpoint_load
)
```
Only the new Stage 2 experiment file sets it `True`.

### New files

- `protomotions/agents/common/lora_residual_mlp.py` — `LoRAResidualMLPWithConcatConfig` +
  `LoRAResidualMLPWithConcat` (matches the existing convention of config-alongside-module used by
  `film_mlp.py`/`shape_embed_mlp.py`/`moe_mlp.py`, not centralized in `agents/common/config.py`).
- `examples/experiments/mimic/mlp_wide_lora_stage2.py` — copy of `mlp_wide.py`, swap `mu_model`
  for `LoRAResidualMLPWithConcatConfig(...)` wrapping the same `layers=[2896]x6` spec,
  `allow_partial_checkpoint_load=True`, actor LR cut ~5-10x (2e-5 -> ~2-4e-6) per the existing
  "Stage 2 LR should be 5-10x lower than Stage 1" decision in the Q2 design question above, critic
  LR unchanged (1e-4). Still runs `tools/reset_morphology_normalizer.py` on the checkpoint first,
  unchanged — that fix is orthogonal (obs-normalizer saturation, not backbone forgetting) and
  still needed regardless of this architecture change.

### Verification plan before RunPod (mirrors the CPU-verification convention already used for
`moe_mlp.py` and the reverted HyperDistill build)

1. CPU dummy-forward on a fake TensorDict: confirm frozen (`norm`/`mlp`) params get zero grad,
   adapter (`adapter_down`/`hypernet`) params get nonzero grad, `delta` is exactly 0 at init, and
   different `morphology_obs` per env produce different `delta` values (mechanism is live).
2. Load an actual `hhi_wide_20946_neutral` checkpoint with `strict=False`; assert `missing_keys`
   is exactly `{adapter_down.*, hypernet.*}` and `unexpected_keys` is empty. Then run one forward
   pass and confirm the output is bit-identical to running that same checkpoint through
   unmodified `mlp_wide.py` (valid since `delta=0` at init) — this is the single check that
   catches a silently-wrong load (key typo, double-normalization, etc.).
3. Small-scale smoke test on the existing 2-shape `hhi_stage1_merged6` data
   ([[stage1_v2_data]]) before committing to the full 128-shape Stage 2 run.

### Escalation ladder if the single global residual underperforms

Increase `r` -> add a second adapter mid-trunk (requires splitting `self.mlp` into two frozen
halves, more invasive than the current design) -> fall back to full fine-tune (the original
Q1/Q2 plan, `mlp.py` unmodified) as the comparison baseline -> revisit true per-layer LoRA (the
original reverted HyperDistill design, full mechanism preserved in `architecture_research.md`
point 3, if this simpler global-residual version proves insufficient).

**How to apply:** this is a design doc, not yet built. Before implementing, re-check which Stage 1
checkpoint (`hhi_wide_20946_neutral` vs. its `_explore` fine-tune vs. the MoE line) is the current
best candidate — the wide-vs-MoE question was still open as of §29/§31 above.

**Decided (2026-07-13): base on `hhi_wide_20946_neutral` directly, not the `_explore` fine-tune.**
User confirmed this explicitly. Consistent with the R2/RunPod checkpoint pull already in flight
per the "Base checkpoint" note above.

### [Implementation] `lora_residual_mlp.py` built and CPU-verified (2026-07-13)

Built as designed above:
- `protomotions/agents/common/lora_residual_mlp.py` — `LoRAResidualMLPWithConcatConfig` +
  `LoRAResidualMLPWithConcat(MLPWithConcat)`. Freeze-on-first-forward and hypernet last-layer
  zero-init both implemented as planned, self-contained in the module (no changes needed to
  `BaseAgent.setup()`'s dummy-forward/materialization flow).
- `protomotions/agents/base_agent/config.py` — added `allow_partial_checkpoint_load: bool = False`
  to `BaseAgentConfig`.
- `protomotions/agents/base_agent/agent.py:262` — `load_parameters` now does
  `self.model.load_state_dict(state_dict["model"], strict=not self.config.allow_partial_checkpoint_load)`,
  logging missing/unexpected keys when non-strict. Default preserves strict loading everywhere
  else (including the existing `hhi_moe_20946_neutral_stable` warm-start pattern).

**CPU verification (ad hoc script, not committed) confirmed all 4 planned checks:** only
`adapter_down`/`hypernet` params have `requires_grad=True` after the first forward; the model's
output is bit-identical to an unmodified `MLPWithConcat` loaded with the same weights (delta
exactly 0 at init, `max_diff=0.0`); no gradient ever reaches the frozen base; different
`morphology_obs` per env produces different outputs after a couple of optimizer steps.

**One real finding, not a bug:** zero-initializing the hypernetwork's *last* layer blocks
backprop to every upstream param on the very first step — `adapter_down` and the hypernet's own
earlier layers (`hypernet.0.*`) get exactly zero gradient on step 1, only `hypernet`'s last
layer (`hypernet.2.*`) does (matches standard LoRA "B=0" behavior: `dL/dA = B^T . dL/d(delta)`,
zero when `B=0`; `dL/dB = dL/d(delta) . A^T`, nonzero since `A` isn't zero-init). Gradients reach
`adapter_down` and earlier hypernet layers starting step 2, once the last layer has moved off
zero. Worth remembering if early Stage 2 training looks inert for the first several steps before
an eval — that's expected, not stalled.

**Not yet built:** `examples/experiments/mimic/mlp_wide_lora_stage2.py` (the actual experiment
file wiring `LoRAResidualMLPWithConcatConfig` into an actor config, warm-started from
`hhi_wide_20946_neutral`). `pre-commit`/`ruff` weren't available in this environment to lint the
changed files — run `pre-commit run --files protomotions/agents/common/lora_residual_mlp.py
protomotions/agents/base_agent/agent.py protomotions/agents/base_agent/config.py` before
committing.

================================================================================

## 33. Stage 2 Streaming Data Loader — implemented (2026-07-14)

Built the design from `note/README.stage2-streaming-loader-plan.md` (328 R2 shards, ~1.1 TB,
too big for RAM/VRAM at once). `mlp_wide_lora_stage2.py` (built since §32, superseding that
section's "not yet built" note) now has a working data path to the full `hhi_stage2` set, not
just the small `hhi_stage1_merged6` smoke-test data.

**New files**, matching the plan exactly:
- `protomotions/components/motion_lib_pool.py` — `StreamingMotionLibConfig(MotionLibConfig)` +
  `MotionLibPool(MotionLib)` + `FileDownloader` (background-thread `rclone copy` wrapper).
  Rotation is a pure function of `current_epoch`
  (`target_shard_idx = (current_epoch // epochs_per_shard) % len(rank_files)`), per-rank
  disjoint shard slices via deterministic shuffle + `files[rank::world_size]`, in-place
  `load_from_file` swap (same object, so `env.motion_lib`/`motion_manager.motion_lib`/
  `agent.motion_lib` never need re-pointing).
- `protomotions/agents/callbacks/motion_shard_rotation.py` — `MotionShardRotationCallback`:
  `before_play_steps` calls `maybe_rotate` and sets `agent._force_full_env_reset` on an actual
  swap; `on_load_checkpoint_end` calls `sync_to_epoch` (unconditional) for resume.

**Modified**, matching the plan except one addition (see gap below):
- `protomotions/agents/base_agent/agent.py` — added `self._force_full_env_reset = False` next
  to `_skip_next_policy_update`; in `fit()`, right after `fabric.call("before_play_steps", self)`
  and before the rollout loop, force `done_indices = arange(num_envs)` when the flag is set.
- `protomotions/train_agent.py` — appends the `MotionShardRotationCallback` target dict to
  `callbacks` when `isinstance(motion_lib_config, StreamingMotionLibConfig)`, next to the
  existing `args.use_slurm` wiring. `motion_lib_config` is in scope there in all three modes
  (resume/warm_start/fresh), confirmed by reading each branch.
- `examples/experiments/mimic/mlp_wide_lora_stage2.py` — added `additional_experiment_arguments`
  (`--r2-motion-source`, `--motion-cache-dir`, `--epochs-per-shard`, `--shard-shuffle-seed`);
  `motion_lib_config()` returns a `StreamingMotionLibConfig` when `--r2-motion-source` is set,
  else falls back to the existing plain `MotionLibConfig(motion_file=...)` path — so the 2a
  smoke-test command is untouched.

**Gap found in the plan and fixed:** `protomotions/utils/component_builder.py`'s
`build_motion_lib_from_config` hardcoded `MotionLib(config=motion_lib_config, device=device)` —
it ignored `motion_lib_config._target_` entirely, so a `StreamingMotionLibConfig` would have
silently built a plain `MotionLib` instead of a `MotionLibPool`. Fixed to resolve the class via
`get_class(motion_lib_config._target_)`, the same convention `agents/ppo/model.py:59-60` already
uses to resolve `mu_model`/critic classes (explicit `config=` kwarg, not the generic
`instantiate()` helper — `instantiate()` flattens config fields into kwargs, which fits
`nn.Module`-style configs but not `MotionLib.__init__(self, config, device)`'s signature).
Verified the ordinary (non-streaming) `MotionLibConfig` path is unaffected: `build_motion_lib_from_config(MotionLibConfig(motion_file=None), device="cpu")` still returns
a plain empty `MotionLib`.

**One robustness addition beyond the plan:** the plan's "kept exactly 2 local files per rank"
claim doesn't quite hold across a discontinuous jump (e.g. `sync_to_epoch` after resume landing
on a shard other than whatever was mid-prefetch) — the superseded prefetch download would be
orphaned on disk with nothing to ever delete it. Added a sweep at the end of
`_ensure_shard_loaded` that deletes any `.pt` file in the rank's cache dir other than the
current shard and the newly-started prefetch target, so the ≤2-files-per-rank bound holds even
across resume, not just during normal one-step-at-a-time rotation.

**CPU verification (ad hoc script, not committed,
`/tmp/.../scratchpad/verify_motion_lib_pool.py`), per the plan's "CPU, no cloud" check:** 3 tiny
dummy shards in a local directory as `r2_source` (rclone works against local paths with no
credentials). Confirmed: construction loads shard 0 synchronously; `maybe_rotate` is a no-op
within `epochs_per_shard`; rotating at the epoch boundary swaps `gts`/etc. to the new shard's
content, deletes the old local file, and starts prefetching the next one; rotation wraps around
correctly at the end of `rank_files`; `sync_to_epoch` after a simulated resume at an arbitrary
epoch lands on the correct (recomputed) shard directly; local cache dir never exceeds 2 files
across the whole sequence including the resume jump.

**Not yet done:** real R2 integration test against `r2:proto-data/hhi_stage2/` on RunPod (the
plan's step 2) — needs an actual pod with the R2 rclone remote configured. `pre-commit`/`ruff`
weren't available in this environment either; run `pre-commit run --files
protomotions/components/motion_lib_pool.py protomotions/agents/callbacks/motion_shard_rotation.py
protomotions/agents/base_agent/agent.py protomotions/utils/component_builder.py
protomotions/train_agent.py examples/experiments/mimic/mlp_wide_lora_stage2.py` before
committing.

================================================================================

## 34. Stage 2 v1 (hypernetwork LoRA) stopped — replaced with plain residual-adapter MLP (2026-07-17)

**Diagnosis (via wandb, run `hhi_wide_lora_stage2` / `nz2kwkr4`):** not crashing —
`bad_grads_count` stayed 0 throughout, only one transient loss spike at epoch 12940 that
settled within 10 epochs. It's a genuine **plateau**: `eval/success_rate` epochs 14200-20200
had slope ≈ 0, oscillating 53-80% (mean 70%, std 6.5pp) with no upward trend. Root cause: only
`adapter_down`/`hypernet` were trainable, at a fixed `actor_optimizer.lr=4e-6` with
`adaptive_lr.enabled=False` and `actor/clip_frac` near zero — updates were too small to keep
making progress once past the initial warm-start ramp.

**Change:** replaced §32's hypernetwork-generated per-env LoRA weight matrix with a much
simpler design — a small trainable MLP reads the same (already morphology-including) input
the frozen trunk sees and outputs the residual directly, no bottleneck/rank/einsum:

    base_out = frozen_trunk(obs)       # unchanged Stage 1 weights
    delta    = adapter_mlp(obs)        # small trainable MLP (2x512 relu default)
    output   = base_out + delta

This is functionally the "just concatenate beta into obs" approach that worked pre-freeze,
scoped to a small trainable head instead of the whole network (trunk is still frozen).
Zero-init on the adapter's last layer preserves the same "exact continuation of Stage 1 at
init" property as §32's design.

**Files:**
- `protomotions/agents/common/residual_adapter_mlp.py` — new,
  `ResidualAdapterMLPWithConcat`/`Config`, replaces (deleted) `lora_residual_mlp.py`.
- `examples/experiments/mimic/mlp_wide_lora_stage2.py` — swapped
  `LoRAResidualMLPWithConcatConfig` → `ResidualAdapterMLPWithConcatConfig`; dropped
  `adapter_rank`/`hyper_hidden_units`; added `ADAPTER_UNITS`/`ADAPTER_NUM_LAYERS` (512, 2).
  `actor_optimizer.lr` left at `4e-6` unchanged — worth revisiting now that the adapter has a
  more direct gradient path.

Verified standalone (CPU, ad hoc): delta exactly 0 at init, trunk correctly frozen/adapter
trainable, different `morphology_obs` → different output post-update, `strict=False` load
against a plain Stage 1 checkpoint leaves only `adapter_mlp.*` missing (no unexpected keys) —
same checkpoint-compat contract as §32.

**Run transition:** old run `hhi_wide_lora_stage2` (wandb `nz2kwkr4`) left to stop on its own /
stopped on the pod, not resumed. New run launched under a new experiment name,
`hhi_wide_residual_stage2`, rather than reusing the old name — reusing it would have hit
`train_agent.py`'s resume-mode auto-detect (`results/<name>/last.ckpt` present →
loads pickled `resolved_configs.pt`, which still references the deleted LoRA class). Same
Stage 1 warm-start checkpoint (`hhi_wide_20946_neutral/last_morph_reset.ckpt`), same streaming
data config, same `--num-envs 6144 --batch-size 24576 --ngpu 6`.

## 35. Stage 2 v2 diagnosis + adapter scoped to morphology_obs only (2026-07-19)

**Diagnosis (wandb API + a remote Claude Code session with log/checkpoint access on the
training box, run `hhi_wide_residual_stage2` / `3skv3b2g`):** plateaued at epoch ~31459,
`eval/success_rate` flat since ~epoch 19000 (78.1%±4.4% over ~12,400 epochs, no net slope),
`info/episode_length` flat in lockstep (~100 vs. a ~200 target — episodes end early via
`tracking_error_term_factory`, so this and success_rate are two views of the same "policy
still falls" signal). Unlike v1, `actor/clip_frac` was healthy (0.0126 mean, ~24x v1's dead
0.0005) — so the v1→v2 direct-gradient-path fix worked, but a different ceiling was hit
anyway. Shard-rotation-driven oscillation was investigated and **ruled out**: binning all 95
eval events by epoch-distance to the nearest `epochs_per_shard=64` rotation boundary gave
near-rotation mean success_rate 0.775 (n=23) vs. far-from-rotation 0.786 (n=40) — 1.1pp
difference on a 4.4pp noise floor.

**Root cause identified: the adapter was reading the full observation, not just body shape.**
`ResidualAdapterMLPWithConcat._adapter_input()` reused `tensordict["norm_max_coords_obs"]`,
which — despite the name — is the full normalized 1014-dim concatenation of ALL `actor_in_keys`
(`max_coords_obs` + `mimic_target_poses` + `previous_actions` + `morphology_obs`), computed by
the parent `MLPWithConcat.forward()`. A CPU checkpoint forward-pass check (epoch-31620
checkpoint, 256 synthetic obs sampled from the checkpoint's own `running_obs_norm` marginals)
found the adapter's output norm was ~40% of the frozen trunk's on average (up to 136% on some
samples) — a large, high-frequency correction, not a small stabilizing one. Since the adapter's
input changed every timestep (pose/motion-target obs are high-frequency), nothing structurally
prevented its output from being high-frequency too, consistent with the run's elevated
`eval/normalized_jerk_mean` (1500-2500, vs. a Stage-1-trunk-only baseline around 358) and
`high_jerk_frame_percentage_mean` (25-40%) staying flat/elevated through the whole plateau. A
first fix attempt (raise `action_smoothness` reward weight -0.02→-0.2,
`mlp_wide_residual_stage2_smooth.py`) was designed but **abandoned before launch** in favor of
the structural fix below — a reward-shaping bandaid was judged unlikely to close a ~78%→95%,
~100→200-step gap on its own, and doesn't address why the adapter is noisy in the first place.

**Fix:** `ResidualAdapterMLPWithConcatConfig` gained a new field, `adapter_in_keys` (default
`["morphology_obs"]`), and `_adapter_input()` now concatenates only those keys instead of
reusing the trunk's full normalized input. `morphology_obs` (the 11-dim `[gender_id,
betas/3.0]`) is constant for the whole episode, so `delta` is now provably constant within an
episode too — verified standalone (CPU): after one training step, adapter output changes
0.0 for a pose-only perturbation (morphology_obs held fixed) vs. 4.05 for a morphology-only
perturbation (pose held fixed) on the same input batch. The adapter can no longer introduce
frame-to-frame jerk by construction, not by reward-penalty tuning. Tradeoff, not yet resolved:
this caps the adapter to a per-body *offset*, unable to express pose-shape interaction
corrections (e.g. a different ankle correction while crouching vs. standing) — open question
whether that matters for this failure mode; no data collected yet either way.

**Also discussed, not changed:** whether MotionLib's streaming/shard-rotation setup should
be replaced with a static full load. Confirmed via code
(`protomotions/components/motion_lib.py:701-754`, `load_from_file()`) that `MotionLib` always
moves the *entire* loaded file to the target device (GPU) — `torch.load(..., map_location="cpu")`
then an unconditional `.to(self.device)` per tensor, no lazy/on-demand paths except `max_motions`
(documented for inference, not training). Full Stage 2 is ~2.68M motion-instances / ~1.1TB,
~447k/rank across 6 ranks — ~64x `hhi_stage1_merged6`'s per-rank size (the largest static,
non-rotating load this codebase has actually run). A literal non-rotating full load for the
full dataset would OOM on load; shard rotation isn't a suboptimal choice here, it's structurally
required given how `MotionLib` loads data. If R2 network reliability during live streaming is
the actual concern, pre-downloading all 328 shards to local disk once (same shard sizing/
rotation cadence, just a local instead of R2 source) is the change that addresses that without
touching the GPU-memory model — not yet implemented, revisit if network stalls are confirmed
as a real problem separately from this diagnosis.

**Launch (not yet run):** new experiment name `hhi_wide_residual_stage2_shapeonly`, same
`mlp_wide_lora_stage2.py` experiment file (architecture default changed in
`residual_adapter_mlp.py`, no experiment-file-level override needed), same base checkpoint
`hhi_wide_20946_neutral/last_morph_reset.ckpt` (deliberately unchanged from v1/v2 — single-
variable test), same streaming data config and `--epochs-per-shard 64` (increasing shard dwell
time was discussed as a secondary/low-priority experiment, not bundled into this one — the
rotation-boundary binning above already argues against rotation cadence being the bottleneck).
Full launch command in `mlp_wide_lora_stage2.py`'s docstring.

## 36. Session summary (2026-07-19)

1. Stopped `hhi_wide_residual_stage2` (v2) after diagnosis (§35) found it plateaued at ~78%
   success_rate with the adapter reading the full observation, not just body shape.
2. Launched `hhi_wide_residual_stage2_shapeonly` (v3) — adapter now reads only `morphology_obs`;
   same base checkpoint, same data, everything else unchanged. Result pending.
3. If v3 doesn't reach target (95% success, ~200-step episodes), next step is a concat-fusion
   design — feed both the trunk's output and betas into new learnable layer(s) instead of pure
   addition — design and known risks in §35's "v4 candidate" note. Not started; waiting on v3's
   result first.


  ====


The basic problem

We have 20,951 motion clips, each with 128 body-shape versions — about 2.7 million individual motions total. That's way too
much data to fit in GPU memory at once (roughly 1.1TB). GPU memory can only hold a few hundred clips' worth at a time. So at
any given moment, only a small slice of the full dataset is actually loaded and usable — call this the "shelf."

Everything below is really about answering one question: which clips go on the shelf, and when do we change what's on it.

Two separate clocks running at different speeds

Clock 1 — every training step (fast, thousands of times per epoch): Whenever a simulated robot finishes its current motion
and needs a new one, we pick a motion only from what's currently on the shelf. This pick is weighted — motions marked "hard"
get picked more often than motions marked "easy." This part is unchanged from how Stage 1 already works; the only
difference is the pool of things to pick from is smaller (just the shelf) instead of the whole dataset.

Clock 2a — every ~64 epochs (rebuild): We look at the shelf and ask: "is there anything better we should swap onto it?" A
clip earns a slot on the shelf for one of two reasons:
- It's known to be hard (robot keeps failing it), so we want it around more.
- It's never been tried, so we don't actually k Jump to bottom (ctrl+End) ↓ — we owe it a turn.

We swap the shelf's contents to reflect this, download whatever's newly needed, throw out the least useful stuff, and continue training.

Clock 2b — every ~200 epochs (evaluation): Separately, we periodically run the robot through a batch of motions currently on
the shelf and check: did it succeed or fail? Motions it failed get marked "harder" (more likely to be picked, and more
likely to earn a shelf slot later); motions it succeeded on get marked "easier." This is the exact same success→harder /
fail→easier math Stage 1 already uses — we're not changing that formula at all.

The one genuinely new idea: a permanent memory of difficulty

Today's system (shard rotation) has no memory — every time it swaps in a new batch of clips, it forgets everything it
learned about difficulty and starts those "hard/easy" labels back at neutral. That's wasteful: if we already learned clip
#4821 is hard, we shouldn't have to relearn that from scratch every time it happens to cycle back onto the shelf.

So the new design keeps one permanent scoreboard, covering all ~20,951 clips (well, this rank's share of them), that never
  resets — even across a crash/resume. "Hard" and "easy" labels accumulate on this scoreboard for the life of the whole
  training run. The shelf is just a small, temporary window into that scoreboard — showing whichever clips currently look most
  important (hard, or unproven) — but the scoreboard itself remembers everything, forever.

  Why two clocks instead of one

  Originally the plan was to only update the shelf right after evaluation (clock 2b) — but that's slow (every 200 epochs),
  slower than what's running today (every 64 epochs). That would mean we introduce fresh, untried clips onto the shelf less
  often than we do now — a step backward. So we decoupled it: the shelf gets refreshed on its own faster schedule (clock 2a,
  every ~64 epochs, matching today's pace), using whatever the scoreboard currently says — even if the scoreboard itself
  hasn't been updated by a fresh evaluation in a while. The scoreboard-updating (clock 2b) stays on its own slower pace,
  because that part genuinely requires running real evaluation rollouts, which is expensive.

## 37. Global Clip-Priority Sampling — implemented (2026-07-24)

Implements the design in `note/README.stage2-global-clip-sampling-plan.md` (per-clip repackaging
to `r2:proto-data/hhi_stage2_per_clip/`, done 2026-07-23) plus the "two clocks" explanation in
§36's addendum above. Full implementation plan: `/home/hlz/.claude/plans/generic-finding-sparkle.md`.

**New files:**
- `protomotions/components/global_clip_pool.py` — `GlobalClipPoolConfig`/`GlobalClipPool`
  (`MotionLib` subclass). Deterministic per-rank clip vocabulary (sorted manifest + seeded
  shuffle + `[rank::world_size]` stride, same idiom as `MotionLibPool`'s shard partition).
  Persistent `global_clip_weights`/`global_clip_visit_counts` scoreboard (the "shelf's
  scoreboard" from §36, CPU tensors, checkpointed, never reset). `_select_top_k()` — a
  combinatorial-UCB priority (`weight + coefficient * sqrt(2*ln(rebuild_count+1)/(visit_count+1))`)
  picks K=256 clips/rank. `_rebuild_resident_pool`/`_materialize_resident_set` — downloads
  missing per-clip files concurrently, concatenates K per-clip dicts in memory (no temp-file
  round trip), overwrites the assembled `motion_weights` with the scoreboard's values broadcast
  per clip, loads via a new `MotionLib._load_motion_state_dict` (split out of `load_from_file` for
  exactly this in-memory use). LRU disk-cache eviction at `cache_size_multiplier * K`.
- `protomotions/agents/callbacks/global_clip_pool_rebuild.py` — `GlobalClipPoolRebuildCallback`,
  triggers `maybe_rebuild()` on its own `pool_rebuild_every` epoch cadence (default 64, matching
  today's `epochs_per_shard`) — deliberately NOT on the evaluator's `on_eval_end`/
  `eval_metrics_every` (default 200), since that would introduce new/unproven clips *less* often
  than today's shard rotation, a regression rather than an improvement (this is clock 2a vs.
  clock 2b from §36's addendum).

**Modified:**
- `protomotions/components/motion_lib.py` — split `load_from_file`, added
  `has_clip_identity_metadata()`/`build_clip_id_to_motion_ids()` (real `motion_clip_ids`-keyed
  grouping, vs. the old shape-major positional-column-index assumption).
- `protomotions/components/motion_lib_pool.py` — one-line fix: invalidate the new
  clip-id-to-motion-ids cache on shard rotation too (same staleness reasoning already documented
  for the asset-id cache at that line), since `mimic_evaluator.py`'s rework below now depends on
  it generally, not just for the new pool.
- `protomotions/envs/motion_manager/motion_manager.py` — `get_state_dict`/`load_state_dict` gain
  a duck-typed `"global_clip_pool"` branch (checked before the existing `motion_file_name`-match
  legacy path, which is meaningless for a pool whose `motion_file` is a fixed synthetic per-rank
  string, not a real filename).
- `protomotions/agents/evaluators/mimic_evaluator.py` — clip-variant grouping
  (`_expand_to_clip_variants`, `_sample_one_shape_per_motion`) now uses real `motion_clip_ids`
  identity when available, falling back to the old positional-grid method otherwise (renamed
  `..._legacy_positional`). `_update_motion_sampling_weights` routes through the pool's
  `update_global_clip_weights`/`project_global_weights_to_resident_motion_weights` when present;
  the discount-factor math itself is unchanged either way.
- `protomotions/train_agent.py` — exempts `GlobalClipPoolConfig` from the `--motion-file`
  requirement; registers `GlobalClipPoolRebuildCallback` via `isinstance` check, mirroring the
  existing `StreamingMotionLibConfig` wiring.
- `examples/experiments/mimic/mlp_wide_fusion_stage2.py` — new `--global-clip-pool-*` CLI flags
  (source, size, cache dir/multiplier, rebuild-every, shuffle seed, exploration coefficient).
  `--r2-motion-source` (old shard streaming) kept, checked second, for in-flight runs/resumes.

**Design decisions made while turning the doc into code** (not spelled out in the original doc):
- `GlobalClipPool.motion_file` is a fixed string per rank (`f"global_clip_pool_rank{rank}"`), set
  once, never reassigned on rebuild — unlike `MotionLibPool`, which reassigns it every rotation.
  `Env.get_task_id()` derives the `env_<task_id>.ckpt` filename from this; if it changed every
  rebuild, resume could never find the right checkpoint file (residency here depends on the
  checkpoint's own contents, unlike shard rotation being a pure function of `current_epoch`).
- No new `on_load_checkpoint_before_env_load` hook: the resync (restore scoreboard, recompute
  top-K, materialize) happens synchronously inside `MotionManager.load_state_dict()` itself,
  since `BaseAgent.fit()` already unconditionally forces a full env reset on every call including
  resume.
- The positional→identity clip-grouping fix in `mimic_evaluator.py` applies whenever
  `motion_clip_ids` is present, not only when `GlobalClipPool` is in use — it's literally
  "Problem #2" the design doc names, and it breaks the moment *any* `MotionLib` holds a partial/
  ragged clip set, which the resident pool structurally is.

**Bug caught by testing, fixed before considering this done:** the first version of
`load_global_clip_weights_state_dict` re-ran the full rebuild-decision path on resume, which
silently re-incremented `rebuild_count`/`visit_counts` a second time for the same decision —
compounding by +1 on every resume, forever. Fixed by splitting `_rebuild_resident_pool`
(decision + bookkeeping) from `_materialize_resident_set` (pure data loading, no bookkeeping);
resume now calls only the latter. Caught via a real round-trip test against the actual R2
manifest, not just a code read-through.

**Verified (real R2 data, `r2:proto-data/hhi_stage2_per_clip/`, small K smoke tests), not yet
run on real training hardware:** clip-vocabulary determinism across separate instantiations,
weight-update math (fail → weight up, success → weight down, matching Stage 1's existing
formula exactly), scoreboard checkpoint round-trip (weights/visit_counts/rebuild_count exact;
resident set itself is *not* guaranteed bit-identical after a round trip — `_select_top_k`'s UCB
term uses `rebuild_count` as a time axis that legitimately shifts by one between "the decision
that produced the saved set" and "reconstructing from the saved counter," same as it would
between any two consecutive live rebuilds — only the scoreboard needs to round-trip exactly),
vocabulary-mismatch checkpoint guard, LRU disk-cache eviction bound, identity-based clip
grouping (both the new path and the legacy-positional fallback), callback cadence gating, and
all three experiment CLI dispatch branches (`--global-clip-pool-source` /
`--r2-motion-source` / `--motion-file`). Not yet run: an actual multi-GPU training job, or the
`nvidia-smi` VRAM dry-run the design doc calls for at real K=256 — both need real RunPod
hardware.

### 37.1 Real-run bug: `EMFILE`/"Too many open files" on the first cold-start rebuild (2026-07-24)

First actual RunPod launch (`hhi_wide_fusion_stage2_clippool`, `--ngpu 6`, `--global-clip-pool-size
256`) hit `[Errno 24] Too many open files` immediately, from `motion_lib_pool.FileDownloader`'s
`rclone copy` calls failing across many different clip files at once.

**Root cause:** `GlobalClipPool._materialize_resident_set`'s original implementation launched one
`FileDownloader` (i.e. one separate `rclone copy` OS process) *per missing clip*, all started
concurrently. At cold start every one of K=256 clips/rank is missing, so that's up to 256
simultaneous rclone processes per rank — and with `--ngpu 6` on one node, up to ~1536 across the
pod — each opening its own file descriptors (network sockets, config/credential files) on top of
the training process's own. That blows past the user's open-file ulimit (commonly 1024). The old
shard-streaming path (`MotionLibPool`) never hit this because it only ever runs 1-2 concurrent
downloads (current shard + one prefetch) — this was specific to the new per-clip pool's K-at-once
downloads.

**Fix:** `GlobalClipPool._download_missing_clips` now issues ONE `rclone copy` call per rebuild,
listing all missing clips via `--files-from=<tempfile>`, with rclone's own bounded *internal*
concurrency (`--transfers`/`--checkers`, new `GlobalClipPoolConfig.download_transfers` field,
default 8) — one OS process regardless of K, parallelism happens inside that process instead of
across processes. `motion_lib_pool.FileDownloader` is no longer used by `GlobalClipPool` (still
used, unchanged, by the old `MotionLibPool` shard path, which was never the problem).

**Re-verified end-to-end at the real production K=256** against the actual R2 data after the fix:
exactly one `rclone copy` process observed throughout (confirmed via `ps aux` while it ran), full
cold-start build completed in ~2m48s, `num_motions == 256*128 == 32768`, `256` distinct resident
clips — no FD exhaustion. Safe to relaunch the same command as before.

### 37.2 Real-run bug (unresolved, likely host-level): `futex_lock_pi` abort inside PhysX's CPU
dispatcher (2026-07-24)

After §37.1's fix, every launch of `hhi_wide_fusion_stage2_clippool` (`--num-envs 6144 --batch-size
24576 --ngpu 6`) still died with:
```
Skipping actor_optimizer state load (allow_partial_checkpoint_load=True): ...
The futex facility returned an unexpected error code.
[rank: N] Child process with PID ... terminated with code -6.
```
one rank, always right after checkpoint load / before the first epoch's rollout, on this specific
RunPod pod — no Python traceback, a bare glibc abort.

**Ruled out, in order tested (each with a real relaunch, not just reasoning):**
- `--use-wandb` — disabled, crash unchanged.
- `GlobalClipPool` forking during download — the manifest download was also made cache-skip (was
  unconditionally re-fetched every launch before this fix, confounding an earlier test), then
  relaunched on the *same* pod with a fully warm cache (clips + manifest) so `GlobalClipPool.
  __init__` made **zero** `subprocess`/`rclone` calls — crashed identically. Repeated on a second,
  brand-new pod (cold cache → real downloads happened → crash; then relaunched same pod, now warm
  → crash again). Conclusively rules out fork-after-CUDA-init in the download path.
- Leftover zombie GPU processes from a prior crashed run — `nvidia-smi` showed stale contexts
  the first time this was suspected, but the crash recurred on a genuinely fresh pod/GPUs too.
- Multi-GPU/NCCL scaling — `--ngpu 2` still crashed, but with a *different* symptom first
  (`PxgCudaDeviceMemoryAllocator fail to allocate memory 1073741824 bytes!! Result = 2`, then a
  segfault, code -11) — a distinct, well-known IsaacGym GPU-buffer allocation failure, not
  investigated further since a smaller repro below superseded it.
- Resource pressure / env count — a minimal smoke test (`--num-envs 32 --batch-size 128
  --global-clip-pool-size 8 --ngpu 1`, i.e. as small as the config allows) still hit the same futex
  abort — but notably *later* this time, during actual rollout collection ("Epoch N, collecting
  data...") rather than at checkpoint load, and as a single process (no Fabric multi-process
  launcher, since `--ngpu 1`).

**Root cause, found via `gdb -batch -ex run -ex "thread apply all bt full" --args python
protomotions/train_agent.py ...`** (needed because `core_pattern` on this pod pipes to `apport`,
which isn't running inside the container, so no core file was ever written — running under gdb
directly sidesteps that). The aborting thread's stack:
```
physx::Ext::CpuWorkerThread::execute()
  → physx::Cm::FanoutTask::removeReference()
    → pthread_mutex_lock() [PTHREAD_PRIO_INHERIT mutex]
      → futex_lock_pi()  -- kernel returned an error glibc's PI-futex path doesn't handle
        → futex_fatal_error() → abort()
```
This is entirely inside IsaacGym's closed-source PhysX plugin (`libcarb.gym.plugin.so`) — one of
PhysX's own CPU worker threads (`physx.num_threads`, default 4) doing routine task-graph reference
counting, using a **priority-inheritance (`FUTEX_LOCK_PI`) mutex**, a distinct and much less common
kernel/glibc code path than an ordinary futex wait. Nothing in ProtoMotions code, the download
logic, GPU count, or env count is anywhere on this stack — every earlier theory was chasing
correlation (crash timing lined up with checkpoint load / rank count / cache state) rather than
the actual cause.

**Working theory:** a host-kernel or container-runtime incompatibility with `FUTEX_LOCK_PI`
specific to this pod's underlying machine — PI futexes have known edge cases across kernel
versions/virtualization layers that ordinary futexes don't hit, and glibc has no fallback when the
kernel returns something the PI-futex path doesn't expect. This would explain why the crash is
100% reproducible on this pod regardless of every application-level variable tried, and why it's
never been reported before on whatever pod(s) this project has used historically.

**Confirmed host-level, not Stage-2-specific (2026-07-24):** relaunched with a completely
unrelated, much simpler script — plain `mlp.py` experiment, `smpl_mor_neutral` robot, a static
`--motion-file` (no `GlobalClipPool`/streaming, no `--checkpoint` warm-start), same
`hhi_20946_neutral` config that has trained successfully on this pod before — same `--ngpu 6`. Hit
the identical `futex` abort. This rules out Stage 2, the clip-pool code, and checkpoint loading
entirely: the fault is in this pod's PhysX/kernel interaction, triggered by essentially any
IsaacGym multi-GPU launch.

**Confirmed hardware-independent (2026-07-25):** retried on two freshly-provisioned pods, one
6xA40 and one 6xRTX A6000 — different GPU models, presumably different physical hosts. Both hit
the identical `futex_lock_pi` abort. This rules out "one degraded machine" and points at something
common across RunPod's fleet rather than specific hardware: a host kernel version rolled out
broadly, a container-runtime/nvidia-container-toolkit version, or a seccomp/cgroup policy change
that all their hosts picked up around the same time (PI-futex mutexes need `sched_setscheduler`
capability / nonzero `RLIMIT_RTPRIO` under the hood — if a security-hardening change restricted
that at the container level fleet-wide, every pod would hit this regardless of GPU).

**Status: unresolved, deferred — waiting on a different pod / RunPod support.** Decided not to
chase further in-application since this RunPod instance has been in use for a long time without
this issue appearing before, and it's now confirmed independent of anything in this codebase or
the specific GPU hardware:
1. First try forcing PhysX single-threaded to remove the CPU worker-thread pool (and this locking
   path) entirely: `--overrides simulator.sim.physx.num_threads=1`. If the crash stops, that
   confirms the PI-mutex theory; if it still crashes, the CPU dispatcher isn't optional at
   `num_threads=1` either and something deeper is going on. (Not yet tried as of 2026-07-25.)
2. Check the container's realtime-scheduling posture directly, since that's the mechanism PI
   futexes depend on: `ulimit -r` (RLIMIT_RTPRIO), `cat /proc/self/status | grep -i seccomp`,
   `cat /proc/self/status | grep Cap` (compare against `capsh --decode=<CapEff>` for `CAP_SYS_NICE`).
   A `0` rtprio limit or a seccomp filter blocking `sched_setscheduler` would explain the kernel
   returning something glibc's PI-futex path can't handle — and would point squarely at a
   RunPod-side container config change rather than hardware.
3. Test the same Docker image on non-RunPod infra (another cloud, or a local/on-prem GPU box) to
   separate "RunPod fleet-wide change" from "something baked into our own image" (e.g. a base
   image bump that pulled in a newer glibc/CUDA driver combo with this bug).
4. If a known-good older tag of the training image exists (from before 2026-07-24), try it
   unchanged on a new RunPod pod — isolates image-side regressions from host-side ones.
5. Report to RunPod support with the strengthened evidence (identical crash on two different GPU
   models same day) plus the `gdb` backtrace above (`futex_lock_pi` fatal error inside PhysX's CPU
   dispatcher) — this is a much stronger "your infra changed" signal than a single pod would be.

Debugging recipe worth keeping for next time (no core file needed, `core_pattern` was piped to a
non-running `apport`):
```bash
which gdb || apt-get update && apt-get install -y gdb
gdb -batch -ex "run" -ex "thread apply all bt full" --args python protomotions/train_agent.py \
  <same args as the failing launch> 2>&1 | tee /tmp/gdb_futex.log
grep -n "SIGABRT\|received signal\|futex_fatal\|__pthread_kill" /tmp/gdb_futex.log
```