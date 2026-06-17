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

**Next to run on RunPod:** Full 192-clip evaluation of both tune and baseline checkpoints on `failed_clips.pt` and at least one full shard, to quantify forgetting vs improvement trade-off.