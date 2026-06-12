# Project Context: Morphology-Generalized Physics-Based Motion Imitation

**Current repo:** `https://github.com/hansen1416/ProtoMotions` (fork of `NVlabs/ProtoMotions`, branch `feature/hhi`)
**Previous repo:** `https://github.com/hansen1416/hhi` (ASE/RL-Games based, documented in `README.done1.md`, `README.done2.md`)
**Paper target:** ICRA 2027, deadline ~September 2026
**W&B:** project `hhi-protomotions`, entity `yugoamaryl`

---

## Research Goal

Train a **single physically simulated humanoid policy** that imitates AMASS/HUMOS motions across multiple SMPL body shapes. The key research question:

> Can one shared physics-based humanoid controller generalize across many body morphologies?

Formally: `(motion, gender, betas) → physically stable humanoid rollout`

**Core contribution:** morphology-conditioned physical motion imitation framework — not motion generation, not distillation, not text-to-motion. The central claim is that the policy learns **shape-adaptive physical control strategies** (different torques, contact timing, balance behavior) rather than blind reference tracking.

**Out of scope for this stage:** text-to-motion generation, Kimodo integration, diffusion models, distillation.

---

## Data

### Source

HUMOS generates per-beta motion predictions for AMASS sequences. Each HUMOS output `.pt` file contains the predicted motion for one AMASS clip across 128 beta variants (64 shapes × 2 genders — male and female only, no neutral). The 64 beta shapes are uniformly sampled in `[-3.0, 3.0]` per component.

- **Full dataset:** 22,459 AMASS motions → 20,951 valid (rest require external support, furniture, terrain, etc.)
- **Valid motion list:** `/home/hlz/repos/hhi/data-processing/valid_motions.txt`
- **Difficulty scoring:** `hhi/scripts/compute_difficulty_score.py`; sorted list at `data-processing/valid_ids_sorted_by_difficulty.txt`
- **Pilot training set:** 1024 motions, 5th–55th difficulty percentile (medium-difficulty, avoids trivial statics and unconverging hard motions)
- **Effective pilot library:** 1024 clips × 128 betas = **131,072 motions**

Difficulty score:
```
difficulty_score = 0.4 · norm(max_root_hvel) + 0.3 · norm(flight_ratio)
                 + 0.2 · norm(max_dof_vel)   + 0.1 · norm(kinetic_var)
```

### Data Pipeline

**Step 1 — HUMOS output → AMASS-style .npz**
```bash
python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/ \
    --out-root /home/hlz/datasets/humos_proto_interm/ \
    --skip-existing
```
Produces `humos_proto_interm/HUMOS/*.npz` and `humos_proto_interm/humos_131072.yaml`.
⚠️ HUMOS was trained on SMPLX, not SMPL — verify this script handles the conversion correctly (open TODO).

**Step 2 — .npz + yaml → MotionLib .pt**
```bash
python tools/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/humos_proto_interm/ \
    /home/hlz/datasets/humos_proto/ \
    --motion-config /home/hlz/datasets/humos_proto_interm/humos_131072.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cuda \
    --batch-size 8192
```
Produces `humos_proto/humos_131072_{chunk_idx:04d}.pt`. Each chunk ~3.6 GB. No merge step.

**Step 3 — Align frame 0 with ground**
```bash
python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/humos_proto/humos_131072_0000.pt \
    --asset-root protomotions/data/assets/mjcf/smpl_mor \
    --out-motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt
# batch version:
tools/run_frame0_offsets.sh
```

**Step 4 — Merge shards for training**
`tools/merge_motion_shards.py` → `/home/hlz/datasets/humos_proto/merged4/humos_slurmrank.pt`
Uploaded to Cloudflare R2: `r2:proto-data/merged4/`

### MotionLib .pt format

Each `.pt` file is a dict with:
```
gts:               [n_frames, 24, 3]    global translations (body joints)
grs:               [n_frames, 24, 4]    global rotations (xyzw quaternion)
gvs:               [n_frames, 24, 3]    global velocities
gavs:              [n_frames, 24, 3]    global angular velocities
dps:               [n_frames, 69]       DOF positions
dvs:               [n_frames, 69]       DOF velocities
lrs:               [n_frames, 24, 4]    local rotations
contacts:          [n_frames, 24]       binary foot contact flags
length_starts:     [n_motions]          frame index where each motion starts
motion_lengths:    [n_motions]
motion_dt:         [n_motions]
motion_num_frames: [n_motions]
motion_weights:    [n_motions]          curriculum weights, updated by evaluator
motion_files:      tuple[n_motions]
motion_betas:      [n_motions, 10]      SMPL beta parameters
motion_gender_ids: [n_motions]          -1=female, 1=male
motion_genders:    tuple[n_motions]     'male' or 'female'
motion_beta_keys:  tuple[n_motions]     8-char hash, e.g. 'bfd4619b'
motion_asset_ids:  tuple[n_motions]     e.g. 'male_bfd4619b'
motion_clip_ids:   tuple[n_motions]     e.g. '000005'
motion_npz_files:  tuple[n_motions]
```

Key design decision: each motion is tied to one specific `asset_id` (gender + beta_key). The motion library maps `asset_id → [motion_ids]` via `build_asset_id_to_motion_ids()`, so sampling is always shape-matched.

---

## Robot Assets

### SMPL humanoid templates

Generated from `SMPLSim/run.py` which produces `all_betas.pt` and one `.xml` per shape.
Metadata YAML generated by `scripts/generate_smpl_mor_asset_info.py`:
- `protomotions/data/assets/mjcf/smpl_mor/assets.yaml`
- `protomotions/data/assets/mjcf/smplx_mor/assets.yaml`

Robot config: `protomotions/robot_configs/smpl_mor.py` → `SmplMorRobotConfig`

The robot config points `asset_info_file` to the YAML, which lists all 128 shape XMLs. The simulator loads whichever subset is needed for the current motion file's unique `asset_ids`.

**Critical historical fix (from old `hhi` repo):** SMPL MJCF files were unstable until adding:
```python
compiler.attrib["coordinate"] = "local"
compiler.attrib["angle"]      = "radian"
```
in `smpl_local_robot.py:load_from_skeleton` after `self.tree = parse(...)`. Without this, joint ranges/axes were silently parsed in degrees → wrong joint limits and incorrect initial poses.

---

## Multi-Shape Simulator Architecture

### Asset loading (IsaacGym)

`simulator.py:_load_humanoid_assets`:
- `selected_asset_ids` is populated from the motion library's unique `asset_ids` before simulator init
- Code path: `env.py:initialize_simulator → simulator.py:_load_humanoid_assets`

### Per-env asset assignment

`IsaacGymSimulator._build_humanoid_asset_assignment`:
- Environments assigned assets by round-robin over the filtered asset set
- Produces `env_id_to_asset_idx`, `env_id_to_asset_name`, `env_morphology`, etc.
- The mapping is injected into the motion manager in `env.py:initialize_simulator`

### Morphology context

`env_morphology = [gender_id, betas / 3.0]` — constructed from XML metadata in
`simulator.py:_build_humanoid_asset_assignment:558-585`.
Exposed as `ctx.env_morphology [num_envs, 11]` in `env.py:_build_global_context`.

Code path: `simulator.py:558-585 → env.py:972 → component_factories.py:1265-1279 → obs/humanoid.py:351`

### Shape-matched motion sampling

`motion_lib.py:sample_motions_for_asset_ids` — only samples from the bucket matching the env's assigned `asset_id`. Called from `mimic_motion_manager.py:sample_motions`.

### Beta consistency check

At startup `env.py:326-350` cross-checks XML beta values vs motion file `motion_betas` per unique shape. Raises `RuntimeError` on mismatch. Enabled always, with extra debug output under `PROTOMOTIONS_DEBUG=1`.

---

## Observation Space

**Policy observation:** `[max_coords_obs, mimic_target_poses, previous_actions, morphology_obs]`

`morphology_obs` = `[gender_id, beta_1/3.0, ..., beta_10/3.0]` — 11-dim, shape `[num_envs, 11]`.

The `morphology_obs` component is registered as a separate observation key, not fused into the base obs. This allows different conditioning strategies (concat vs FiLM vs shape-embed) to handle it differently in the model.

---

## Policy Architecture

Three variants have been implemented, all drop-in replacements via experiment file:

### 1. MLP (baseline) — `examples/experiments/mimic/mlp.py`

`morphology_obs` is listed in `in_keys` and concatenated into the flat obs tensor alongside all other keys. The trunk receives one large concatenated vector. The network must learn all shape reasoning from raw 11-dim float values mixed in with 400–600+ dim state obs.

### 2. FiLM MLP — `examples/experiments/mimic/mlp_film.py`

`protomotions/agents/common/film_mlp.py` — `FiLMMLPConfig` + `FiLMMLPWithCond`

`morphology_obs` is a **conditioner key** (not in trunk input). A small conditioner MLP maps the 11-dim morphology → per-layer gamma/beta scales:

```
h_l = h_l * gamma_l(morphology) + beta_l(morphology)
```

Config: `cond_keys=["morphology_obs"]`, `cond_hidden_units=[64, 64]`, `beta_norm_scale=3.0`

**Why FiLM failed at scale** (`note/README.film-fail.md`):
- **Fanout bottleneck**: conditioner (64 units) must produce `2 × 6 × 1024 = 12,288` outputs (gamma + beta per unit per layer). A 64-unit net producing 12k values has diluted gradients — conditioner is hard to train.
- **Multiplicative instability**: trunk gradients at layer l are scaled by gamma_l. If gamma_l drifts from 1.0 early (likely with noisy conditioner), effective learning rate becomes shape-dependent and unstable. Worse with diverse motion datasets where the conditioner sees widely varying shapes per minibatch.
- Result: FiLM reached only reward 0.72 vs MLP's 0.84 at comparable steps, on 1024-motion dataset.

### 3. Shape Embed MLP — `examples/experiments/mimic/mlp_shape_embed.py`

`protomotions/agents/common/shape_embed_mlp.py` — `ShapeEmbedMLPConfig` + `ShapeEmbedMLP`

```
morphology_obs (11-dim)
    → normalize (betas / 3.0)
    → Linear(→ 64) → SiLU           ← 1 or 2 layers, controlled by cond_hidden_units
    → shape_embed (64-dim)
         │
[main obs keys] ──cat──→ normalized flat obs → standard MLP trunk → output
```

**Why better than FiLM:** conditioner output is 64-dim (not 12,288), additive coupling (concat), trunk gradients unaffected by conditioner quality.
**Why better than raw concat:** 11-dim raw betas mixed into 500+ dim obs; nonlinear projector lets the model learn a compact shape basis (body proportions, limb ratios) before the trunk. Analogous to a learned positional encoding.

Config knobs: `cond_hidden_units=[64]` (default) or `[64, 64]`, `cond_activation=silu`, `beta_norm_scale=3.0`.

### Ablation plan (architecture comparison)

| Config | Conditioning | Status |
|--------|-------------|--------|
| `mlp.py` | Raw concat of 11-dim morphology_obs | ~8k steps, reward 0.84 |
| `mlp_film.py` | FiLM, conditioner → 12,288 outputs | ~2.3k steps, reward 0.72 (failed) |
| `mlp_shape_embed.py` | Small encoder → 64-dim embed, concat | planned |

---

## Training

### Full training command template (RunPod, 4× A40)
```bash
python -u protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/<mlp|mlp_film|mlp_shape_embed>.py \
    --experiment-name <run_name> \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs <4096|8192> \
    --batch-size <16384|32768> \
    --ngpu 4 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group <run_name>
```

### Active / completed runs (as of 2026-06-12)

| Run name | Experiment | Envs/Batch | Step time | Status |
|----------|-----------|-----------|----------|--------|
| `hhi_1024_motion` | `mlp.py` | 4096/16384 | ~22s/step | ~8k steps, reward 0.84, +2 days |
| `hhi_film_1024_motion` | `mlp_film.py` | 8192/32768 | ~34s/step | ~2.3k steps, reward 0.72, failed |
| `hhi_se_1024_motion` | `mlp_shape_embed.py` | 4096/16384 | — | planned |

**Throughput context:**
- Non-FiLM: ~2.7M samples/hour; FiLM: ~3.5M samples/hour (2× envs compensates for slower step)
- Bottleneck: IsaacGym physics sim (30–50% GPU util). A40 ≈ A100 for this workload.
- Single-motion runs converged at reward 0.95+. MLP at 0.84 on 1024 diverse motions is strong — 0.95 is the wrong benchmark here.

### Local test commands (CPU / small-scale)
```bash
# MLP baseline
python protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 --batch-size 16

# ShapeEmbed (2-layer encoder override)
python protomotions/train_agent.py ... \
    --overrides agent.model.actor.mu_model.cond_hidden_units=[64,64] \
                agent.model.critic.cond_hidden_units=[64,64]
```

---

## Inference

```bash
python protomotions/inference_agent_mor.py \
    --checkpoint results/<run>/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --compact-spawn-spacing 1.5 \
    --num-envs 16

# Filter to specific shapes
python protomotions/inference_agent_mor.py ... \
    --gender-beta male:bfd4619b male:c1d2c0ef ... \
    --compact-spawn-spacing 1.2

# Large motion file (memory cap)
python protomotions/inference_agent_mor.py ... \
    --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0001_offset.pt \
    --max-motions 1024
```

`--max-motions`: loads full file to CPU, slices to first N motions, moves to GPU. Use ≥ 128 to have at least one motion per shape. Implemented in `MotionLibConfig.max_motions`.

---

## Evaluator

`protomotions/evaluate_hhi_faults.py` + `HHIFaultEvaluator`

Two purposes:
1. **Logging:** per-frame robot state → smoothness, jitter, success rates → W&B
2. **Curriculum update:** failed motions get higher sampling weight; succeeded motions get discounted (`0.999^200 ≈ 0.82`). Only feedback loop from evaluation into training.

Runs every 200 epochs. `max_eval_motions=2000` default (subsamples for large datasets).

```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/<run>/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --num-envs 8 \
    --output /home/hlz/Downloads/hhi_distance_report.csv
```

**Key evaluation metric:** not mean body distance alone — **per-shape variance and worst-case degradation**. Plot body distance vs beta L2 norm. FiLM/ShapeEmbed should degrade more gracefully on extreme shapes.

---

## Bugs Fixed

### IsaacGym crash at 4096+ envs (CUDA error 700 / segfault)

**Root cause:** 5 projectile cubes × 4096 envs × ~4 patches each ≈ 80k+ contact patches, hitting PhysX GPU `maxRigidPatchCount` (compile-time limit, not exposed via Python API).

**Fix:** `protomotions/simulator/isaacgym/config.py` — `IsaacGymSimulatorConfig` overrides `projectile` to `num_projectiles=1` by default (was 5). Also added `hide_spacing` to spread hidden cubes across different z-levels per pool index, and spread x/y by `env_id` to avoid broadphase contact at world origin. See `note/README.note.md` for full code snippets. PR #223 to upstream.

### Multi-GPU deadlock (30-min NCCL hang)

**Root cause:** NCCL P2P init (`cudaIpcGetMemHandle()`) conflicts with IsaacGym's CUDA context.

**Fix:** `NCCL_P2P_DISABLE=1` in `train_agent.py` via `os.environ.setdefault`. Forces NCCL to use PCIe copy-reduce path. Performance impact: negligible (<2ms/backward, <<1% of epoch time).

### GPU underutilization in headless multi-GPU training

**Root cause:** `simulator.py:99-100` forced `_graphics_device_id = 0` for all headless ranks. IsaacGym uses the graphics device for internal state tensor management even headless, so all 4 ranks funneled through GPU 0.

**Fix:** Remove the `if self.headless: self._graphics_device_id = 0` override. Each rank uses its own GPU as graphics device.

Symptom: GPU 0 at 97-100% SM, GPUs 1-3 at 0-57%. After fix all 4 should be equal.

---

## Experiment / Paper Plan

### Held-out evaluation (needs to be generated)

**Interpolation set:** 16–32 new random betas in `[-3, 3]` (different seed from training 128)
**Extrapolation set:** betas scaled to `[-5, 5]` range

Generate via HUMOS, same 1024 motion clips. Then run `evaluate_hhi_faults.py` on both checkpoints.

### Key experiments

| # | Experiment | Status |
|---|-----------|--------|
| 4.1 | Per-shape tracking on 128 training betas | Needs checkpoint |
| 4.2 | Generalization to held-out shapes (interp + extrap) | Needs held-out data + checkpoint |
| 4.3a | Shape-conditioned torque/energy analysis | Needs checkpoint |
| 4.3b | Stability / COM trajectory across shapes | Needs checkpoint |
| 4.3c | Contact quality across shapes | Needs checkpoint |
| 4.3d | Shape extremity vs physical failure correlation | Needs checkpoint |
| 4.3e | FiLM activation analysis (linear probe) | Needs FiLM checkpoint |
| 4.4 | MLP vs FiLM/ShapeEmbed ablation | In progress |
| 4.5 | Qualitative side-by-side visualization | Needs checkpoint |

**Strongest single experiment:** S2 embodiment encoding probe — extract FiLM/ShapeEmbed activations per shape, fit linear probe to predict physical body properties (height, mass, limb lengths from SMPL beta → URDF). If predictable with low error: network has built a physically meaningful internal representation purely from imitation learning.

---

## Infrastructure

### Docker image

`hansen1416/hhi-protomotions-isaacgym:v1` — contains IsaacGym, ProtoMotions, and all deps.

```bash
docker run -d --name proto --network=host --gpus=all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt:/mnt hansen1416/hhi-protomotions-isaacgym:v1 tail -f /dev/null
docker exec -it proto /bin/bash
```

### RunPod setup

```bash
cd /workspace && git clone -b feature/hhi https://github.com/hansen1416/ProtoMotions.git && cd ProtoMotions
# Download merged motion data from R2
rclone copy r2:proto-data/merged4/ /workspace/merged4/ \
    --transfers=4 --multi-thread-streams=16 --multi-thread-chunk-size=128M --progress
pip install -e .
wandb login <token>
```

R2 endpoint: `https://a17f581e2d142fd42fd7169cd4c48c8c.r2.cloudflarestorage.com`
R2 bucket: `r2:proto-data/merged4/`
Google Drive: `gdrive:humos_output` (raw HUMOS output, 778 GB), `gdrive:ckpt/` (checkpoints)

### Data paths

| Path | Contents |
|------|---------|
| `/home/hlz/datasets/humos_output/` | Raw HUMOS .pt output (22,459 clips × 128 variants) |
| `/home/hlz/datasets/humos_proto_interm/` | Intermediate AMASS-style .npz files |
| `/home/hlz/datasets/humos_proto/` | Processed MotionLib .pt shards |
| `/home/hlz/datasets/humos_proto/offset/` | Frame-0-aligned MotionLib .pt shards |
| `/home/hlz/datasets/humos_proto/humos_128_offset.pt` | Small test file (1 clip × 128 shapes) |
| `/home/hlz/datasets/humos_proto/merged4/humos_slurmrank.pt` | Training file (1024 clips × 128 shapes) |
| `protomotions/data/assets/mjcf/smpl_mor/*.xml` | SMPL shape MJCF templates |
| `protomotions/data/assets/mjcf/smpl_mor/assets.yaml` | Shape metadata YAML |

---

## Key Files (custom additions)

| File | Purpose |
|------|---------|
| `protomotions/robot_configs/smpl_mor.py` | SmplMorRobotConfig |
| `protomotions/components/motion_lib.py` | Morphology metadata + asset_id-matched sampling |
| `protomotions/simulator/isaacgym/simulator.py` | Multi-shape XML loading, per-env assignment, GPU fix |
| `protomotions/simulator/base_simulator/simulator.py` | Projectile fix + morphology base changes |
| `protomotions/envs/base_env/env.py` | ctx.env_morphology [num_envs, 11] |
| `protomotions/agents/common/film_mlp.py` | FiLMMLPConfig + FiLMMLPWithCond |
| `protomotions/agents/common/shape_embed_mlp.py` | ShapeEmbedMLPConfig + ShapeEmbedMLP |
| `examples/experiments/mimic/mlp.py` | Baseline: morphology_obs raw concat |
| `examples/experiments/mimic/mlp_film.py` | FiLM experiment config |
| `examples/experiments/mimic/mlp_shape_embed.py` | ShapeEmbed experiment config |
| `examples/motion_libs_visualizer_mor.py` | Multi-shape motion visualizer |
| `protomotions/inference_agent_mor.py` | Inference with --gender-beta + --max-motions |
| `protomotions/evaluate_hhi_faults.py` | Per-(gender,beta_key) evaluator, outputs CSV |
| `tools/export_humos_to_amass_npz.py` | HUMOS .pt → AMASS .npz |
| `tools/convert_amass_to_motionlib_with_morphology.py` | .npz → MotionLib .pt with morphology |
| `tools/compute_humos_frame0_offsets.py` | Align frame 0 with ground |
| `tools/merge_motion_shards.py` | Merge chunked .pt shards |
| `scripts/generate_smpl_mor_asset_info.py` | Generate assets.yaml |

---

## Open TODO

1. Verify `tools/export_humos_to_amass_npz.py` handles SMPLX → SMPL conversion correctly (HUMOS was trained on SMPLX)
2. Generate held-out betas (interpolation and extrapolation sets) via HUMOS — can do now without waiting for checkpoints
3. Run `evaluate_hhi_faults.py` on 128 training shapes when MLP checkpoint converges
4. Start `hhi_se_1024_motion` (ShapeEmbed) training run
5. Profile timing breakdown (physics step / obs / reward / policy) using `wandb` `perf/` metrics
