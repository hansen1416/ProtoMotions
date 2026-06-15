# Generate SMPL humanoid robto templates

Use SMPLSim run.py to generate all_betas.pt and .xml files for sml and smplx.

use `scripts/generate_smpl_mor_asset_info.py` to geenrtae the asset information .yaml files:
protomotions/data/assets/mjcf/smpl_mor/assets.yaml
protomotions/data/assets/mjcf/smplx_mor/assets.yaml

They are used in `protomotions/robot_configs/smpl_mor.py`

```
asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            # morphology asset set
            asset_folder_name="mjcf/smpl_mor/",
            asset_info_file="mjcf/smpl_mor/assets.yaml",
            ...
        )
    )
```

All SMPL .xml templates are in protomotions/data/assets/mjcf/smpl_mor/*.xml

------

# Data Preprocessing

## 1. Convert HUMOS output to AMASS-style `.npz` files

**Batch mode — all 1024 clips, all 128 variants each:**
```bash
python tools/export_humos_to_amass_npz.py \
    --input-dir /home/hlz/datasets/humos_output/ \
    --out-root /home/hlz/datasets/humos_proto_interm/ \
    --skip-existing
```
Produces `humos_proto_interm/HUMOS/*.npz` and `humos_proto_interm/humos_131072.yaml` (1024 clips × 128 variants).
`--skip-existing` makes re-runs safe after interruption.

**Single-clip mode (testing / backward compat):**
```bash
python tools/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto_interm_single/
```

**Small test (8 variants from one clip):**
```bash
python tools/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto_interm_8/ --num 8
```

## 2. generate the .pt files used in protomotions from the intermediate .npz and .yaml config.

**Full dataset (131072 motions, batched to stay within RAM):**
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
Produces `humos_proto/humos_131072_{chunk_idx:04d}.pt` per chunk (e.g. `humos_131072_0000.pt`, `humos_131072_0001.pt`, …). Each chunk ~3.6 GB. No merge step — chunks are the final output.

**Single-clip / small test:**
```bash
python tools/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/humos_proto_interm_8/ \
    /home/hlz/datasets/humos_proto/ \
    --motion-config /home/hlz/datasets/humos_proto_interm_8/humos_8.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cuda \
    --force-remake
```

## 3. Align the 1st frame with ground. save the offseted file to a copy, eg. /home/hlz/datasets/humos_proto/humos_128_offset.pt
```bash
python tools/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/humos_proto/humos_131072_0000.pt \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --out-motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt \
    --limit -1 \
    --overwrite
```

run script to align all motion files
```bash
tools/run_frame0_offsets.sh
```

## merge the shards into n files:
```bash
tools/merge_motion_shards.py
```

## 4. visualize it
```bash
python examples/motion_libs_visualizer_mor.py \
    --motion_files ~/datasets/humos_proto/offset/humos_131072_0015_offset.pt \
    --robot smpl_mor \
    --simulator isaacgym \
    --start 360 --batch-size 16

```

## 5. data format (~/datasets/humos_proto/humos_8_offset.pt)

gts: tensor [n_frames, 24, 3]
grs: tensor [n_frames, 24, 4]
gvs: tensor [n_frames, 24, 3]
gavs: tensor [n_frames, 24, 3]
dvs: tensor [n_frames, 69]
dps: tensor [n_frames, 69]
length_starts: tensor [n_envs]
motion_lengths: tensor [n_envs]
motion_dt: tensor [n_envs]
motion_num_frames: tensor [n_envs]
motion_weights: tensor [n_envs]
contacts: tensor [n_frames, 24]
motion_files: tuple [n_envs]
lrs: tensor [n_frames, 24, 4]
motion_betas: tensor [n_envs, 10]
motion_gender_ids: tensor [n_envs] -1, 1
motion_genders: tuple [n_envs] 'male', 'female'
motion_beta_keys: tuple [n_envs] eg: '1e5a1c90'
motion_asset_ids: tuple [n_envs] eg: 'male_0e26b88d'
motion_clip_ids: tuple [n_envs] eg: '000005'
motion_npz_files: tuple [n_envs] *.npz files

------

## Morphology related change

examples/motion_libs_visualizer_mor.py
protomotions/robot_configs/smpl_mor.py
protomotions/components/motion_lib.py
protomotions/simulator/isaacgym/simulator.py
protomotions/envs/base_env/env.py
protomotions/simulator/base_simulator/simulator.py

robot_config: RobotConfig in `protomotions/robot_configs/factory.py` defines all robot config, SMPL, SMPLX, etc

The `robot_config` typically passed to one of `SimulatorConfig` and `SimulatorClass` 

`SimulatorConfig` (protomotions/simulator/isaacgym/config.py) and 
`SimulatorClass` (protomotions/simulator/isaacgym/simulator.py) 
includes IsaacGym, IsaacLab, Genesis, Newton and MuJoCo (CPU-only)


* **Training asset load**

  * `selected_asset_ids` is auto-populated from the motion library’s unique `asset_ids` before simulator initialization.
  * Code path: `env.py:initialize_simulator → simulator.py:_load_humanoid_assets`

* **Per-env asset assignment**

  * Environments are assigned assets by round-robin over the filtered asset set. in IsaacGymSimulator._build_humanoid_asset_assignment
  * This produces `env_id_to_asset_idx`, `env_id_to_asset_name`, `env_morphology`, etc.
  * The mapping is injected into the motion manager, also in `env.py:initialize_simulator` the of morphology block ater `_initialize_with_markers`. it's like env.py:initialize_simulator  -> simulator.py -> env.py:initialize_simulator

* **Motion sampling**

  * `sample_motions_for_asset_ids` only samples motions from the bucket matching the environment’s assigned `asset_id`.
  * Code path: `mimic_motion_manager.py:sample_motions → motion_lib.py:sample_motions_for_asset_ids` 
  * the asset id to motion_ids mapping `asset_id_to_motion_ids` is built in motion_lib.py:build_asset_id_to_motion_ids`

* **Morphology observation**

  * `env_morphology = [gender_id, betas / 3.0]` is constructed from XML metadata.
  * This morphology vector is passed into the observation pipeline each step.
  * Code path: `simulator.py:_build_humanoid_asset_assignment:558-585 → env.py:_build_global_context.EnvContext:972 → component_factories.py:morphology_obs_factory:1265-1279 → obs/humanoid.py:compute_morphology_obs:351`

* **Reset pose**

  * `motion_lib.get_motion_state` fetches reference position, rotation, and DOF state for the environment’s current `motion_id`.
  * Code path: `env.py:compute_ref_reset_state → env.py:reset`

* **Betas XML ↔ motion file consistency**

  * Consistency is currently trusted through the shared `beta_key` hash.
  * runtime in `env.py` `if os.environ.get("PROTOMOTIONS_DEBUG"):`.
  * Code path: `simulator.py._load_humanoid_assets`:444-456 (asset_id constructed and validated from YAML gender+beta_key) → motion_lib.py:136-140
  (motion-side field declarations: motion_betas, motion_gender_ids, motion_genders, motion_beta_keys, motion_asset_ids) →
  env.py:326-350 (cross-check: XML env_id_beta vs motion-file motion_betas per unique shape, raises RuntimeError on mismatch)

-----

## Expand the obs space

simulator._create_envs()
  → reads assets.yaml per env
  → self.env_morphology = torch.cat([gender_id, betas], dim=-1)  # [num_envs, 11]

          ↓  (built once at startup, static for the whole run)

_build_global_context()   ← called every step
  → ctx.env_morphology = self.simulator.env_morphology   # same tensor, no copy

          ↓

ComponentManager.execute_all(observation_components)
  → resolves EnvContext.env_morphology → gets the [num_envs, 11] tensor
  → calls compute_morphology_obs(morphology=tensor)
  → returns tensor unchanged

          ↓

_observation_buffer["morphology_obs"]   # [num_envs, 11]

          ↓

get_obs() → network reads it by key

------

## Sampling motions

**Full dataset:** 20,951 motions across 128 beta variants (64 shapes × 2 genders), listed in `/home/hlz/repos/hhi/data-processing/valid_motions.txt`. Each motion was predicted by HUMOS for all 128 beta inputs.

**Current stage:** pilot training on ~1,024 motions sampled from the 5th–55th difficulty percentile. Skips pure static poses (bottom) and motions too hard to converge within budget (top). Goal is a discriminative difficulty range where good architectures clearly outperform weak ones within a reasonable training window.

**Scale-up plan:**
- Pilot with ~1,024 motions first — faster to diagnose convergence and architecture differences (MLP vs FiLM)
- Once pilot converges, jump to all 20,951 motions — they fit easily in memory
- Scale `num_envs` to 8192 and `batch_size` to 32768 (~31 GB GPU, safe headroom on A40)

------

## Training

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_film.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_shape_embed.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_physics.py \
    --experiment-name hhi_physics_feat_1024 \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16
------

## inference motion

python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_1_motion_128_shape/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --compact-spawn-spacing 1.5 \
    --num-envs 16

python protomotions/inference_agent_mor.py \
      --checkpoint results/hhi_se_1024_motion/score_based.ckpt \
      --simulator isaacgym \
      --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0001_offset.pt \
      --compact-spawn-spacing 1.5 \
      --num-envs 16 \
      --max-motions 128


female:093098f0 female:09a0fcbd female:0e26b88d female:0f05fd5a female:10900e9a female:10c258c2 female:1658f5d3 female:1e5a1c90 female:2286da8c female:25247499 female:2e949ac0 female:30f6048e female:312bf810 female:324b2d00 female:36baeba5 female:371b5e94

female:3b4a94c2 female:3c2cfe86 female:3faff413 female:42909c1b female:443d6b3e female:4dd55cac female:4de6c13b female:52d9e1de female:546170ba female:653185e6 female:71fbbe41 female:724d4ad2 female:770f9e2c female:78613653 female:7b3c6576 female:7d706ded

female:7e492dfc female:7f246a41 female:82266732 female:944474c9 female:97b473d4 female:9b4a6dda female:9d418743 female:a0720cb2 female:a2c978d0 female:a9143d09 female:abbf826b female:ad5728e1 female:b3fd6d6b female:b8e5fb4e female:b928198f female:bd3137aa

female:bfd4619b female:c1d2c0ef female:ca12d763 female:cf7925fd female:d1dc53df female:d495801e female:d4c80970 female:d6f908ec female:d9dbd795 female:da7b9ae1 female:df1b853d female:dfd2d9cf female:e57f26a5 female:e5c9712a female:f0de7631 female:fb454239

male:093098f0 male:09a0fcbd male:0e26b88d male:0f05fd5a male:10900e9a male:10c258c2 male:1658f5d3 male:1e5a1c90 male:2286da8c male:25247499 male:2e949ac0 male:30f6048e male:312bf810 male:324b2d00 male:36baeba5 male:371b5e94

male:3b4a94c2 male:3c2cfe86 male:3faff413 male:42909c1b male:443d6b3e male:4dd55cac male:4de6c13b male:52d9e1de male:546170ba male:653185e6 male:71fbbe41 male:724d4ad2 male:770f9e2c male:78613653 male:7b3c6576 male:7d706ded

male:7e492dfc male:7f246a41 male:82266732 male:944474c9 male:97b473d4 male:9b4a6dda male:9d418743 male:a0720cb2 male:a2c978d0 male:a9143d09 male:abbf826b male:ad5728e1 male:b3fd6d6b male:b8e5fb4e male:b928198f male:bd3137aa

male:bfd4619b male:c1d2c0ef male:ca12d763 male:cf7925fd male:d1dc53df male:d495801e male:d4c80970 male:d6f908ec male:d9dbd795 male:da7b9ae1 male:df1b853d male:dfd2d9cf male:e57f26a5 male:e5c9712a male:f0de7631 male:fb454239

python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_single_motion_multi_shape/score_based.ckpt \
    --simulator isaacgym \
    --num-envs 16 \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --gender-beta male:bfd4619b male:c1d2c0ef male:ca12d763 male:cf7925fd male:d1dc53df male:d495801e male:d4c80970 male:d6f908ec male:d9dbd795 male:da7b9ae1 male:df1b853d male:dfd2d9cf male:e57f26a5 male:e5c9712a male:f0de7631 male:fb454239 \
    --compact-spawn-spacing 1.2

------

## Evaluator

python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_single_motion_multi_shape/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --num-envs 8 \
    --output /home/hlz/Downloads/hhi_distance_report.csv

------

## Fix: IsaacGym crash at 4096+ envs (CUDA error 700 / segfault)

### Root cause

`SimulatorConfig` defaults to **5 projectile cubes per env**. Each cube is a full
dynamic rigid body that PhysX must track contact patches for against the
`add_triangle_mesh` terrain (which has wildcard collidability).

Projectiles are pre-allocated dynamic rigid body cubes (pool of 5 per env) for perturbation testing — press J to throw one at the humanoid. They park underground at hide_z = -2.0 when idle and disappear after hide_delay = 2.0s.
  
Why they crashed PhysX GPU at 4096 envs:

The terrain is a triangle mesh, which in PhysX has wildcard broadphase coverage — it generates contact pair candidates with
every standalone rigid body every frame. Projectile cubes (non-articulated) go through a path governed by
maxRigidPatchCount:

5 cubes × 4096 envs × ~4 patches each ≈ 80,000+ patches  ≥  limit (~80K)
5 cubes × 1024 envs × ~4 patches each ≈ 20,000 patches   <  limit  ✓

That's exactly why 1024 worked and 4096 didn't. Idle cubes at z = -2.0 are still inside the mesh AABB, so they generate
patches every frame regardless.

maxRigidPatchCount is set at PhysX compile time and is not exposed through any Python API — max_gpu_contact_pairs,
default_buffer_size_multiplier, etc. all touch unrelated budgets.

(NVlabs/ProtoMotions PR #223)

#### 1. Cap IsaacGym to 1 projectile by default

`protomotions/simulator/isaacgym/config.py` — override the `projectile` field on
`IsaacGymSimulatorConfig` so IsaacGym gets 1 cube instead of the base default of 5:

```python
# Before: inherited SimulatorConfig.projectile (num_projectiles=5)

# After:
from protomotions.simulator.base_simulator.config import ProjectileConfig, ...

@dataclass
class IsaacGymSimulatorConfig(SimulatorConfig):
    ...
    projectile: ProjectileConfig = field(
        default_factory=lambda: ProjectileConfig(num_projectiles=1),
        metadata={"help": "Projectile pool config (IsaacGym defaults to 1 cube)."},
    )
```

`1 cube × 4096 envs = 4096 rigid bodies` — well under the ~80K patch budget.
To opt back into more cubes (e.g. for interactive viewer use):
```
--overrides simulator.projectile.num_projectiles=5
```

#### 2. Give each hidden cube a unique z slot

Without spacing, all hidden cubes for an env stack at the same `hide_z = -2.0`,
generating spurious contact patches between them. `hide_spacing` gives each pool
index its own z level.

`protomotions/simulator/base_simulator/config.py`:

```python
@dataclass
class ProjectileConfig:
    ...
    hide_z: float = -2.0
    hide_spacing: float = 4.0          # NEW: z-gap between hidden slots

    def hidden_z_for_index(self, projectile_index: int) -> float:
        """Return a hidden z-position that avoids projectile-projectile overlap."""
        return self.hide_z - self.hide_spacing * projectile_index
        # projectile 0 → z = -2.0
        # projectile 1 → z = -6.0
        # projectile 2 → z = -10.0  etc.
```

All spawn sites (IsaacGym `_build_projectile_actors`, IsaacLab scene, Newton, MuJoCo)
now call `hidden_z_for_index(i)` instead of the bare `hide_z`:

```python
# IsaacGym simulator.py — _build_projectile_actors
# Before:
start_pose.p = gymapi.Vec3(0.0, 0.0, self._proj_config.hide_z)

# After:
start_pose.p = gymapi.Vec3(
    env_id,
    env_id,
    self._proj_config.hidden_z_for_index(proj_idx),
)
```

The `env_id` x/y spread (see point 3) is also applied at spawn time so the initial
layout matches the runtime hide layout.

#### 3. Spread hidden cubes across x/y by env_id at runtime

When a cube is hidden (teleported underground), all backends previously placed it at
`(0, 0, hide_z)`. With thousands of envs, all hidden cubes pile up at the world
origin, regenerating contact patches every frame.

`_set_projectile_root_states` in each backend now detects hidden positions
(`z <= hide_z`) and rewrites x/y to `env_id`:

```python
# Added to IsaacGym, IsaacLab, Newton _set_projectile_root_states:
positions = positions.clone()
hidden_mask = positions[:, 2] <= self._proj_config.hide_z
if hidden_mask.any():
    hidden_env_offsets = env_ids[hidden_mask].to(positions.dtype)
    positions[hidden_mask, 0] = hidden_env_offsets   # x = env_id
    positions[hidden_mask, 1] = hidden_env_offsets   # y = env_id
```

This also applies inside `_update_projectiles` and `_hide_projectiles_for_envs` in
`base_simulator/simulator.py`, which use `proj_indices` to further separate cubes
within the same env:

```python
# _update_projectiles (expired cubes being hidden):
hide_pos[:, 2] = self._proj_config.hide_z
hide_pos[:, 2] -= self._proj_config.hide_spacing * proj_indices.to(hide_pos.dtype)

# _hide_projectiles_for_envs (env reset):
hide_pos[:, 2] = self._proj_config.hide_z
hide_pos[:, 2] -= self._proj_config.hide_spacing * proj_expanded.to(hide_pos.dtype)
```

#### 4. Early-return in `_throw_projectile` when num_projectiles == 0

If a backend sets `num_projectiles=0` to opt out entirely, pressing J (throw key)
would previously crash trying to index an empty pool.

`protomotions/simulator/base_simulator/simulator.py`:

```python
def _throw_projectile(self) -> None:
    cfg = self._proj_config
    if cfg.num_projectiles == 0:   # NEW: safe no-op
        return
    ...
```

------

## Multi GPU Deadlock

**Root cause:** NCCL P2P initialization conflicts with IsaacGym's CUDA context. When DDP broadcasts parameters, NCCL's `cudaIpcGetMemHandle()` call silently hangs (30-min timeout) because IsaacGym already holds the CUDA context.

**Fix:** `NCCL_P2P_DISABLE=1` in `train_agent.py` — forces NCCL to use PCIe copy-reduce path instead of NVLink P2P. Set via `os.environ.setdefault` so it can still be overridden.

**Performance impact:** Negligible. Gradient sync on ~1M params adds <2ms/backward pass — well under 1% of epoch time when sim dominates.

**Diagnostic (if it recurs):**
```bash
NCCL_DEBUG=WARN python -u protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 1024 --batch-size 16384 --ngpu 2
```
Last print before hang identifies the exact deadlock point.

------

## Evaluator (runs every 200 epochs)

**Two purposes:**
- **Logging only:** records per-frame robot state → computes smoothness, jitter, success rates → logs to wandb
- **Curriculum update (affects training):** failed motions get higher sampling weight, succeeded motions get discounted (`0.999^200 ≈ 0.82`). This is the only evaluator feedback loop into training.

**Evaluator changes and training impact:**
- CPU storage for MotionMetrics — no impact, purely memory
- `motion_lens.to(device)` fix — no impact, correctness fix for cross-device indexing
- `max_eval_motions=2000` subsampling — mild impact: only 2000 motions get weight updates per eval cycle

**`max_eval_motions=2000` tradeoff:**
- With 25k+ motions, each motion gets a weight update every ~2500 epochs (12.5 eval cycles) — slower curriculum feedback
- Options:
  - Keep 2000 — fine for large datasets, memory-safe
  - Set `null` via `--overrides agent.config.evaluator.max_eval_motions=null` to evaluate all — safe since CPU metric storage already avoids GPU OOM

------

## FiLM MLP Implementation

**`protomotions/agents/common/film_mlp.py`**
- `FiLMMLPConfig`: extends `NormObsBaseConfig`, adds `cond_keys` (default `["morphology_obs"]`), `cond_hidden_units` (`[64,64]`), `beta_norm_scale` (`3.0`)
- `FiLMMLPWithCond`: trunk is a `ModuleList` of blocks (not fused) so FiLM scale/shift can be applied between layers; conditioner MLP maps morphology → gamma/beta per layer; `_split_film_params` handles both `[B, D]` and `[T, N, D]` shapes

**`examples/experiments/mimic/mlp_film.py`**
- Identical env/reward/termination config to `mlp.py`
- Only change: actor and critic use `FiLMMLPConfig` with `morphology_obs` as `cond_keys` (not in `in_keys` / trunk concat)
- `_MAIN_OBS_KEYS = ["max_coords_obs", "mimic_target_poses", "previous_actions"]`

------

## Training Speed (current runs, 4× A40)

| Run | Envs/Batch | Step time | Samples/hour | Status (as of Jun 10) |
|---|---|---|---|---|
| Non-FiLM | 4096 / 16384 | ~22s/step | ~2.7M | ~8k steps, reward 0.84 |
| FiLM | 8192 / 32768 | ~34s/step | ~3.5M | ~2.3k steps, reward 0.72 |

**Key points:**
- Bottleneck is IsaacGym physics sim (30–50% GPU util), not NN compute — A40 ≈ A100 for this workload
- FiLM is slower per step but processes 2× data → 30% more samples/hour
- FiLM's lower reward despite more samples/hour confirms it's harder to optimize, not data-starved
- **Projected finish:** Non-FiLM ~+2 days, FiLM ~+5–7 days (~$380 total, worth it for generalization goal)

------

## Evaluation Plan

**Infrastructure ready:** `evaluate_hhi_faults.py` + `HHIFaultEvaluator` handles morphology-matched batching and outputs per-(gender, beta_key) CSV report.

**What's missing:** held-out motion files. All 128 training betas (64 shapes × 2 genders) are in the training set.

**Held-out eval sets to generate via HUMOS:**
1. **Interpolation** — ~16–32 new random betas sampled from `[-3, 3]` (different seed from training 128)
2. **Extrapolation** — betas in `[-5, 5]` range (scale existing betas by 5/3 is cleanest)

**Run evaluation on both checkpoints:**
```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion/last.ckpt \
    --simulator isaacgym \
    --motion-file /path/to/held_out_shapes.pt \
    --num-envs 64 \
    --output results/eval_mlp_heldout.csv

python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_film_1024_motion/last.ckpt \
    --simulator isaacgym \
    --motion-file /path/to/held_out_shapes.pt \
    --num-envs 64 \
    --output results/eval_film_heldout.csv
```

**Key metric:** not mean body distance — **per-shape variance and worst-case degradation**
- Plot body distance vs beta L2 norm — FiLM should degrade more gracefully on extreme shapes
- Compare worst 10–20 betas between MLP and FiLM — that's where FiLM's value shows up

**Note on ±5 betas:** SMPL motions at this range may be noisier — account for this when interpreting extrapolation results.

**Timeline:**
1. Generate held-out motion files now while training runs
2. Non-FiLM converges (~+2 days) → validate pipeline end-to-end
3. FiLM converges (~+7 days) → run full MLP vs FiLM comparison

------

## Evaluator Sampling Strategy — `eval_one_shape_per_motion`

### Problem

When training switched from 4 GPUs × 4096 envs to 1 GPU × 8192 envs, the periodic
evaluation loop became stuck with the GPU at 100% utilisation. The evaluator
iterates every motion in the library across all envs, so with `~131k` total motions
(1024 clips × 128 body shapes) the evaluation ran for hours and never completed.

### Commit history

| Commit | Change | Outcome |
|---|---|---|
| `42b48cf` | Added `max_eval_motions=2000` — cap the number of randomly sampled motions per eval run | Limits GPU memory and eval duration, but samples a random subset each time, so not all clips are covered |
| `a08ccd7` | Added `eval_one_per_shape` — sample 1 clip per body shape (128 shapes → 128 eval motions) | Wrong direction: covers all shapes but misses most clips |
| `8e74268` | Replaced with `eval_one_shape_per_motion` — sample 1 shape per clip (N clips → N eval motions) | Correct: covers every unique clip, each paired with one randomly drawn body shape |

### Final approach — `eval_one_shape_per_motion=True` (default)

**Goal:** evaluate every unique motion clip, but pair each clip with exactly one
randomly sampled gender-beta body shape per eval run. This gives full clip coverage
while reducing the total evaluation set from `num_clips × num_shapes` to `num_clips`.

**Code changes**

`protomotions/agents/evaluators/config.py` — two fields on `MimicEvaluatorConfig`:

```python
max_eval_motions: Optional[int] = field(
    default=2000,
    metadata={
        "help": (
            "Cap the number of motions evaluated per eval run. "
            "None = evaluate all motions. "
            "Ignored when eval_one_shape_per_motion=True. "
            ...
        ),
    },
)
eval_one_shape_per_motion: bool = field(
    default=True,
    metadata={
        "help": (
            "When True and the motion library has morphology metadata, "
            "cover every unique motion clip but pair each with one randomly "
            "sampled gender-beta shape per evaluation run. "
            "Overrides max_eval_motions."
        )
    },
)
```

`protomotions/agents/evaluators/mimic_evaluator.py`:

```python
def _sample_one_shape_per_motion(self) -> torch.Tensor:
    asset_to_motion_ids = self.motion_lib.build_asset_id_to_motion_ids()
    shape_lists = list(asset_to_motion_ids.values())  # each: [num_clips]

    num_shapes = len(shape_lists)
    num_clips = shape_lists[0].shape[0]

    # [num_shapes, num_clips] — row k = all motion IDs for shape k
    all_ids = torch.stack(shape_lists, dim=0)

    # For each clip position, draw one random shape index
    shape_picks = torch.randint(num_shapes, (num_clips,), device=all_ids.device)
    clip_idx = torch.arange(num_clips, device=all_ids.device)

    selected = all_ids[shape_picks, clip_idx]
    return selected.sort().values
```

`initialize_eval()` branches on the new flag before falling back to `max_eval_motions`:

```python
if self.config.eval_one_shape_per_motion and self.motion_lib.has_morphology_metadata():
    self._eval_motion_subset = self._sample_one_shape_per_motion()
else:
    max_eval = self.config.max_eval_motions
    if max_eval is not None and total_motions > max_eval:
        perm = torch.randperm(total_motions, device=self.device)[:max_eval].sort().values
        self._eval_motion_subset = perm
    else:
        self._eval_motion_subset = None
```

**Key assumption:** `build_asset_id_to_motion_ids()` accumulates motion IDs in the
order they appear in the flat `motion_asset_ids` tuple. For HUMOS (all 1024 clips
retargeted to all 128 body shapes in a consistent order), position `i` in every
shape's list refers to the same underlying motion clip. This makes `all_ids[k, i]`
the global motion ID for "clip i under shape k", so the column-wise random selection
correctly picks one shape per clip.

**Result:** with 1024 clips the eval set is 1024 motions — well within the 8192-env
budget — and each eval run samples a fresh random shape assignment, giving uniform
coverage of the shape space over many eval cycles. Because the eval set is bounded
to `num_clips`, MotionMetrics tensors are small enough to keep on GPU (no CPU
offload needed).

------

## Clip-level curriculum propagation

### Finding

All three pilot runs (mlp, shape_embed, physics_feat) showed virtually identical
convergence curves over 1d18h of training. Analysis of the per-epoch failure files
revealed that only **3 global motion IDs** were commonly failed across all three runs
at epoch 6800 — out of ~800 failures per run. This exposed two issues:

1. **Cross-run comparison noise**: `eval_one_shape_per_motion` draws a different random
   shape per clip each eval cycle, so the failure set is largely determined by which
   shape was sampled, not architecture. Per-epoch failure files from different runs are
   not directly comparable.

2. **Weight updates don't cross shape boundaries**: Even within a single run, when clip
   X fails under `shape_A`, only motion ID `(clip_X, shape_A)` gets `weight = 1.0`.
   The other 127 shape variants of clip X keep their old weights. During training,
   each env is assigned a fixed shape; `shape_B` envs therefore never increased their
   sampling of clip X, even though it's equally hard for them.

This is the binding constraint on curriculum quality, not the 13% clip miss rate of
the old `max_eval_motions=2000` sampling.

### Fix — clip-level weight propagation

**Key insight**: Motion difficulty is intrinsic to the clip (balance demands, speed,
contact pattern), not the body shape. A clip that fails under one gender-beta is
expected to be hard for all 128 shapes. Therefore, curriculum updates should be
applied to **all shape variants** of a failed (or succeeded) clip simultaneously.

**Implementation** in `protomotions/agents/evaluators/mimic_evaluator.py`:

```python
def _build_clip_expansion_index(self):
    # Builds [num_shapes, num_clips] matrix and reverse global_id → clip_col map.
    # Cached after first call. Relies on positional-order assumption (same as
    # _sample_one_shape_per_motion): all_ids[k, i] = global ID for (clip i, shape k).

def _expand_to_clip_variants(self, motion_ids: torch.Tensor) -> torch.Tensor:
    # Given a set of global motion IDs, returns IDs for ALL shape variants
    # of those clips via unique clip-column lookup in all_ids.
```

In `_update_motion_sampling_weights`, after mapping local → global IDs:
```python
# Log unexpanded failures (what actually failed in this eval cycle)
self._save_failed_motions(global_failed.tolist(), self.agent.current_epoch)

# Then expand to all shape variants before updating weights
if self.motion_lib.has_morphology_metadata():
    global_failed = self._expand_to_clip_variants(global_failed)
    global_success = self._expand_to_clip_variants(global_success)
```

**Note**: `_save_failed_motions` is called with the pre-expansion IDs (the specific
`(clip, shape)` pairs that actually failed), so the `failed_motions/` log files
remain interpretable. The weight update uses the expanded set.

**Effect**: After one eval cycle where clip X fails under `shape_A`:
- Before: only `(clip_X, shape_A)` gets `weight = 1.0`; the other 127 shape-envs
  still sample clip X at its old (possibly low) weight
- After: all 128 shape variants of clip X get `weight = 1.0`; every env immediately
  increases its sampling of the hard clip

**Condition**: only activated when `motion_lib.has_morphology_metadata()` is True,
so it's a no-op for single-morphology datasets.

------

## `max_motions` — inference with large motion files

**Problem:** `humos_131072_0000_offset.pt` is 3.6 GB. Loading it during inference with 16 envs exceeds GPU memory.

**Fix:** Added `max_motions: Optional[int]` to `MotionLibConfig` (default `None`). When set, `load_from_file` loads the full file to CPU, slices to the first N motions, then moves to GPU. Training is unaffected when `max_motions` is not set (the only behavior change for the unset case is CPU→GPU via `.to()` instead of `map_location=device`, which is functionally identical).

**Usage:**
```bash
python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_1024_motion/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/offset/humos_131072_0000_offset.pt \
    --compact-spawn-spacing 1.5 \
    --num-envs 16 \
    --max-motions 1024
```

**Sizing:** with 128 morphology shapes, set `--max-motions` ≥ 128 so morphology-consistent sampling has at least one motion per shape. 1024 is comfortable (~30 MB GPU vs 3.6 GB).