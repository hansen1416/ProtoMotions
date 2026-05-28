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

```bash
python scripts/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto_interm/
```

or for testing

```bash
python scripts/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto_interm_8/ --num 8
```

This command converts the HUMOS output file 000005.pt into AMASS-style .npz motion files under /home/hlz/datasets/humos_proto/.

## 2. generate the .pt files used in protomotions from the intermediate .npz and .yaml config. It will save a motion file in /home/hlz/datasets/humos_proto, eg. /home/hlz/datasets/humos_proto/humos_128.pt
```bash
python data/scripts/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/humos_proto_interm/ \
    /home/hlz/datasets/humos_proto/ \
    --motion-config /home/hlz/datasets/humos_proto_interm/humos_128.yaml \
    --humanoid-type smpl \
    --output-fps 30 \
    --device cuda \
    --force-remake
```
or for testing
```bash
python data/scripts/convert_amass_to_motionlib_with_morphology.py \
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
python scripts/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/humos_proto/humos_128.pt \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --out-motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --limit -1 \
    --overwrite
```
or for testing
```bash
python scripts/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/humos_proto/humos_8.pt \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --out-motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --limit -1 \
    --overwrite
```

## 4. visualize it
```bash
python examples/motion_libs_visualizer_mor.py \
    --motion_files ~/datasets/humos_proto/humos_8_offset.pt \
    --robot smpl_mor \
    --simulator isaacgym
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
  * Code path: `simulator.py:558-585 → env.py:972 → component_factories.py:1265-1279 → obs/humanoid.py:351`

* **Reset pose**

  * `motion_lib.get_motion_state` fetches reference position, rotation, and DOF state for the environment’s current `motion_id`.
  * Code path: `env.py:1082-1083 → env.py:1141-1164`

* **Betas XML ↔ motion file consistency**

  * Consistency is currently trusted through the shared `beta_key` hash.
  * There is no runtime check yet verifying that XML betas exactly match motion-file betas.
  * Code path: `simulator.py:442-449 → motion_lib.py:133-141`

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

## Training

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16

------

## inference motion

python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_single_motion_multi_shape/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --compact-spawn-spacing 1.5 \
    --num-envs 8


female:093098f0 female:09a0fcbd female:0e26b88d female:0f05fd5a female:10900e9a female:10c258c2 #
female:1658f5d3 female:1e5a1c90 female:2286da8c female:25247499 female:2e949ac0 female:30f6048e
female:312bf810 female:324b2d00 female:36baeba5 female:371b5e94 female:3b4a94c2 female:3c2cfe86
female:3faff413 female:42909c1b female:443d6b3e female:4dd55cac female:4de6c13b female:52d9e1de
female:546170ba female:653185e6 female:71fbbe41 female:724d4ad2 female:770f9e2c female:78613653
female:7b3c6576 female:7d706ded female:7e492dfc female:7f246a41 female:82266732 female:944474c9
female:97b473d4 female:9b4a6dda female:9d418743 female:a0720cb2 female:a2c978d0 female:a9143d09
female:abbf826b female:ad5728e1 female:b3fd6d6b female:b8e5fb4e female:b928198f female:bd3137aa
female:bfd4619b female:c1d2c0ef female:ca12d763 female:cf7925fd female:d1dc53df female:d495801e #
female:d4c80970 female:d6f908ec female:d9dbd795 female:da7b9ae1 female:df1b853d female:dfd2d9cf
female:e57f26a5 female:e5c9712a female:f0de7631 female:fb454239 male:093098f0 male:09a0fcbd #
male:0e26b88d male:0f05fd5a male:10900e9a male:10c258c2 male:1658f5d3 male:1e5a1c90
male:2286da8c male:25247499 male:2e949ac0 male:30f6048e male:312bf810 male:324b2d00
male:36baeba5 male:371b5e94 male:3b4a94c2 male:3c2cfe86 male:3faff413 male:42909c1b
male:443d6b3e male:4dd55cac male:4de6c13b male:52d9e1de male:546170ba male:653185e6
male:71fbbe41 male:724d4ad2 male:770f9e2c male:78613653 male:7b3c6576 male:7d706ded
male:7e492dfc male:7f246a41 male:82266732 male:944474c9 male:97b473d4 male:9b4a6dda
male:9d418743 male:a0720cb2 male:a2c978d0 male:a9143d09 male:abbf826b male:ad5728e1
male:b3fd6d6b male:b8e5fb4e male:b928198f male:bd3137aa male:bfd4619b male:c1d2c0ef
male:ca12d763 male:cf7925fd male:d1dc53df male:d495801e male:d4c80970 male:d6f908ec
male:d9dbd795 male:da7b9ae1 male:df1b853d male:dfd2d9cf male:e57f26a5 male:e5c9712a
male:f0de7631 male:fb454239

python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_single_motion_multi_shape/score_based.ckpt \
    --simulator isaacgym \
    --num-envs 6 \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --gender-beta male:9d418743 male:a0720cb2 male:a2c978d0 male:a9143d09 male:abbf826b male:ad5728e1 \
    --compact-spawn-spacing 1.2


<!-- there is actually only one is failing down, the others are fine, so the evaluation script need to change -->

------

## Evaluator

python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_single_motion_multi_shape/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --num-envs 8 \
    --output /home/hlz/Downloads/hhi_distance_report.csv