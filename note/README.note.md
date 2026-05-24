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
    --motion_files /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --robot smpl_mor \
    --simulator isaacgym
```

------

robot_config: RobotConfig in `protomotions/robot_configs/factory.py` defines all robot config, SMPL, SMPLX, etc


The `robot_config` typically passed to one of `SimulatorConfig` and `SimulatorClass` 

`SimulatorConfig` (protomotions/simulator/isaacgym/config.py) and 
`SimulatorClass` (protomotions/simulator/isaacgym/simulator.py) 
includes IsaacGym, IsaacLab, Genesis, Newton and MuJoCo (CPU-only)

--------

```text
MotionLib .pt
  -> each motion has motion_asset_id = "{gender}_{beta_key}"
  -> build asset_id -> compatible motion_ids

Visualizer
  -> collect unique asset_ids
  -> create one env per unique body shape
  -> pass requested_morphology_asset_ids to simulator
  -> sample env_motion_ids only from the matching asset_id group

Simulator
  -> load all morphology XMLs
  -> assign each env the requested XML asset
  -> assert visualizer env_asset_ids == simulator env_id_to_asset_name
```

Concretely:

1. `motion_lib.py` now stores morphology metadata: `motion_betas`, `motion_gender_ids`, `motion_genders`, `motion_beta_keys`, and `motion_asset_ids`, and it has `build_asset_id_to_motion_ids()` plus `sample_motions_for_asset_ids(...)`. That is the required motion-side matching logic. 

2. `simulator.py` accepts `morphology_asset_ids`, validates their length against `num_envs`, loads all XML assets from the morphology folder, and assigns each env using the requested asset id. That is the required multi-body-shape humanoid loading logic. 

3. `motion_libs_visualizer_mor.py` now creates one env per unique `asset_id`, samples one compatible motion per env through `sample_motions_for_asset_ids(self.env_asset_ids, ...)`, and passes `morphology_asset_ids` into the simulator. It also checks:

```python
assert self.simulator.env_id_to_asset_name == self.env_asset_ids
```

So the visualizer verifies that the simulator asset assignment matches the visualizer’s morphology assignment. 

4. `base.py`, `factory.py`, and `smpl_mor.py` support the new robot type: `RobotAssetConfig` can resolve a canonical XML from `asset_folder_name`, `factory.py` registers `"smpl_mor"`, and `SmplMorRobotConfig` points to `mjcf/smpl_mor/assets.yaml`.   

So the visualizer-side goal is satisfied:

```text
multiple body-shape humanoids loaded
each env has one morphology
each env only samples motions with the same gender/beta_key
```

```text
env_id -> env_asset_id -> compatible motion_ids -> sampled motion_id
```

```
python examples/motion_libs_visualizer_mor.py \
    --motion_files /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
    --robot smpl_mor \
    --simulator isaacgym

```

--------

## inference motion

python protomotions/inference_agent_mor.py \
    --checkpoint results/hhi_single_motion_multi_shape/score_based.ckpt \
    --simulator isaacgym \
    --motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --compact-spawn-spacing 1.5 \
    --num-envs 8


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