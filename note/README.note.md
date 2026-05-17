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

`scripts/export_humos_to_amass_npz.py` for prepare the morphology dataset

------

python examples/motion_libs_visualizer.py \
  --motion_files \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
  --robot smpl_mor \
  --simulator isaacgym

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

python examples/motion_libs_visualizer.py \
  --motion_files /home/hlz/datasets/humos_proto_motionlib/humos_1.pt \
  --robot hhi_smpl_single \
  --simulator isaacgym
```

----

# 1. Install rclone
curl https://rclone.org/install.sh | bash

# 2. Configure Google Drive (follow the prompts)
rclone config
# choose: n (new remote) → name it "gdrive" → choose Google Drive
# it will give you an auth URL → open in browser → paste the code back

# 3. Upload your checkpoint folder
rclone copy results.zip gdrive:ckpt --progress

----

python protomotions/inference_agent.py \
    --checkpoint results/hhi_single_male_0e26b88d_cloud_smoke/score_based.ckpt \
    --simulator isaacgym \
    --num-envs 1 \
    --motion-file /home/hlz/datasets/humos_proto_motionlib/humos_1.pt


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