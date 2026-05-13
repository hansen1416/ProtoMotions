from dataclasses import dataclass, field

from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.robot_configs.base import RobotAssetConfig

@dataclass
class HHISmplSingleRobotConfig(SmplRobotConfig):
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/male_0e26b88d_smpl_ig.xml",
            usd_asset_file_name=None,
            usd_bodies_root_prim_path=None,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )
