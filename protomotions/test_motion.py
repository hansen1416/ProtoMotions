from protomotions.components.motion_lib import MotionLib, MotionLibConfig

motion_lib = MotionLib(
    MotionLibConfig(
        motion_file="/home/hlz/datasets/humos_proto_motionlib/humos_8.pt"
    ),
    device="cuda",
)

print(motion_lib.has_morphology_metadata())

asset_ids = [
    "male_0e26b88d",
    "female_0e26b88d",
]

motion_ids = motion_lib.sample_motions_for_asset_ids(
    asset_ids,
    deterministic=True,
)

print(motion_ids)
print(motion_lib.get_motion_asset_ids(motion_ids))
print(motion_lib.get_motion_betas(motion_ids).shape)