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


SimulatorConfig