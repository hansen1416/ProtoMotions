"""Run HHI morphology fault detection from a trained checkpoint.

This script mirrors the inference setup path, but instead of visualizing forever
it evaluates the motion library batch by batch and writes a CSV report showing
which body-shape templates fail clearly.
"""


def create_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate HHI morphology faults batch by batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--simulator", type=str, required=True)
    parser.add_argument("--motion-file", type=str, default=None)
    parser.add_argument("--scenes-file", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format",
    )
    return parser


import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.agents.evaluators.hhi_fault_evaluator import HHIFaultEvaluator  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)

    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_configs_path.exists(), (
        f"Could not find resolved configs at {resolved_configs_path}"
    )

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' to '{args.simulator}'"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file

    from protomotions.utils.config_utils import (
        apply_config_overrides,
        parse_cli_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None
    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = getattr(env_config, "save_dir", None)
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=checkpoint.parent,
    )
    agent.setup()
    agent.load(args.checkpoint, load_env=False)

    output_path = (
        Path(args.output)
        if args.output is not None
        else checkpoint.parent / "hhi_fault_report.csv"
    )

    evaluator = HHIFaultEvaluator(
        agent=agent,
        fabric=fabric,
        config=agent_config.evaluator,
        output_path=output_path,
        progress_every=args.progress_every,
    )

    try:
        log_values, score = evaluator.evaluate()
        print("\n" + "=" * 60)
        print("HHI FAULT EVALUATION RESULTS")
        print("=" * 60)
        for key, value in sorted(log_values.items()):
            print(f"  {key}: {value:.6f}")
        if score is not None:
            print(f"  score: {score:.6f}")
        print("=" * 60 + "\n")
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()
