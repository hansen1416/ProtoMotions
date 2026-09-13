"""Record one clip across body shapes in Isaac Gym using a frozen policy.

Outputs: traces.pt (CPU tensors), summary.csv (one row per shape), metadata.json.
The DOF sensor measures total generalized joint force, not validated isolated
actuator torque. Sensor power/work and effort-limit comparisons are diagnostics.
Run --help without importing Isaac Gym or torch.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def select_variants(clip_ids, asset_ids, max_shapes):
    """Select evenly spaced asset IDs, keeping one variant per shape."""
    if not clip_ids or len(set(clip_ids)) != 1:
        raise ValueError("Supply exactly one clip, with one variant per body shape.")
    if len(clip_ids) != len(asset_ids) or len(set(asset_ids)) != len(asset_ids):
        raise ValueError("Clip/asset metadata must align and contain unique shapes.")
    ordered = sorted(range(len(asset_ids)), key=lambda i: asset_ids[i])
    if max_shapes == 0 or max_shapes >= len(ordered):
        return ordered
    if max_shapes < 1:
        raise ValueError("--max-shapes must be nonnegative (0 selects all).")
    if max_shapes == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[round(i * (len(ordered) - 1) / (max_shapes - 1))]
            for i in range(max_shapes)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--motion-file", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-shapes", type=int, default=8, help="0 selects all shapes.")
    result.add_argument("--max-steps", type=int, default=600)
    result.add_argument("--seed", type=int, default=42)
    return result


def main():
    args = parser().parse_args()
    if args.max_steps < 1 or args.max_shapes < 0:
        raise ValueError("Use positive --max-steps and nonnegative --max-shapes.")
    checkpoint = args.checkpoint.resolve()
    motion_path = args.motion_file.resolve()
    config_path = checkpoint.parent / "resolved_configs_inference.pt"
    for path in (checkpoint, motion_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new, empty --output-dir to preserve previous results.")

    # Must precede all imports that can transitively import torch.
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    import_simulator_before_torch("isaacgym")
    import numpy as np
    import torch
    import yaml
    from lightning.fabric import Fabric
    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.robot_configs.base import ControlType
    from protomotions.utils.component_builder import build_all_components
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    Fabric.seed_everything(args.seed)
    configs = torch.load(config_path, map_location="cpu", weights_only=False)
    robot = configs["robot"]
    sim_config = configs["simulator"]
    env_config = configs["env"]
    if "isaacgym" not in sim_config._target_:
        raise ValueError("This pilot requires an Isaac Gym checkpoint.")
    apply_backward_compatibility_fixes(robot, sim_config, env_config)
    data = torch.load(motion_path, map_location="cpu", weights_only=False)
    selected = select_variants(
        data["motion_clip_ids"], data["motion_asset_ids"], args.max_shapes
    )
    clip_id = data["motion_clip_ids"][0]
    asset_ids = [data["motion_asset_ids"][i] for i in selected]
    betas = data["motion_betas"][selected].clone()
    motion_gender_ids = data["motion_gender_ids"][selected].clone()
    asset_dir = Path(robot.asset.asset_root) / robot.asset.asset_folder_name
    asset_info_path = Path(robot.asset.asset_root) / robot.asset.asset_info_file
    with open(asset_info_path) as stream:
        asset_info = {row["asset_id"]: row for row in yaml.safe_load(stream)}
    for i, asset_id in enumerate(asset_ids):
        if asset_id not in asset_info:
            raise ValueError("Missing matching asset metadata: " + asset_id)
        if not np.allclose(betas[i].numpy(), asset_info[asset_id]["betas"], atol=1e-5, rtol=0):
            raise ValueError("Motion/asset betas disagree: " + asset_id)
        if not (asset_dir / (asset_id + "_smpl.xml")).is_file():
            raise FileNotFoundError(asset_dir / (asset_id + "_smpl.xml"))
    del data

    # Do not instantiate or restore the checkpoint's streaming training pool.
    motion_config = MotionLibConfig(
        motion_file=str(motion_path),
        get_motion_state_use_blend=configs["motion_lib"].get_motion_state_use_blend,
    )
    robot.asset.selected_asset_ids = asset_ids
    sim_config.num_envs = len(selected)
    sim_config.headless = True
    # Retain termination flags for diagnosis; this recorder never resets mid-rollout.
    fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric.launch()
    terrain_config, sim_config = convert_friction_for_simulator(configs["terrain"], sim_config)
    components = build_all_components(
        terrain_config=terrain_config, scene_lib_config=configs["scene_lib"],
        motion_lib_config=motion_config, simulator_config=sim_config,
        robot_config=robot, device=fabric.device, morphology_asset_ids=asset_ids,
    )
    env = get_class(env_config._target_)(
        config=env_config, robot_config=robot, device=fabric.device, **components
    )
    sim = env.simulator
    if list(sim.env_id_to_asset_name) != asset_ids:
        raise RuntimeError("Simulator environment-to-shape assignment does not match.")
    if sim.control_type != ControlType.BUILT_IN_PD:
        raise ValueError("This pilot currently requires built-in position PD control.")
    agent = get_class(configs["agent"]._target_)(
        config=configs["agent"], env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False)
    agent.eval()

    # Read actual actor properties after simulator mass scaling, in common order.
    dof_order = sim.data_conversion.dof_convert_to_common.cpu().numpy()
    body_order = sim.data_conversion.body_convert_to_common.cpu().numpy()
    properties = {key: [] for key in ("effort", "stiffness", "damping", "armature", "friction")}
    masses = []
    for handle_env, actor in zip(sim._envs, sim._humanoid_handles):
        props = sim._gym.get_actor_dof_properties(handle_env, actor)
        for key in properties:
            properties[key].append(torch.tensor(props[key][dof_order].copy()))
        bodies = sim._gym.get_actor_rigid_body_properties(handle_env, actor)
        masses.append(torch.tensor([bodies[j].mass for j in body_order]))
    properties = {key: torch.stack(value) for key, value in properties.items()}
    masses = torch.stack(masses)
    if not torch.isfinite(masses).all() or not (masses > 0).all():
        raise ValueError("Invalid simulator body masses.")
    if not torch.isfinite(properties["effort"]).all() or not (properties["effort"] > 0).all():
        raise ValueError("Nonpositive DOF effort limits.")
    motion_ids = torch.tensor(selected, device=fabric.device)
    env_ids = torch.arange(len(selected), device=fabric.device)
    lengths = env.motion_lib.get_motion_length(motion_ids).detach().cpu()
    steps = min(args.max_steps, int(math.floor(float(lengths.max()) / env.dt + 1e-4)))
    if steps < 1:
        raise ValueError("Motion has no complete control intervals.")

    trace_lists = {}
    def append(name, tensor):
        trace_lists.setdefault(name, []).append(tensor.detach().cpu().clone())

    # Observe the actual target submitted after simulator-side filtering/noise.
    # Keep the final physics-step target within each control interval.
    submitted_targets = {}
    original_apply_targets = sim._apply_simulator_pd_targets
    def record_targets(targets):
        submitted_targets["last"] = targets[:, sim.data_conversion.dof_convert_to_common].clone()
        original_apply_targets(targets)
    sim._apply_simulator_pd_targets = record_targets

    def capture_state():
        # Context reference is already aligned to each environment's spawn offset.
        context = env.context
        state = sim.get_robot_state()
        reference = context.mimic.ref_state
        predicted = context.current.rigid_body_pos
        target = reference.rigid_body_pos
        error = (predicted - target).norm(dim=-1)
        append("time_s", env.motion_manager.motion_times)
        append("dof_pos_rad", state.dof_pos)
        append("dof_vel_rad_s", state.dof_vel)
        append("reference_dof_pos_rad", reference.dof_pos)
        append("reference_dof_vel_rad_s", reference.dof_vel)
        append("body_pos_world_m", predicted)
        append("reference_body_pos_world_m", target)
        append("body_position_error_m", error)
        if not all(torch.isfinite(x).all() for x in
                   (state.dof_pos, state.dof_vel, reference.dof_pos, reference.dof_vel)):
            raise RuntimeError("Nonfinite joint state or reference.")
        return error.mean(dim=-1)

    with torch.no_grad():
        env.motion_manager.motion_ids[:] = motion_ids
        env.motion_manager.motion_times[:] = 0
        obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
        initial_error = capture_state()
        print("Initial mean body error, worst shape: %.6f m" % initial_error.max(), flush=True)
        if not torch.isfinite(initial_error).all() or initial_error.max() > 0.05:
            raise RuntimeError("Initial alignment check failed; investigate before recording.")
        ema = configs["agent"].evaluator.eval_action_ema_alpha
        previous = None
        for step in range(steps):
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            model_output = agent.model(obs_td)
            actions = model_output.get("mean_action")
            if actions is None:
                raise RuntimeError("Policy did not return deterministic mean_action.")
            if not torch.isfinite(actions).all():
                raise RuntimeError("Nonfinite policy action.")
            if ema is not None:
                previous = actions.clone() if previous is None else previous
                actions = ema * actions + (1 - ema) * previous
                previous = actions.clone()
            obs, _, done, terminated, _ = env.step(actions)
            if not torch.equal(env.motion_manager.motion_ids, motion_ids):
                raise RuntimeError("A motion was resampled during the recorded rollout.")
            error = capture_state()
            sensor = sim.get_dof_forces().dof_forces
            contact = sim.get_bodies_contact_buf().rigid_body_contact_forces
            if not all(torch.isfinite(x).all() for x in (sensor, contact, error)):
                raise RuntimeError("Nonfinite simulator signal; output is not a valid pilot.")
            append("dof_force_sensor_Nm", sensor)
            append("body_contact_force_world_N", contact)
            append("raw_action", actions)
            append("processed_action", env._current_processed_action)
            append("pd_target_rad", submitted_targets["last"])
            append("done_flag", done)
            append("terminated_flag", terminated)
            if (step + 1) % 50 == 0 or step + 1 == steps:
                print("Recorded %d/%d steps; mean tracking error %.4f m" %
                      (step + 1, steps, error.mean()), flush=True)

    traces = {key: torch.stack(value) for key, value in trace_lists.items()}
    del trace_lists
    times = traces["time_s"]
    expected = torch.arange(steps + 1)[:, None] * env.dt
    if not torch.allclose(times, expected.expand_as(times), atol=1e-4, rtol=1e-5):
        raise RuntimeError("Motion clock advanced unexpectedly; refusing misaligned traces.")
    valid = times[1:] <= lengths[None, :] + 1e-5
    traces["valid_step"] = valid
    # Endpoint sensor x velocity: a control-rate proxy, not substep motor work.
    power = traces["dof_force_sensor_Nm"] * traces["dof_vel_rad_s"][1:]
    traces["sensor_power_proxy_W"] = power
    traces["effort_limit_Nm"] = properties["effort"]
    traces["pd_stiffness"] = properties["stiffness"]
    traces["pd_damping"] = properties["damping"]
    traces["armature"] = properties["armature"]
    traces["joint_friction"] = properties["friction"]
    traces["body_mass_kg"] = masses
    traces["motion_betas"] = betas
    traces["motion_gender_ids"] = motion_gender_ids
    traces["motion_ids"] = torch.tensor(selected)
    traces["initial_body_position_error_m"] = traces["body_position_error_m"][0]
    rows = []
    for i, asset_id in enumerate(asset_ids):
        mask = valid[:, i]
        if not mask.any():
            raise RuntimeError("No valid samples for " + asset_id)
        torque = traces["dof_force_sensor_Nm"][mask, i]
        tracking = traces["body_position_error_m"][1:][mask, i].mean(dim=-1)
        work = float(power[mask, i].abs().sum() * env.dt)
        mass = float(masses[i].sum())
        rows.append(dict(
            clip_id=clip_id, asset_id=asset_id, motion_id=selected[i], mass_kg=mass,
            recorded_steps=int(mask.sum()), duration_s=float(mask.sum() * env.dt),
            full_horizon=bool(float(lengths[i]) <= steps * env.dt + 1e-4),
            mean_body_error_m=float(tracking.mean()), max_mean_body_error_m=float(tracking.max()),
            tracking_pass_0p5m=bool(tracking.max() <= 0.5),
            sensor_rms_Nm=float(torque.square().mean().sqrt()),
            sensor_peak_abs_Nm=float(torque.abs().max()),
            sensor_limit_exceedance_fraction=float((torque.abs() >= 0.99 * properties["effort"][i]).float().mean()),
            sensor_abs_work_proxy_J=work, sensor_abs_work_proxy_J_per_kg=work / mass,
        ))
    metadata = dict(
        schema_version=1, clip_id=clip_id, asset_ids=asset_ids,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        checkpoint_epoch=int(agent.current_epoch), config_sha256=sha256(config_path),
        motion_file=str(motion_path), motion_sha256=sha256(motion_path),
        asset_xml_sha256={a: sha256(asset_dir / (a + "_smpl.xml")) for a in asset_ids},
        asset_info_sha256=sha256(asset_info_path),
        recorder_sha256=sha256(Path(__file__)), seed=args.seed, dt_s=env.dt,
        body_names=list(robot.kinematic_info.body_names),
        dof_names=list(robot.kinematic_info.dof_names),
        control_type=str(sim.control_type), eval_action_ema_alpha=ema,
        signal_notes=[
            "All tensors use common body/DOF order; world coordinates are Z-up.",
            "State samples have T+1 frames (including reset); force/action samples have T.",
            "Force samples follow each control step; power uses matching endpoint velocity.",
            "pd_target_rad is the last target submitted to PhysX within the control interval.",
            "DOF sensor is total generalized joint force; isolated actuator torque is unverified.",
            "Power/work and sensor/effort comparisons are proxies, not validated motor energy or saturation.",
            "Contact tensor is net body contact force, not isolated ground reaction force.",
            "Sampling is at control rate; simulation-substep peaks and work are not resolved.",
            "Use valid_step to exclude padded time beyond individual motion horizons.",
            "One clip across shapes may have shape-dependent reference kinematics.",
        ],
    )
    traces["metadata"] = metadata
    temporary = output / "traces.pt.tmp"
    torch.save(traces, temporary)
    temporary.replace(output / "traces.pt")
    with open(output / "summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "metadata.json", "w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print("Saved physics pilot to %s (%d shapes)." % (output, len(rows)), flush=True)
    if hasattr(sim, "shutdown"):
        sim.shutdown()


if __name__ == "__main__":
    main()

