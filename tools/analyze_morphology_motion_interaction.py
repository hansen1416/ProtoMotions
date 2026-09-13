# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Physical analysis (paper section 8): does the effect of morphology on control demand
depend on the motion being performed? See note/README.physical-analysis-plan.md.

Replays the trained checkpoint, deterministically, on every (clip, shape) pair in the
matched `small150_128shape_refined.pt` corpus (150 clips x 128 shapes = 19,200 motions),
and logs per-(clip, shape) summary statistics: mean/normalized torque, power consumption
(raw + mass-normalized), normalized jerk, and tracking success -- everything the paper's
per-class cross-shape-variation comparison needs (aggregation happens in a separate script,
tools/aggregate_morphology_motion_interaction.py).

This extends tools/check_replay_torque_saturation.py's batched one-motion-per-env replay
skeleton (torque logging, mass-scaled effort limits) to full 19,200-pair coverage, plus
dof_vel/rigid_body_pos logging for power and jerk.

Batching: the simulator's per-env body shape is fixed at build time, so we can't give every
env a new shape each round -- only a new clip. We therefore fix a (num_assets x
envs_per_asset) env grid once (each asset repeated `envs_per_asset` times) and rotate which
clip each replica plays across ceil(num_clips / envs_per_asset) rounds. With the default
128 assets x 32 envs/asset = 4096 envs and 150 clips, that's 5 rounds.

Motion corpus: built directly from `--motion-file` via a fresh MotionLibConfig, NOT from the
checkpoint's own frozen `resolved_configs` motion_lib config -- that config is a
GlobalClipPool pointed at the training-time streaming R2 corpus (18,855 clips), which
ignores any `motion_file` override (it always sets `self.motion_file =
f"global_clip_pool_rank{rank}"` in its own __init__). The small 150x128 corpus needs a
plain MotionLib instead.

IMPORTANT: IsaacGym must be imported before torch, including transitively -- argument
parsing and the isaacgym import happen at module level before any other project import,
mirroring inference_agent.py / check_replay_torque_saturation.py.

Usage (smoke test, on a RunPod GPU instance):
    python tools/analyze_morphology_motion_interaction.py \\
        --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt \\
        --motion-file data_cache/small150_128shape_refined.pt \\
        --max-clips 3 --max-shapes 5 --envs-per-asset 3 --max-episode-steps 50 \\
        --output data_cache/morphology_motion_interaction.smoketest.csv

Usage (full run):
    python tools/analyze_morphology_motion_interaction.py \\
        --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt \\
        --motion-file data_cache/small150_128shape_refined.pt \\
        --output data_cache/morphology_motion_interaction.csv
"""

from __future__ import annotations

import argparse  # noqa: E402
import csv
import math
from pathlib import Path  # noqa: E402

ASSET_ROOT_DEFAULT = "protomotions/data/assets/mjcf/smpl_mor"
PHYSICS_FEATURES_DEFAULT = "protomotions/data/assets/mjcf/smpl_mor/physics_features.pt"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt"),
    )
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path(ASSET_ROOT_DEFAULT))
    parser.add_argument("--physics-features", type=Path, default=Path(PHYSICS_FEATURES_DEFAULT))
    parser.add_argument("--envs-per-asset", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--failure-gt-error-threshold", type=float, default=0.5)
    parser.add_argument("--jerk-window-sec", type=float, default=0.4)
    # Smoke-test knobs (see instrumentation validation, plan step 1): restrict to a tiny
    # slice of clips/shapes rather than requiring a whole separate code path.
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument(
        "--clip-id-filter",
        type=str,
        default=None,
        help="Comma-separated clip_ids to restrict to (applied before --max-clips), for "
        "smoke-testing specific (e.g. dynamic vs. quasi-static) clips.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


# Parse arguments first (argparse is safe, doesn't import torch) -- mirrors
# inference_agent.py's module-level ordering exactly.
_parser = create_parser()
_args = _parser.parse_args()

# IsaacGym must be imported before torch, including transitively -- so this happens before
# any project import below.
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("isaacgym")

# Now safe to import everything else including torch.
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402
from protomotions.agents.evaluators.metrics import MotionMetrics  # noqa: E402
from protomotions.agents.evaluators.smoothness_calculator import SmoothnessCalculator  # noqa: E402
from protomotions.components.motion_lib import MotionLibConfig  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.rewards.base import power_consumption_sum  # noqa: E402
from protomotions.envs.terminations.tracking import mean_body_pos_error  # noqa: E402
from protomotions.utils.component_builder import (  # noqa: E402
    build_all_components,
    build_motion_lib_from_config,
)
from protomotions.components.pose_lib import extract_body_masses  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.inference_utils import apply_backward_compatibility_fixes  # noqa: E402

OUTPUT_FIELDS = [
    "clip_id",
    "asset_id",
    "motion_id",
    "episode_len_steps",
    "failed",
    "max_gt_error",
    "mean_gt_error",
    "mean_abs_torque_raw",
    "mean_abs_torque_normalized",
    "mean_power_raw",
    "mean_power_mass_normalized",
    "normalized_jerk",
    "total_mass",
]


def load_asset_id_to_mass(physics_features_path: Path) -> dict:
    features = torch.load(physics_features_path, map_location="cpu", weights_only=False)
    return {asset_id: float(vals[0].item()) for asset_id, vals in features["features_raw"].items()}


def build_pair_index(motion_lib_probe) -> tuple[dict, list, list]:
    """Return ((clip_id, asset_id) -> motion_id, ordered unique clip_ids, ordered unique asset_ids)."""
    if motion_lib_probe.motion_clip_ids is None or motion_lib_probe.motion_asset_ids is None:
        raise RuntimeError(
            "Motion file is missing motion_clip_ids/motion_asset_ids -- is this really the "
            "matched clip x shape corpus (small150_128shape_refined.pt)?"
        )
    clip_ids = list(motion_lib_probe.motion_clip_ids)
    asset_ids = list(motion_lib_probe.motion_asset_ids)
    pair_to_motion_id = {(c, a): i for i, (c, a) in enumerate(zip(clip_ids, asset_ids))}
    unique_clip_ids = list(dict.fromkeys(clip_ids))
    unique_asset_ids = list(dict.fromkeys(asset_ids))
    return pair_to_motion_id, unique_clip_ids, unique_asset_ids


def compute_per_env_effort_limits(robot_config, asset_root: Path, asset_ids) -> np.ndarray:
    """Per-env effective effort limit, using the exact mass-scaling formula IsaacGym itself
    applies at actor-build time (isaacgym/simulator.py:1146-1177): gain_scale[dof] = this
    env's own body mass / reference shape's body mass. Same formula as
    tools/check_replay_torque_saturation.py:compute_per_env_effort_limits and
    tools/check_reference_torque_feasibility.py:compute_effective_effort_limits (copied
    inline rather than imported -- those scripts parse CLI args at module import time,
    which would collide with this script's own arguments)."""
    reference_body_masses = robot_config.control.reference_body_masses
    kin = robot_config.kinematic_info
    base_effort = np.array(
        [robot_config.control.control_info[n].effort_limit for n in kin.dof_names],
        dtype=np.float64,
    )

    mass_cache: dict = {}
    limits = np.zeros((len(asset_ids), kin.num_dofs), dtype=np.float64)
    for env_id, asset_id in enumerate(asset_ids):
        if asset_id not in mass_cache:
            mass_cache[asset_id] = extract_body_masses(str(asset_root / f"{asset_id}_smpl.xml"))
        shape_masses = mass_cache[asset_id]
        for i in range(kin.num_dofs):
            body_name = kin.body_names[kin.dof_body_ids[i]]
            gain_scale = shape_masses[body_name] / reference_body_masses[body_name]
            limits[env_id, i] = base_effort[i] * gain_scale
    return limits


def main():
    args = _args
    print(f"Loading resolved configs from {args.checkpoint.parent} ...")
    resolved_configs = torch.load(
        args.checkpoint.parent / "resolved_configs_inference.pt",
        map_location="cpu",
        weights_only=False,
    )
    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]  # already isaacgym -- no swap needed
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]

    # Do NOT reuse resolved_configs["motion_lib"] -- it's a GlobalClipPool pointed at the
    # training-time streaming corpus and ignores motion_file overrides. Build a plain
    # MotionLib against the matched clip x shape corpus instead.
    old_motion_lib_config = resolved_configs["motion_lib"]
    motion_lib_config = MotionLibConfig(
        motion_file=str(args.motion_file),
        get_motion_state_use_blend=getattr(old_motion_lib_config, "get_motion_state_use_blend", True),
    )

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)
    robot_config.asset.selected_asset_ids = None  # load all 128 shape assets

    fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric.launch()
    device = fabric.device

    print(f"Probing motion lib (CPU) from {args.motion_file} ...")
    motion_lib_probe = build_motion_lib_from_config(motion_lib_config, "cpu")
    pair_to_motion_id, clip_ids, asset_ids = build_pair_index(motion_lib_probe)
    print(
        f"Corpus: {motion_lib_probe.num_motions()} motions, "
        f"{len(clip_ids)} unique clips, {len(asset_ids)} unique shapes."
    )

    if args.clip_id_filter is not None:
        wanted = set(args.clip_id_filter.split(","))
        clip_ids = [c for c in clip_ids if c in wanted]
        missing_wanted = wanted - set(clip_ids)
        if missing_wanted:
            raise RuntimeError(f"--clip-id-filter clip_ids not found in corpus: {missing_wanted}")
    if args.max_clips is not None:
        clip_ids = clip_ids[: args.max_clips]
    if args.max_shapes is not None:
        asset_ids = asset_ids[: args.max_shapes]
    missing_pairs = [(c, a) for c in clip_ids for a in asset_ids if (c, a) not in pair_to_motion_id]
    if missing_pairs:
        raise RuntimeError(f"{len(missing_pairs)} (clip, shape) pairs missing, e.g. {missing_pairs[:5]}")

    envs_per_asset = min(args.envs_per_asset, len(clip_ids))
    num_envs = envs_per_asset * len(asset_ids)
    num_rounds = math.ceil(len(clip_ids) / envs_per_asset)
    print(
        f"Analyzing {len(clip_ids)} clips x {len(asset_ids)} shapes = "
        f"{len(clip_ids) * len(asset_ids)} pairs, as {num_envs} envs "
        f"({envs_per_asset} clip-replicas/asset) x {num_rounds} rounds.\n"
    )

    # Fixed env -> asset assignment for the whole run (simulator shape is set at build time).
    asset_ids_per_env = [asset for asset in asset_ids for _ in range(envs_per_asset)]

    simulator_config.num_envs = num_envs
    print("Building terrain/scene_lib/motion_lib/simulator (GPU) ...")
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        morphology_asset_ids=asset_ids_per_env,
    )
    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=args.checkpoint.parent
    )
    agent.setup()
    agent.load(str(args.checkpoint), load_env=False)

    effort_limits = compute_per_env_effort_limits(robot_config, args.asset_root, asset_ids_per_env)
    effort_limits_t = torch.tensor(effort_limits, device=device, dtype=torch.float32)

    asset_id_to_mass = load_asset_id_to_mass(args.physics_features)
    mass_per_env = np.array([asset_id_to_mass[a] for a in asset_ids_per_env], dtype=np.float64)

    num_dofs = robot_config.kinematic_info.num_dofs
    num_bodies = robot_config.kinematic_info.num_bodies
    smoothness_calc = SmoothnessCalculator(device=device, dt=env.dt, window_sec=args.jerk_window_sec)

    env_ids_all = torch.arange(num_envs, device=device, dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(args.output, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for round_idx in range(num_rounds):
            clip_slice = clip_ids[round_idx * envs_per_asset : (round_idx + 1) * envs_per_asset]
            print(f"=== Round {round_idx + 1}/{num_rounds}: {len(clip_slice)} clips ===")

            motion_ids_this_round = []
            valid_mask = []
            for asset in asset_ids:
                for k in range(envs_per_asset):
                    if k < len(clip_slice):
                        motion_ids_this_round.append(pair_to_motion_id[(clip_slice[k], asset)])
                        valid_mask.append(True)
                    else:
                        motion_ids_this_round.append(pair_to_motion_id[(clip_ids[0], asset)])
                        valid_mask.append(False)

            motion_ids_t = torch.tensor(motion_ids_this_round, device=device, dtype=torch.long)
            env.motion_manager.motion_ids[env_ids_all] = motion_ids_t
            env.motion_manager.motion_times[env_ids_all] = 0.0
            obs, _ = env.reset(env_ids_all, sample_flat=True, disable_motion_resample=True)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            motion_lengths_steps = (motion_lib.get_motion_length(motion_ids_t) / env.dt).floor().long()
            max_steps = int(min(motion_lengths_steps.max().item(), args.max_episode_steps))

            torque_trace = torch.zeros((max_steps, num_envs, num_dofs), dtype=torch.float32, device=device)
            dof_vel_trace = torch.zeros((max_steps, num_envs, num_dofs), dtype=torch.float32, device=device)
            rigid_body_pos_trace = torch.zeros(
                (max_steps, num_envs, num_bodies * 3), dtype=torch.float32, device=device
            )
            gt_error_trace = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
            terminated_trace = torch.zeros((max_steps, num_envs), dtype=torch.bool, device=device)

            for step_idx in range(max_steps):
                with torch.no_grad():
                    model_outs = agent.model(obs_td)
                action = model_outs.get("mean_action", model_outs.get("action"))
                obs, rewards, dones, terminated, extras = env.step(action)
                obs = agent.add_agent_info_to_obs(obs)
                obs_td = agent.obs_dict_to_tensordict(obs)

                torque_trace[step_idx] = extras["raw/dof_forces"]
                dof_vel_trace[step_idx] = env.context.current.dof_vel
                rigid_body_pos_trace[step_idx] = env.context.current.rigid_body_pos.reshape(num_envs, -1)
                gt_error_trace[step_idx] = mean_body_pos_error(
                    env.context.current.rigid_body_pos, env.context.mimic.ref_state.rigid_body_pos
                )
                terminated_trace[step_idx] = terminated.bool()

            # --- per-env summary stats, then discard the raw per-step tensors ---
            ep_lens = torch.minimum(motion_lengths_steps, torch.full_like(motion_lengths_steps, max_steps))

            power_trace = power_consumption_sum(
                torque_trace.reshape(-1, num_dofs), dof_vel_trace.reshape(-1, num_dofs)
            ).reshape(max_steps, num_envs)
            normalized_torque_trace = torque_trace.abs() / effort_limits_t[None, :, :]

            motion_lens_for_jerk = torch.clamp(ep_lens, max=max_steps)
            pos_metric = MotionMetrics(
                num_motions=num_envs,
                motion_lens=motion_lens_for_jerk,
                max_motion_len=max_steps,
                num_sub_features=num_bodies * 3,
                device=device,
                dtype=torch.float32,
            )
            pos_metric.data[:, :max_steps, :] = rigid_body_pos_trace.permute(1, 0, 2)
            pos_metric.frame_counts = motion_lens_for_jerk
            per_motion_nj, _, _ = smoothness_calc.compute_normalized_jerk_from_pos(pos_metric, num_bodies)

            for env_id in range(num_envs):
                if not valid_mask[env_id]:
                    continue
                ep_len = int(ep_lens[env_id].item())
                if ep_len < 2:
                    continue
                clip_id = clip_slice[env_id % envs_per_asset]
                asset_id = asset_ids_per_env[env_id]

                torque_ep = torque_trace[:ep_len, env_id, :]
                norm_torque_ep = normalized_torque_trace[:ep_len, env_id, :]
                power_ep = power_trace[:ep_len, env_id]
                gt_error_ep = gt_error_trace[:ep_len, env_id]
                failed = bool(terminated_trace[:ep_len, env_id].any().item())

                mass = mass_per_env[env_id]
                row = {
                    "clip_id": clip_id,
                    "asset_id": asset_id,
                    "motion_id": int(motion_ids_this_round[env_id]),
                    "episode_len_steps": ep_len,
                    "failed": int(failed),
                    "max_gt_error": float(gt_error_ep.max().item()),
                    "mean_gt_error": float(gt_error_ep.mean().item()),
                    "mean_abs_torque_raw": float(torque_ep.abs().mean().item()),
                    "mean_abs_torque_normalized": float(norm_torque_ep.mean().item()),
                    "mean_power_raw": float(power_ep.mean().item()),
                    "mean_power_mass_normalized": float(power_ep.mean().item() / mass),
                    "normalized_jerk": float(per_motion_nj[env_id].item()),
                    "total_mass": float(mass),
                }
                writer.writerow(row)
                rows_written += 1

            out_f.flush()
            del torque_trace, dof_vel_trace, rigid_body_pos_trace, gt_error_trace, terminated_trace
            del power_trace, normalized_torque_trace, pos_metric
            print(f"  Round {round_idx + 1} done: {max_steps} steps, {rows_written} total rows so far.")

    print(f"\nWrote {rows_written} rows to {args.output}")


if __name__ == "__main__":
    main()
