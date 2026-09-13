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
Unseen-morphology generalization (paper section 6.5.1). See
note/README.heldout-pipeline.md and note/README.note.md (search "E7") for the data-generation
history: 32 new body shapes ("smpl_mor_interp": 16 beta vectors x 2 genders), sampled in the
same [-3,3]^10 range as the 128 trained shapes but never used in training, paired with 1024 of
the corpus's own clip identities (not new motions). This isolates the shape axis: for any clip
whose id is in the training split, the policy has seen that motion before, on trained shapes --
only the body is new.

Replays the trained checkpoint, deterministically, on every (clip, unseen-shape) pair, and logs
per-(clip, shape) success/error -- exactly the metrics in the main validation/test tables -- for
a separate script to join against distance-to-nearest-trained-morphology and clip split
membership (train vs. validation vs. test).

Deliberately does NOT reuse the 128 trained-shape asset folder: robot_config.asset is
repointed at protomotions/data/assets/mjcf/smpl_mor_interp/ (asset_folder_name + asset_info_file
+ selected_asset_ids=None) so the simulator discovers exactly the 32 new shapes. Mass-scaled PD
gains still use resolved_configs' own reference_body_masses (a concrete dict already resolved at
training time, not re-derived here), so this override does not change actuation elsewhere.

IMPORTANT: IsaacGym must be imported before torch, including transitively -- argument parsing and
the isaacgym import happen at module level before any other project import, mirroring
tools/analyze_morphology_motion_interaction.py.

Usage (smoke test):
    python tools/analyze_unseen_morphology_generalization.py \\
        --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt \\
        --motion-file data_cache/unseen_morphology_merged_0.pt \\
        --max-clips 3 --envs-per-asset 3 --max-episode-steps 50 \\
        --output data_cache/unseen_morphology_generalization.smoketest.csv

Usage (full run):
    python tools/analyze_unseen_morphology_generalization.py \\
        --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt \\
        --motion-file data_cache/unseen_morphology_merged_0.pt \\
        --output data_cache/unseen_morphology_generalization.csv
"""

from __future__ import annotations

import argparse  # noqa: E402
import csv
import math
from pathlib import Path  # noqa: E402

UNSEEN_ASSET_FOLDER_DEFAULT = "mjcf/smpl_mor_interp/"
UNSEEN_ASSET_INFO_DEFAULT = "mjcf/smpl_mor_interp/assets.yaml"


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
    parser.add_argument("--unseen-asset-folder", type=str, default=UNSEEN_ASSET_FOLDER_DEFAULT)
    parser.add_argument("--unseen-asset-info", type=str, default=UNSEEN_ASSET_INFO_DEFAULT)
    parser.add_argument("--envs-per-asset", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=210)
    parser.add_argument("--success-gt-error-threshold", type=float, default=0.5)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-id-filter", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


# Parse arguments first (argparse is safe, doesn't import torch) -- mirrors
# inference_agent.py / analyze_morphology_motion_interaction.py's module-level ordering.
_parser = create_parser()
_args = _parser.parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("isaacgym")

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402
from protomotions.components.motion_lib import MotionLibConfig  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.terminations.tracking import (  # noqa: E402
    mean_body_pos_error,
    mean_body_rot_error,
)
from protomotions.utils.component_builder import (  # noqa: E402
    build_all_components,
    build_motion_lib_from_config,
)
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.inference_utils import apply_backward_compatibility_fixes  # noqa: E402

OUTPUT_FIELDS = [
    "clip_id",
    "asset_id",
    "motion_id",
    "episode_len_steps",
    "success",
    "terminated",
    "max_gt_error",
    "mean_gt_error",
    "mean_gr_error",
]


def build_pair_index(motion_lib_probe) -> tuple[dict, list, list]:
    if motion_lib_probe.motion_clip_ids is None or motion_lib_probe.motion_asset_ids is None:
        raise RuntimeError(
            "Motion file is missing motion_clip_ids/motion_asset_ids -- is this really the "
            "merged unseen-morphology corpus?"
        )
    clip_ids = list(motion_lib_probe.motion_clip_ids)
    asset_ids = list(motion_lib_probe.motion_asset_ids)
    pair_to_motion_id = {(c, a): i for i, (c, a) in enumerate(zip(clip_ids, asset_ids))}
    unique_clip_ids = list(dict.fromkeys(clip_ids))
    unique_asset_ids = list(dict.fromkeys(asset_ids))
    return pair_to_motion_id, unique_clip_ids, unique_asset_ids


def main():
    args = _args
    print(f"Loading resolved configs from {args.checkpoint.parent} ...")
    resolved_configs = torch.load(
        args.checkpoint.parent / "resolved_configs_inference.pt",
        map_location="cpu",
        weights_only=False,
    )
    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]

    old_motion_lib_config = resolved_configs["motion_lib"]
    motion_lib_config = MotionLibConfig(
        motion_file=str(args.motion_file),
        get_motion_state_use_blend=getattr(old_motion_lib_config, "get_motion_state_use_blend", True),
    )

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # Repoint the simulator at the 32 unseen-morphology assets instead of the 128 trained ones.
    # reference_body_masses is already a concrete resolved dict from training time (not
    # re-derived from asset_folder_name here), so mass-scaled PD gains are unaffected.
    robot_config.asset.asset_folder_name = args.unseen_asset_folder
    robot_config.asset.asset_info_file = args.unseen_asset_info
    robot_config.asset.selected_asset_ids = None

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

            gt_error_trace = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
            gr_error_trace = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
            terminated_trace = torch.zeros((max_steps, num_envs), dtype=torch.bool, device=device)

            for step_idx in range(max_steps):
                with torch.no_grad():
                    model_outs = agent.model(obs_td)
                action = model_outs.get("mean_action", model_outs.get("action"))
                obs, rewards, dones, terminated, extras = env.step(action)
                obs = agent.add_agent_info_to_obs(obs)
                obs_td = agent.obs_dict_to_tensordict(obs)

                gt_error_trace[step_idx] = mean_body_pos_error(
                    env.context.current.rigid_body_pos, env.context.mimic.ref_state.rigid_body_pos
                )
                gr_error_trace[step_idx] = mean_body_rot_error(
                    env.context.current.rigid_body_rot, env.context.mimic.ref_state.rigid_body_rot
                )
                terminated_trace[step_idx] = terminated.bool()

            ep_lens = torch.minimum(motion_lengths_steps, torch.full_like(motion_lengths_steps, max_steps))

            for env_id in range(num_envs):
                if not valid_mask[env_id]:
                    continue
                ep_len = int(ep_lens[env_id].item())
                if ep_len < 2:
                    continue
                clip_id = clip_slice[env_id % envs_per_asset]
                asset_id = asset_ids_per_env[env_id]

                gt_error_ep = gt_error_trace[:ep_len, env_id]
                gr_error_ep = gr_error_trace[:ep_len, env_id]
                max_gt_error = float(gt_error_ep.max().item())
                success = int(max_gt_error <= args.success_gt_error_threshold)
                failed_by_termination = bool(terminated_trace[:ep_len, env_id].any().item())

                row = {
                    "clip_id": clip_id,
                    "asset_id": asset_id,
                    "motion_id": int(motion_ids_this_round[env_id]),
                    "episode_len_steps": ep_len,
                    "success": success,
                    "terminated": int(failed_by_termination),
                    "max_gt_error": max_gt_error,
                    "mean_gt_error": float(gt_error_ep.mean().item()),
                    "mean_gr_error": float(gr_error_ep.mean().item()),
                }
                writer.writerow(row)
                rows_written += 1

            out_f.flush()
            del gt_error_trace, gr_error_trace, terminated_trace
            print(f"  Round {round_idx + 1} done: {max_steps} steps, {rows_written} total rows so far.")

    print(f"\nWrote {rows_written} rows to {args.output}")


if __name__ == "__main__":
    main()
