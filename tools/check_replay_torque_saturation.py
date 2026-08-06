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
Replay a trained checkpoint against specific (clip, shape) motion_ids in IsaacGym (the
simulator it was trained in), and log the real applied torque (already effort-limit-
clamped by PhysX) alongside per-step tracking error, all envs in parallel.

Why: note/README.note.md 51/52 narrowed the `discover` plateau to two hypotheses --
(1) actuator saturation (a physics ceiling) vs (2) an RL precision plateau. Two earlier
approaches were abandoned:
  - Offline inverse dynamics (tools/check_reference_torque_feasibility.py): self-collision
    between adjacent limb capsules in the raw reference pose made mj_inverse report
    physically-meaningless torque spikes.
  - Replay via the MuJoCo CPU backend: MuJoCo has no native per-env morphology support
    (needed shims for env_morphology/env_id_to_asset_name/etc, all worked and were
    verified against the repo's own betas-consistency check), and MuJoCo's default
    implicit-PD mode never updates its applied-torque cache (fixed by forcing explicit
    PD) -- but the replay then showed a violent, unexplained physics blow-up in the first
    ~150 steps of every episode, contaminating the saturation signal. Root cause not
    isolated; abandoned in favor of just using the real training simulator.

This script uses IsaacGym directly -- the same simulator `discover` was trained and
evaluated in -- so none of the above workarounds are needed: IsaacGym natively supports
per-env body shapes (`morphology_asset_ids`, isaacgym/simulator.py:124-137) and natively
tracks real applied DOF torque via PhysX's force sensor tensor
(isaacgym/simulator.py:1672-1673). Requires a GPU (RunPod).

All ~60 treatment+control motion_ids are pinned one-per-env and run in ONE batched
simulator instance (num_envs = number of motion_ids), stepped in parallel -- much cheaper
than one simulator rebuild per motion/shape group (the approach the MuJoCo version needed).

IMPORTANT: IsaacGym must be imported before torch (protomotions/utils/simulator_imports.py)
-- this holds even transitively, so argument parsing and the isaacgym import happen at
module level BEFORE any other project import, mirroring inference_agent.py's structure.

Usage (on a RunPod GPU instance):
    python tools/check_replay_torque_saturation.py \\
        --checkpoint results/hhi_wide_150motion_128shape_discover/last.ckpt \\
        --results-dir results/hhi_wide_150motion_128shape_discover \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --top-n 30 \\
        --output results/hhi_wide_150motion_128shape_discover/replay_torque_saturation_analysis.txt
"""

from __future__ import annotations

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

ASSET_ROOT_DEFAULT = "protomotions/data/assets/mjcf/smpl_mor"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/hhi_wide_150motion_128shape_discover/last.ckpt"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/hhi_wide_150motion_128shape_discover"),
    )
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path(ASSET_ROOT_DEFAULT))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--max-eval-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    return parser


# Parse arguments first (argparse is safe, doesn't import torch) -- mirrors
# inference_agent.py's module-level ordering exactly.
_parser = create_parser()
_args = _parser.parse_args()

# IsaacGym must be imported before torch, including transitively -- so this happens before
# any project import below (pose_lib, tracking, analyze_shape_failure_correlation, etc. all
# import torch transitively).
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("isaacgym")

# Now safe to import everything else including torch.
import contextlib  # noqa: E402
import sys  # noqa: E402
from typing import Dict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402
from protomotions.components.pose_lib import extract_body_masses  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.terminations.tracking import mean_body_pos_error  # noqa: E402
from protomotions.utils.component_builder import (  # noqa: E402
    build_all_components,
    build_motion_lib_from_config,
)
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.inference_utils import apply_backward_compatibility_fixes  # noqa: E402
from tools.analyze_shape_failure_correlation import read_failed_motion_events  # noqa: E402


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def build_treatment_and_control_sets(motion_lib, failed_dir: Path, top_n: int):
    """Treatment = top-N motion_ids by fail-event count (the exact (clip, shape) pairs
    discover's own eval logs showed failing). Control = one representative motion_id each
    from the N clips with zero fail events."""
    fail_counts, n_epochs = read_failed_motion_events(failed_dir)
    print(f"Read {n_epochs} eval epochs, {len(fail_counts)} unique failing motion ids.")

    treatment = [mid for mid, _ in fail_counts.most_common(top_n)]

    clip_to_motion_ids = motion_lib.build_clip_id_to_motion_ids()
    failing_motion_ids = set(fail_counts.keys())
    zero_fail_clips = [
        clip_id
        for clip_id, mids in clip_to_motion_ids.items()
        if not any(int(mid) in failing_motion_ids for mid in mids)
    ]
    control = [int(clip_to_motion_ids[clip_id][0]) for clip_id in zero_fail_clips[:top_n]]

    print(f"Treatment set: {len(treatment)} motion ids (highest fail-event count).")
    print(f"Control set: {len(control)} motion ids (from clips with zero fail events).\n")
    return treatment, control


def compute_per_env_effort_limits(robot_config, asset_root: Path, asset_ids) -> np.ndarray:
    """Per-env effective effort limit, using the exact mass-scaling formula IsaacGym
    itself applies at actor-build time (isaacgym/simulator.py:1146-1177):
    gain_scale[dof] = this env's own body mass / reference shape's body mass. Recomputed
    independently here (rather than read back from PhysX internals) since it's a pure
    function of each shape's MJCF + the checkpoint's own reference_body_masses."""
    reference_body_masses = robot_config.control.reference_body_masses
    kin = robot_config.kinematic_info
    base_effort = np.array(
        [robot_config.control.control_info[n].effort_limit for n in kin.dof_names],
        dtype=np.float64,
    )

    mass_cache: Dict[str, dict] = {}
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


def analyze_episode(
    torque: np.ndarray, gt_error: np.ndarray, effort_limits: np.ndarray, threshold: float = 0.5
) -> dict:
    """torque: [num_steps, num_dofs]. gt_error: [num_steps]. effort_limits: [num_dofs]."""
    ratio = np.abs(torque) / effort_limits[None, :]
    saturating = ratio >= 0.99

    breach_steps = np.where(gt_error > threshold)[0]
    first_breach = int(breach_steps[0]) if breach_steps.size > 0 else None

    frac_sat_overall = float(saturating.mean())
    frac_sat_failure_window = (
        float(saturating[first_breach:].mean()) if first_breach is not None else float("nan")
    )
    # Near-miss residual error: what gt_error typically looks like right before the policy
    # first crosses the failure threshold. Used to calibrate reward-gradient-sharpening
    # coefficients (note.md §53 / discover_pilot_status.md) -- see if the "close but not
    # exact" near-misses sit well below threshold or right up against it.
    pre_breach_gt_error = (
        float(gt_error[max(0, first_breach - 5):first_breach].mean())
        if first_breach is not None and first_breach > 0
        else float("nan")
    )

    return {
        "episode_len": torque.shape[0],
        "first_breach_step": first_breach,
        "frac_sat_overall": frac_sat_overall,
        "frac_sat_failure_window": frac_sat_failure_window,
        "max_ratio": float(ratio.max()),
        "pre_breach_gt_error": pre_breach_gt_error,
    }


def main():
    args = _args
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
                _run(args)
        print(f"\nReport written to {args.output}")
    else:
        _run(args)


def _run(args) -> None:
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
    motion_lib_config = resolved_configs["motion_lib"]
    motion_lib_config.motion_file = str(args.motion_file)
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric.launch()
    device = fabric.device

    # First pass: load just the motion lib (CPU-only, cheap) to pick treatment/control ids
    # and figure out per-env asset_ids before building the (expensive) GPU simulator.
    motion_lib_probe = build_motion_lib_from_config(motion_lib_config, "cpu")
    failed_dir = args.results_dir / "failed_motions"
    treatment, control = build_treatment_and_control_sets(motion_lib_probe, failed_dir, args.top_n)
    all_ids = treatment + control
    labels = ["treatment"] * len(treatment) + ["control"] * len(control)
    asset_ids = list(motion_lib_probe.get_motion_asset_ids(torch.tensor(all_ids)))
    num_envs = len(all_ids)
    print(f"{num_envs} motion ids ({len(treatment)} treatment + {len(control)} control), "
          f"{len(set(asset_ids))} distinct shapes -- running as {num_envs} parallel envs.\n")

    simulator_config.num_envs = num_envs
    robot_config.asset.selected_asset_ids = None  # load all 128 shape assets

    print("Building terrain/scene_lib/motion_lib/simulator (GPU) ...")
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        morphology_asset_ids=asset_ids,  # env i gets exactly asset_ids[i] (isaacgym/simulator.py:124-137)
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

    effort_limits = compute_per_env_effort_limits(robot_config, args.asset_root, asset_ids)

    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    motion_ids_t = torch.tensor(all_ids, device=device, dtype=torch.long)
    env.motion_manager.motion_ids[env_ids] = motion_ids_t
    env.motion_manager.motion_times[env_ids] = 0.0
    obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
    obs = agent.add_agent_info_to_obs(obs)
    obs_td = agent.obs_dict_to_tensordict(obs)

    motion_lengths_steps = (motion_lib.get_motion_length(motion_ids_t) / env.dt).floor().long()
    max_steps = int(min(motion_lengths_steps.max().item(), args.max_eval_steps))
    num_dofs = robot_config.kinematic_info.num_dofs

    torque_trace = np.zeros((max_steps, num_envs, num_dofs), dtype=np.float64)
    gt_error_trace = np.zeros((max_steps, num_envs), dtype=np.float64)

    print(f"Running {max_steps} steps across {num_envs} parallel envs ...")
    for step_idx in range(max_steps):
        with torch.no_grad():
            model_outs = agent.model(obs_td)
        action = model_outs.get("mean_action", model_outs.get("action"))
        obs, rewards, dones, terminated, extras = env.step(action)
        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)

        torque_trace[step_idx] = extras["raw/dof_forces"].detach().cpu().numpy()
        gt_error = mean_body_pos_error(
            env.context.current.rigid_body_pos, env.context.mimic.ref_state.rigid_body_pos
        )
        gt_error_trace[step_idx] = gt_error.detach().cpu().numpy()

    results = []
    for env_id in range(num_envs):
        ep_len = int(min(motion_lengths_steps[env_id].item(), max_steps))
        stats = analyze_episode(
            torque_trace[:ep_len, env_id, :], gt_error_trace[:ep_len, env_id], effort_limits[env_id]
        )
        stats.update({"motion_id": all_ids[env_id], "asset_id": asset_ids[env_id], "label": labels[env_id]})
        results.append(stats)
        print(
            f"  [{labels[env_id]}] motion_id={all_ids[env_id]:<7} len={stats['episode_len']:<4} "
            f"first_breach={str(stats['first_breach_step']):<6} "
            f"sat_overall={stats['frac_sat_overall']:.3f} "
            f"sat_failure_window={stats['frac_sat_failure_window']:.3f} "
            f"max_ratio={stats['max_ratio']:.2f} "
            f"pre_breach_gt_error={stats['pre_breach_gt_error']:.3f}"
        )

    treatment_results = [r for r in results if r["label"] == "treatment"]
    control_results = [r for r in results if r["label"] == "control"]

    def has_saturation(r: dict) -> bool:
        return r["max_ratio"] >= 0.99

    n_treat_sat = sum(has_saturation(r) for r in treatment_results)
    n_control_sat = sum(has_saturation(r) for r in control_results)
    n1, n2 = len(treatment_results), len(control_results)
    p_treat = n_treat_sat / n1 if n1 else float("nan")
    p_control = n_control_sat / n2 if n2 else float("nan")

    print("\n=== Aggregate: treatment vs. control ===")
    print(f"Treatment: {n_treat_sat}/{n1} ({100*p_treat:.1f}%) episodes show real torque saturation")
    print(f"Control:   {n_control_sat}/{n2} ({100*p_control:.1f}%) episodes show real torque saturation")

    if n1 and n2:
        p_pool = (n_treat_sat + n_control_sat) / (n1 + n2)
        se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5
        z = (p_treat - p_control) / se if se > 0 else float("nan")
        print(f"\nTwo-proportion z-test (treatment vs. control saturation rate): z = {z:.2f}")
        print(
            "|z| > ~1.96 => statistically distinguishable at p<0.05. High treatment "
            "saturation rate + significant z => supports actuator-saturation hypothesis "
            "(H1). Low/similar rates => supports RL-precision-plateau hypothesis (H2)."
        )

    pre_breach_values = [
        r["pre_breach_gt_error"] for r in treatment_results if not np.isnan(r["pre_breach_gt_error"])
    ]
    print("\n=== Near-miss residual error (gt_error in the 5 steps before first breach) ===")
    if pre_breach_values:
        arr = np.array(pre_breach_values)
        print(
            f"n={len(arr)}  median={np.median(arr):.3f}  mean={arr.mean():.3f}  "
            f"min={arr.min():.3f}  max={arr.max():.3f}"
        )
        print(
            "Use the median above as e_target to calibrate reward-gradient sharpening: "
            "gt_coef_new = -3 / (2 * e_target**2)."
        )
    else:
        print("No treatment episodes breached the threshold in this run -- nothing to report.")


if __name__ == "__main__":
    main()
