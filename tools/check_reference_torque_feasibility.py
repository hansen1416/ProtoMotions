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
Offline inverse-dynamics check: does the raw mocap reference trajectory require more torque
than the actuator (effort-limit) model allows, for a given SMPL body shape?

Why: `discover` (relaxed termination + reward, see note/README.note.md §50) plateaus at
75-78% success with fall rate ~=0 and "close but not exact" near-miss failures. It already
removed every jerk/smoothness/effort penalty, so jerk-avoidance itself is ruled out as the
blocker (note.md §51). Two hypotheses remain: (1) actuator saturation -- the reference simply
demands more torque than the PD gains/effort limits can deliver for that body, a physics
ceiling independent of the RL policy; (2) an RL precision plateau -- physically achievable,
optimization just isn't converging tightly enough. This script tests (1) directly and offline:
for each (clip, shape) pair, treat the reference trajectory as ground truth, use MuJoCo's
native mj_inverse to compute the joint torque required to reproduce it exactly, and compare
against that shape's actual effort limit (mass-scaled the same way training does).

This needs no GPU/IsaacGym and no trained checkpoint -- runs entirely on CPU locally.

Usage:
    python tools/check_reference_torque_feasibility.py \\
        --results-dir results/hhi_wide_150motion_128shape_discover \\
        --motion-file data_cache/small150_128shape.pt \\
        --top-n 30 \\
        --output results/hhi_wide_150motion_128shape_discover/torque_feasibility_analysis.txt
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.components.pose_lib import extract_body_masses
from protomotions.robot_configs.smpl_mor import SmplMorRobotConfig
from protomotions.simulator.mujoco.simulator import MujocoSimulator
from protomotions.utils import rotations
from tools.analyze_shape_failure_correlation import read_failed_motion_events


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


ASSET_ROOT_DEFAULT = "protomotions/data/assets/mjcf/smpl_mor"


def build_treatment_and_control_sets(
    motion_lib: MotionLib, failed_dir: Path, top_n: int
) -> Tuple[List[int], List[int]]:
    """Treatment = top-N motion_ids by fail-event count (the exact (clip, shape) pairs
    discover's own eval logs showed failing). Control = one representative motion_id each
    from the N clips with zero fail events (same "success bucket" reasoning as
    tools/select_visualization_clips.py -- eval_one_shape_per_motion means a specific
    (clip, shape) pair is rarely revisited, but a clip that never failed regardless of
    shape is a genuine easy case; shape 0 of that clip is a fair representative)."""
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


def build_dof_addr_map(model: mujoco.MjModel, dof_names: List[str]) -> Dict[str, Tuple[int, int]]:
    """dof_name -> (qpos_addr, qvel_addr). Assumes every named DOF is a single-DOF hinge
    joint (confirmed true for smpl_mor's MJCFs -- 3 stacked hinge joints per anatomical
    joint, e.g. L_Hip_x/L_Hip_y/L_Hip_z -- not a true ball joint), so qpos and qvel each
    hold exactly one scalar per named joint."""
    name_to_addrs = {}
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        assert name in dof_names, f"Joint '{name}' in MJCF not found in kinematic_info.dof_names"
        name_to_addrs[name] = (int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i]))
    missing = set(dof_names) - set(name_to_addrs.keys())
    assert not missing, f"dof_names not found as MJCF joints: {missing}"
    return name_to_addrs


def load_shape_model(asset_root: Path, asset_id: str, robot_config) -> mujoco.MjModel:
    """Load one shape's MJCF and normalize its physics to match training: MuJoCo's own
    passive joint stiffness/damping are zeroed (PD control comes entirely from control_info,
    not MJCF passive terms) and per-joint armature is overridden from control_info -- exactly
    what MujocoSimulator._zero_passive_forces/_override_joint_properties do at train/eval
    time (mujoco/simulator.py:372-419), replicated here as plain array mutations since we're
    not instantiating the full Simulator class."""
    xml_path = asset_root / f"{asset_id}_smpl.xml"
    model = MujocoSimulator._load_mjcf_stripped(str(xml_path))

    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0

    control_info = robot_config.control.control_info
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        info = control_info.get(name)
        if info is not None and info.armature is not None:
            dof_addr = model.jnt_dofadr[i]
            model.dof_armature[dof_addr] = info.armature

    return model


def compute_effective_effort_limits(
    robot_config, asset_root: Path, asset_id: str, reference_body_masses: Dict[str, float]
) -> np.ndarray:
    """Effective per-DOF effort limit for this shape, using the exact mass-scaling formula
    training uses (isaacgym/simulator.py:1146-1177): gain_scale[dof] = this shape's own
    body mass / reference shape's body mass, for the body that DOF actuates."""
    shape_body_masses = extract_body_masses(str(asset_root / f"{asset_id}_smpl.xml"))
    kin = robot_config.kinematic_info
    control_info = robot_config.control.control_info

    limits = np.zeros(kin.num_dofs, dtype=np.float64)
    for i, dof_name in enumerate(kin.dof_names):
        body_name = kin.body_names[kin.dof_body_ids[i]]
        gain_scale = shape_body_masses[body_name] / reference_body_masses[body_name]
        base_limit = control_info[dof_name].effort_limit
        limits[i] = base_limit * gain_scale
    return limits


def finite_difference(vel: np.ndarray, dt: float) -> np.ndarray:
    """Central difference (interior frames), forward/backward at the ends. No
    differentiation of qpos/quaternions needed -- qvel is already in MuJoCo's tangent
    space, so plain d(qvel)/dt gives qacc directly."""
    acc = np.zeros_like(vel)
    acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * dt)
    acc[0] = (vel[1] - vel[0]) / dt
    acc[-1] = (vel[-1] - vel[-2]) / dt
    return acc


def compute_required_torques(
    model: mujoco.MjModel,
    dof_addr_map: Dict[str, Tuple[int, int]],
    dof_names: List[str],
    root_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    root_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    dt: float,
) -> np.ndarray:
    """Returns required torque per frame per DOF, in common (dof_names) order,
    shape [num_frames, num_dofs]."""
    num_frames = dof_pos.shape[0]
    num_dofs = len(dof_names)
    root_rot_wxyz = rotations.xyzw_to_wxyz(root_rot_xyzw)

    # Finite-difference each velocity stream independently, then scatter into MuJoCo's
    # native qvel/qacc layout via the SAME name-based dof_addr_map used for qpos/qvel --
    # common (dof_names) order is not guaranteed to match MuJoCo's own joint order.
    root_lin_acc = finite_difference(root_vel.numpy(), dt)
    root_ang_acc = finite_difference(root_ang_vel.numpy(), dt)
    dof_acc = finite_difference(dof_vel.numpy(), dt)

    data = mujoco.MjData(model)
    torques = np.zeros((num_frames, num_dofs), dtype=np.float64)

    for t in range(num_frames):
        data.qpos[0:3] = root_pos[t].numpy()
        data.qpos[3:7] = root_rot_wxyz[t].numpy()
        data.qvel[0:3] = root_vel[t].numpy()
        data.qvel[3:6] = root_ang_vel[t].numpy()
        data.qacc[0:3] = root_lin_acc[t]
        data.qacc[3:6] = root_ang_acc[t]
        for i, name in enumerate(dof_names):
            qpos_addr, qvel_addr = dof_addr_map[name]
            data.qpos[qpos_addr] = dof_pos[t, i].item()
            data.qvel[qvel_addr] = dof_vel[t, i].item()
            data.qacc[qvel_addr] = dof_acc[t, i]

        mujoco.mj_inverse(model, data)

        for i, name in enumerate(dof_names):
            _, qvel_addr = dof_addr_map[name]
            torques[t, i] = data.qfrc_inverse[qvel_addr]

    return torques


def analyze_motion(
    motion_lib: MotionLib,
    motion_id: int,
    robot_config,
    asset_root: Path,
    reference_body_masses: Dict[str, float],
    model_cache: Dict[str, mujoco.MjModel],
    addr_map_cache: Dict[str, Dict[str, Tuple[int, int]]],
    limit_cache: Dict[str, np.ndarray],
) -> dict:
    asset_id = motion_lib.get_motion_asset_ids(torch.tensor([motion_id]))[0]
    if asset_id not in model_cache:
        model_cache[asset_id] = load_shape_model(asset_root, asset_id, robot_config)
        addr_map_cache[asset_id] = build_dof_addr_map(
            model_cache[asset_id], robot_config.kinematic_info.dof_names
        )
        limit_cache[asset_id] = compute_effective_effort_limits(
            robot_config, asset_root, asset_id, reference_body_masses
        )

    num_frames = int(motion_lib.get_motion_num_frames(torch.tensor([motion_id])).item())
    dt = motion_lib.motion_dt[motion_id].item()
    motion_ids_batch = torch.full((num_frames,), motion_id, dtype=torch.long)
    frame_indices = torch.arange(num_frames)
    state = motion_lib.get_motion_state_exact_frame(motion_ids_batch, frame_indices)

    torques = compute_required_torques(
        model_cache[asset_id],
        addr_map_cache[asset_id],
        robot_config.kinematic_info.dof_names,
        state.root_pos,
        state.root_rot,
        state.root_vel,
        state.root_ang_vel,
        state.dof_pos,
        state.dof_vel,
        dt,
    )

    # Drop first/last frame -- least-reliable finite-difference acceleration.
    torques = torques[1:-1]
    limits = limit_cache[asset_id]
    ratio = np.abs(torques) / limits[None, :]

    worst_flat = np.argmax(ratio)
    worst_frame, worst_dof = np.unravel_index(worst_flat, ratio.shape)

    return {
        "motion_id": motion_id,
        "asset_id": asset_id,
        "max_ratio": float(ratio.max()),
        "frac_violating": float((ratio > 1.0).mean()),
        "worst_dof_name": robot_config.kinematic_info.dof_names[worst_dof],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path(ASSET_ROOT_DEFAULT))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
                _run(args)
        print(f"\nReport written to {args.output}")
    else:
        _run(args)


def _run(args) -> None:
    print(f"Loading motion library from {args.motion_file} ...")
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=str(args.motion_file)), device="cpu"
    )
    if not motion_lib.has_morphology_metadata():
        raise SystemExit("MotionLib has no morphology metadata (betas/gender/asset ids).")

    failed_dir = args.results_dir / "failed_motions"
    if not failed_dir.exists():
        raise SystemExit(f"No failed_motions dir at {failed_dir}")

    treatment, control = build_treatment_and_control_sets(motion_lib, failed_dir, args.top_n)

    robot_config = SmplMorRobotConfig()
    reference_body_masses = robot_config.control.reference_body_masses

    model_cache: Dict[str, mujoco.MjModel] = {}
    addr_map_cache: Dict[str, Dict[str, Tuple[int, int]]] = {}
    limit_cache: Dict[str, np.ndarray] = {}

    def run_group(motion_ids: List[int], label: str) -> List[dict]:
        results = []
        print(f"=== {label} ({len(motion_ids)} motion ids) ===")
        for mid in motion_ids:
            r = analyze_motion(
                motion_lib, mid, robot_config, args.asset_root, reference_body_masses,
                model_cache, addr_map_cache, limit_cache,
            )
            results.append(r)
            print(
                f"  motion_id={r['motion_id']:<7} asset={r['asset_id']:<20} "
                f"max_ratio={r['max_ratio']:>6.2f}  frac_violating={r['frac_violating']:>6.3f}  "
                f"worst_dof={r['worst_dof_name']}"
            )
        return results

    treatment_results = run_group(treatment, "Treatment (discover's actual failing motions)")
    control_results = run_group(control, "Control (never-failing clips)")

    def has_violation(r: dict) -> bool:
        return r["max_ratio"] > 1.0

    n_treat_viol = sum(has_violation(r) for r in treatment_results)
    n_control_viol = sum(has_violation(r) for r in control_results)
    p_treat = n_treat_viol / len(treatment_results)
    p_control = n_control_viol / len(control_results)

    print("\n=== Aggregate: treatment vs. control ===")
    print(
        f"Treatment: {n_treat_viol}/{len(treatment_results)} ({100*p_treat:.1f}%) have a "
        f"real torque violation (max_ratio > 1.0)"
    )
    print(
        f"Control:   {n_control_viol}/{len(control_results)} ({100*p_control:.1f}%) have a "
        f"real torque violation (max_ratio > 1.0)"
    )

    n1, n2 = len(treatment_results), len(control_results)
    p_pool = (n_treat_viol + n_control_viol) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p_treat - p_control) / se if se > 0 else float("nan")
    print(f"\nTwo-proportion z-test (treatment vs. control violation rate): z = {z:.2f}")
    print(
        "|z| > ~1.96 => statistically distinguishable at p<0.05. High treatment violation "
        "rate + significant z => supports actuator-saturation hypothesis (H1). Low/similar "
        "rates => supports RL-precision-plateau hypothesis (H2)."
    )


if __name__ == "__main__":
    main()
