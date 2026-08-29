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
#
"""
tools/refine_humos_motion.py

Offline, non-physics refinement of a HUMOS-sourced, packaged MotionLib `.pt` file:
contact-aware footskate correction, per-frame ground-penetration correction, and
contact-aware rotation smoothing. No physics simulator or RL tracker is involved.

Pipeline position: takes the output of `convert_amass_to_motionlib_with_morphology.py`
(Phase 11) and REPLACES `compute_humos_frame0_offsets.py` (Phase 12) — this tool's
per-frame ground-penetration correction subsumes that script's frame-0-only version.

Algorithm:
  1. Validate/detect contacts and clean the four ankle/toe contact masks per motion.
  2. Smooth 3-DOF joint rotations as sign-continuous unit quaternions. Corrections
     taper to zero at contact transitions and are capped by geodesic angle.
  3. Re-derive body poses from the smoothed joints using morphology-specific FK.
  4. Reduce stance-foot drift with a conservative root-XY correction. Multiple
     planted contact points are solved in least squares; this is intentionally not
     described as exact foot locking because root translation cannot repair relative
     foot motion or foot pivots without IK.
  5. Apply collision-geometry ground clearance to the resulting final pose, then
     re-run FK and derive velocities separately within each motion.
  6. Save cleaned semantic contacts and corrected tensors in the original structure.
  7. Save: mutate the original loaded dict in place, `torch.save` the same object
     (preserves every other key untouched) as `<name>_refined.pt`.

Example:
    python tools/refine_humos_motion.py \\
        --motion-file data_cache/small150_128shape.pt \\
        --asset-root protomotions/data/assets/mjcf/smpl_mor \\
        --limit 200
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

from data.scripts.contact_detection import compute_contact_labels_from_pos_and_vel
from protomotions.components.collision_geometry import (
    compute_lowest_z_for_shape,
    discover_assets,
    parse_mjcf_collision_shapes,
)
from protomotions.components.pose_lib import (
    KinematicInfo,
    build_body_ids_tensor,
    compute_angular_velocity,
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_transforms_from_qpos_non_root,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quat_diff_norm,
    quat_to_exp_map,
    quaternion_to_matrix,
)

# First-guess defaults, documented as unvalidated in the plan file.
CONTACT_VEL_THRESH = 0.15
CONTACT_HEIGHT_THRESH = 0.1
HYSTERESIS_MAX_GAP = 2
HYSTERESIS_MIN_LEN = 3
ROOT_CORRECTION_SMOOTH_WINDOW = 21
MAX_ROOT_XY_CORRECTION = 0.10
SAVGOL_WINDOW = 9
SAVGOL_POLYORDER = 3
SMOOTH_STRENGTH = 0.5
SMOOTH_TRANSITION_GUARD = 2
MAX_ROTATION_CORRECTION_DEG = 8.0
GROUND_SMOOTH_WINDOW = 5
TARGET_Z = 0.005  # matches compute_humos_frame0_offsets.py's default


def motion_ranges(length_starts: torch.Tensor, motion_num_frames: torch.Tensor):
    """Yield (motion_id, start, end) for every motion."""
    for m in range(length_starts.shape[0]):
        s = int(length_starts[m].item())
        e = s + int(motion_num_frames[m].item())
        yield m, s, e


def clean_contact_mask(mask: np.ndarray, max_gap: int, min_len: int) -> np.ndarray:
    """Merge short gaps, then drop short windows, in a 1D boolean contact signal."""
    mask = mask.copy()
    T = len(mask)

    # Merge gaps <= max_gap between two True runs.
    i = 0
    while i < T:
        if not mask[i]:
            j = i
            while j < T and not mask[j]:
                j += 1
            gap_len = j - i
            if 0 < i and j < T and gap_len <= max_gap:
                mask[i:j] = True
            i = j
        else:
            i += 1

    # Drop True runs shorter than min_len.
    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if (j - i) < min_len:
                mask[i:j] = False
            i = j
        else:
            i += 1

    return mask


def contact_windows(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a boolean 1D mask into a list of [start, end) windows."""
    windows = []
    i = 0
    T = len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            windows.append((i, j))
            i = j
        else:
            i += 1
    return windows


def smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving-average smoothing (used to crossfade the footskate
    correction back to zero outside contact windows)."""
    if window <= 1 or len(signal) < 2:
        return signal
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(signal)]


def smooth_clearance_lift(signal: np.ndarray, window: int) -> np.ndarray:
    """Build a smooth lift envelope that is never smaller than required clearance."""
    if window <= 1 or len(signal) < 2:
        return signal
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    envelope = windows.max(axis=-1)[: len(signal)]
    return np.maximum(signal, smooth_1d(envelope, window))


def get_dof_slice(kinematic_info: KinematicInfo, body_name: str) -> slice:
    """DOF-channel slice (into `dps`) for a given body, in the same order
    `extract_transforms_from_qpos_non_root` consumes `dps`."""
    body_idx = kinematic_info.body_names.index(body_name)
    start = 0
    for b_idx, axes in kinematic_info.hinge_axes_map.items():
        n = len(axes)
        if b_idx == body_idx:
            return slice(start, start + n)
        start += n
    raise KeyError(f"Body {body_name} has no DOFs in hinge_axes_map")


def step1_contact_masks(
    gts: torch.Tensor,
    gvs: torch.Tensor,
    contacts: torch.Tensor | None,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    foot_ids: torch.Tensor,
) -> Tuple[Dict[int, np.ndarray], torch.Tensor]:
    """Return cleaned ankle/toe masks and the semantic contact tensor to save.

    Stored contacts are treated as desired contact timing, but only after validating
    their shape and thresholding non-binary values. Missing, malformed, non-finite,
    or all-zero contacts are recomputed from the input motion.
    """
    expected_shape = gts.shape[:2]
    shape_is_valid = (
        contacts is not None
        and contacts.ndim == 2
        and tuple(contacts.shape) == tuple(expected_shape)
        and (not contacts.is_floating_point() or bool(torch.isfinite(contacts).all()))
    )
    semantic_contacts = None
    if shape_is_valid:
        candidate = contacts.bool() if contacts.dtype == torch.bool else contacts >= 0.5
        if bool(candidate.any()):
            semantic_contacts = candidate

    if semantic_contacts is not None:
        print("[STEP1] using stored contacts after validation")
    else:
        print(
            "[STEP1] recomputing contacts "
            "(missing, malformed, non-finite, or empty after thresholding)"
        )
        semantic_contacts = compute_contact_labels_from_pos_and_vel(
            gts,
            gvs,
            vel_thres=CONTACT_VEL_THRESH,
            height_thresh=CONTACT_HEIGHT_THRESH,
        ).bool()

    semantic_contacts = semantic_contacts.clone()
    raw_foot_contacts = semantic_contacts[:, foot_ids].cpu().numpy().astype(bool)
    masks_by_motion: Dict[int, np.ndarray] = {}

    for m, s, e in motion_ranges(length_starts, motion_num_frames):
        cleaned = np.zeros((e - s, foot_ids.numel()), dtype=bool)
        for contact_idx in range(foot_ids.numel()):
            cleaned[:, contact_idx] = clean_contact_mask(
                raw_foot_contacts[s:e, contact_idx],
                HYSTERESIS_MAX_GAP,
                HYSTERESIS_MIN_LEN,
            )
        masks_by_motion[m] = cleaned
        semantic_contacts[s:e, foot_ids] = torch.as_tensor(
            cleaned, dtype=torch.bool, device=semantic_contacts.device
        )

    return masks_by_motion, semantic_contacts


def step4_root_drift_correction(
    gts: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    foot_contact_masks: Dict[int, np.ndarray],
    foot_anchor_ids: torch.Tensor,
) -> None:
    """Reduce stance-foot XY drift using a smooth root-only least-squares shift.

    Each cleaned ankle/toe contact window contributes a desired translation toward
    that contact point's median location. Simultaneous constraints are averaged.
    The result is deliberately smoothed and capped: this improves global drift but
    does not pretend to solve relative foot motion, foot pivots, or leg IK.
    """
    anchor_ids = [int(v) for v in foot_anchor_ids.cpu().tolist()]
    for m, s, e in tqdm(
        list(motion_ranges(length_starts, motion_num_frames)),
        desc="[STEP4] stance-root correction",
    ):
        T = e - s
        raw_delta = np.zeros((T, 2), dtype=np.float32)
        counts = np.zeros((T,), dtype=np.float32)
        motion_gts = gts[s:e].cpu().numpy()

        for contact_idx, anchor_id in enumerate(anchor_ids):
            for w_start, w_end in contact_windows(foot_contact_masks[m][:, contact_idx]):
                actual_xy = motion_gts[w_start:w_end, anchor_id, :2]
                locked_xy = np.median(actual_xy, axis=0)
                raw_delta[w_start:w_end] += locked_xy - actual_xy
                counts[w_start:w_end] += 1.0

        constrained = counts > 0
        raw_delta[constrained] /= counts[constrained, None]
        smoothed_delta = np.stack(
            [
                smooth_1d(raw_delta[:, 0], ROOT_CORRECTION_SMOOTH_WINDOW),
                smooth_1d(raw_delta[:, 1], ROOT_CORRECTION_SMOOTH_WINDOW),
            ],
            axis=-1,
        )

        magnitude = np.linalg.norm(smoothed_delta, axis=-1, keepdims=True)
        scale = np.minimum(1.0, MAX_ROOT_XY_CORRECTION / np.maximum(magnitude, 1e-8))
        smoothed_delta *= scale

        delta_t = torch.as_tensor(smoothed_delta, dtype=gts.dtype, device=gts.device)
        gts[s:e, :, :2] += delta_t.unsqueeze(1)


def step5_ground_penetration_correction(
    gts: torch.Tensor,
    grs: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    motion_asset_ids: Tuple[str, ...],
    asset_index: Dict[str, "AssetEntry"],  # noqa: F821 (forward ref to collision_geometry.AssetEntry)
    body_name_to_idx: Dict[str, int],
    shapes_cache: Dict[str, list],
) -> None:
    """In-place: lift gts[..., 2] per shape so no frame penetrates the floor."""
    asset_to_motions: Dict[str, List[int]] = defaultdict(list)
    for m in range(length_starts.shape[0]):
        asset_to_motions[motion_asset_ids[m]].append(m)

    for asset_id, motion_ids in tqdm(
        asset_to_motions.items(), desc="[STEP5] ground clearance"
    ):
        if asset_id not in asset_index:
            print(f"[STEP5][WARN] missing asset XML for {asset_id}, skipping")
            continue

        shapes = shapes_cache.setdefault(
            asset_id, parse_mjcf_collision_shapes(asset_index[asset_id].xml_path)
        )

        ranges = []
        idx_chunks = []
        for m in motion_ids:
            s = int(length_starts[m].item())
            e = s + int(motion_num_frames[m].item())
            ranges.append((m, s, e))
            idx_chunks.append(torch.arange(s, e, device=gts.device))
        frame_idx = torch.cat(idx_chunks)

        body_pos = gts[frame_idx]
        body_rot = grs[frame_idx]

        lowest_z = compute_lowest_z_for_shape(
            body_pos, body_rot, shapes, body_name_to_idx
        )
        raw_lift = torch.clamp(TARGET_Z - lowest_z, min=0.0)
        lift = raw_lift.clone()
        offset = 0
        for _, s, e in ranges:
            count = e - s
            smooth_np = smooth_clearance_lift(
                raw_lift[offset : offset + count].detach().cpu().numpy(),
                GROUND_SMOOTH_WINDOW,
            )
            lift[offset : offset + count] = torch.as_tensor(
                smooth_np, dtype=gts.dtype, device=gts.device
            )
            offset += count

        gts[frame_idx, :, 2] += lift.unsqueeze(-1)


def step2_smooth_dps(
    dps: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    foot_contact_masks: Dict[int, np.ndarray],
    kinematic_info: KinematicInfo,
) -> None:
    """Smooth SMPL joint rotations on the quaternion manifold, in place.

    HUMOS/SMPL stores every non-root 3-DOF rotation as an exponential map. We
    convert those rotations to sign-continuous quaternions, apply Savitzky-Golay
    to quaternion components, renormalize, and blend conservatively back toward
    the original. The correction is zero at contact-state boundaries and capped
    by geodesic angle, avoiding raw exponential-map wraparound artifacts.
    """
    dofs_per_body = [len(axes) for axes in kinematic_info.hinge_axes_map.values()]
    if not dofs_per_body or any(n != 3 for n in dofs_per_body):
        raise ValueError(
            "Quaternion refinement requires the SMPL layout with three exponential-map "
            f"DOFs per non-root body; got {dofs_per_body}"
        )
    if dps.shape[1] != 3 * len(dofs_per_body):
        raise ValueError(
            f"dps has {dps.shape[1]} channels, expected {3 * len(dofs_per_body)}"
        )

    max_correction = np.deg2rad(MAX_ROTATION_CORRECTION_DEG)
    for m, s, e in tqdm(
        list(motion_ranges(length_starts, motion_num_frames)),
        desc="[STEP2] SO(3) smoothing",
    ):
        T = e - s
        foot_mask = foot_contact_masks[m]
        left_state = foot_mask[:, :2].any(axis=1)
        right_state = foot_mask[:, 2:].any(axis=1)
        combined = left_state.astype(np.int8) * 2 + right_state.astype(np.int8)
        change_points = np.where(np.diff(combined) != 0)[0] + 1
        boundaries = [0] + change_points.tolist() + [T]

        exp_maps = dps[s:e].reshape(T, -1, 3)
        original_q = axis_angle_to_quaternion(exp_maps, w_last=True)
        continuous_q = original_q.clone()
        if T > 1:
            adjacent_dot = (continuous_q[1:] * continuous_q[:-1]).sum(dim=-1)
            step_sign = torch.where(
                adjacent_dot < 0,
                -torch.ones_like(adjacent_dot),
                torch.ones_like(adjacent_dot),
            )
            sign_parity = torch.cumprod(step_sign, dim=0)
            continuous_q[1:] *= sign_parity.unsqueeze(-1)

        smoothed_q = continuous_q.clone()
        for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
            seg_len = seg_end - seg_start
            window = min(SAVGOL_WINDOW, seg_len if seg_len % 2 == 1 else seg_len - 1)
            if window < SAVGOL_POLYORDER + 2:
                continue

            filtered_np = savgol_filter(
                continuous_q[seg_start:seg_end].detach().cpu().numpy(),
                window_length=window,
                polyorder=SAVGOL_POLYORDER,
                axis=0,
            )
            filtered = torch.as_tensor(
                filtered_np, dtype=dps.dtype, device=dps.device
            )
            filtered /= torch.clamp(filtered.norm(dim=-1, keepdim=True), min=1e-8)

            reference = continuous_q[seg_start:seg_end]
            dot = (reference * filtered).sum(dim=-1, keepdim=True)
            filtered = torch.where(dot < 0, -filtered, filtered)
            dot = (reference * filtered).sum(dim=-1).clamp(-1.0, 1.0)
            correction_angle = 2.0 * torch.acos(dot.abs())

            edge_distance = np.minimum(
                np.arange(seg_len), np.arange(seg_len)[::-1]
            ).astype(np.float32)
            taper = np.clip(
                edge_distance / max(SMOOTH_TRANSITION_GUARD, 1), 0.0, 1.0
            )
            base_alpha = torch.as_tensor(
                taper * SMOOTH_STRENGTH, dtype=dps.dtype, device=dps.device
            ).unsqueeze(-1)
            angle_cap_alpha = max_correction / torch.clamp(
                correction_angle, min=1e-8
            )
            alpha = torch.minimum(base_alpha, angle_cap_alpha).unsqueeze(-1)

            # Shortest-path normalized lerp is stable here because corrections are
            # small and explicitly angle-capped.
            blended = (1.0 - alpha) * reference + alpha * filtered
            blended /= torch.clamp(blended.norm(dim=-1, keepdim=True), min=1e-8)
            smoothed_q[seg_start:seg_end] = blended

        # Canonicalize quaternion signs before mapping back to the principal
        # exponential-map branch used by MotionLib.
        smoothed_q = torch.where(smoothed_q[..., 3:4] < 0, -smoothed_q, smoothed_q)
        dps[s:e] = quat_to_exp_map(smoothed_q, w_last=True).reshape(T, -1)


def step3_rederive_tensors(
    gts: torch.Tensor,
    grs: torch.Tensor,
    gvs: torch.Tensor,
    gavs: torch.Tensor,
    dps: torch.Tensor,
    dvs: torch.Tensor,
    lrs: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    motion_dt: torch.Tensor,
    motion_asset_ids: Tuple[str, ...],
    asset_index: Dict[str, "AssetEntry"],  # noqa: F821
    kinematic_info_cache: Dict[str, KinematicInfo],
    recompute_velocities: bool,
) -> None:
    """Regenerate morphology-specific body poses and, optionally, velocities."""
    asset_to_motions: Dict[str, List[int]] = defaultdict(list)
    for m in range(length_starts.shape[0]):
        asset_to_motions[motion_asset_ids[m]].append(m)

    pass_name = (
        "[STEP6] final FK + velocities"
        if recompute_velocities
        else "[STEP3] post-smoothing FK"
    )
    for asset_id, motion_ids in tqdm(
        asset_to_motions.items(), desc=pass_name
    ):
        if asset_id not in asset_index:
            raise KeyError(f"Missing asset XML for {asset_id}")

        kinematic_info = kinematic_info_cache.setdefault(
            asset_id, extract_kinematic_info(str(asset_index[asset_id].xml_path))
        )

        ranges = []
        idx_chunks = []
        for m in motion_ids:
            s = int(length_starts[m].item())
            e = s + int(motion_num_frames[m].item())
            ranges.append((m, s, e))
            idx_chunks.append(torch.arange(s, e, device=gts.device))
        frame_idx = torch.cat(idx_chunks)

        root_pos = gts[frame_idx, 0, :]
        root_rot_mat = quaternion_to_matrix(grs[frame_idx, 0, :], w_last=True)
        joint_rot_mats = extract_transforms_from_qpos_non_root(
            kinematic_info, dps[frame_idx], qpos_is_exp_map_on_3dof_joints=True
        )
        joint_rot_mats[:, 0, :, :] = root_rot_mat

        state = fk_from_transforms_with_velocities(
            kinematic_info, root_pos, joint_rot_mats,
            fps=None, compute_velocities=False,
        )
        gts[frame_idx] = state.rigid_body_pos
        grs[frame_idx] = state.rigid_body_rot
        lrs[frame_idx] = matrix_to_quaternion(joint_rot_mats, w_last=True)

        if not recompute_velocities:
            continue

        # Derive velocities only within contiguous motions. The multi-horizon
        # convention matches convert_amass_to_proto.py; reporting uses exact
        # one-frame derivatives so this filtering cannot hide residual jitter.
        for m, s, e in ranges:
            fps_m = round(1.0 / motion_dt[m].item())
            gvs[s:e] = compute_cartesian_velocity(
                gts[s:e], fps_m, velocity_max_horizon=3
            )
            grs_mat_m = quaternion_to_matrix(grs[s:e], w_last=True)
            gavs[s:e] = compute_angular_velocity(
                grs_mat_m, fps_m, velocity_max_horizon=3
            )
            local_rot_mats_m = extract_transforms_from_qpos_non_root(
                kinematic_info, dps[s:e], qpos_is_exp_map_on_3dof_joints=True
            )
            ang_vel = compute_angular_velocity(
                local_rot_mats_m[:, 1:, :, :], fps=fps_m
            )
            dvs[s:e] = ang_vel.reshape(e - s, -1)


def _stance_speeds(
    gts: torch.Tensor,
    foot_ids: torch.Tensor,
    foot_contact_masks: Dict[int, np.ndarray],
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    motion_dt: torch.Tensor,
) -> torch.Tensor:
    """Exact one-frame horizontal contact-point speeds, without velocity filtering."""
    values = []
    for m, s, e in motion_ranges(length_starts, motion_num_frames):
        if e - s < 2:
            continue
        dt = float(motion_dt[m].item())
        pos = gts[s:e, foot_ids, :2]
        speed = (pos[1:] - pos[:-1]).norm(dim=-1) / dt
        mask_np = foot_contact_masks[m]
        active_np = mask_np[1:] & mask_np[:-1]
        active = torch.as_tensor(active_np, dtype=torch.bool, device=gts.device)
        if bool(active.any()):
            values.append(speed[active])
    if not values:
        return torch.empty(0, dtype=gts.dtype, device=gts.device)
    return torch.cat(values)


def _per_motion_position_jerk(
    gts: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    motion_dt: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean all-body and root jerk for each motion, respecting clip boundaries."""
    all_body, root = [], []
    for m, s, e in motion_ranges(length_starts, motion_num_frames):
        if e - s < 4:
            continue
        dt3 = float(motion_dt[m].item()) ** 3
        segment = gts[s:e]
        d3 = segment[3:] - 3 * segment[2:-1] + 3 * segment[1:-2] - segment[:-3]
        jerk = d3.norm(dim=-1) / dt3
        all_body.append(float(jerk.mean().item()))
        root.append(float(jerk[:, 0].mean().item()))
    return np.asarray(all_body), np.asarray(root)


def _summary(values: torch.Tensor) -> str:
    if values.numel() == 0:
        return "n/a"
    return (
        f"mean={values.mean().item():.4f} "
        f"p95={values.quantile(0.95).item():.4f} "
        f"max={values.max().item():.4f}"
    )


def print_report(
    name: str,
    gts_before: torch.Tensor,
    gts_after: torch.Tensor,
    grs_before: torch.Tensor,
    grs_after: torch.Tensor,
    dps_before: torch.Tensor,
    dps_after: torch.Tensor,
    foot_ids: torch.Tensor,
    foot_contact_masks: Dict[int, np.ndarray],
    shapes_cache: Dict[str, list],
    asset_index: Dict[str, "AssetEntry"],  # noqa: F821
    body_name_to_idx: Dict[str, int],
    motion_asset_ids: Tuple[str, ...],
    motion_clip_ids: Tuple[str, ...] | None,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    motion_dt: torch.Tensor,
    kinematic_info: KinematicInfo,
) -> None:
    print(f"\n===== {name} =====")

    speed_before = _stance_speeds(
        gts_before, foot_ids, foot_contact_masks,
        length_starts, motion_num_frames, motion_dt,
    )
    speed_after = _stance_speeds(
        gts_after, foot_ids, foot_contact_masks,
        length_starts, motion_num_frames, motion_dt,
    )
    print(
        "[footskate] exact horizontal speed on cleaned semantic-contact intervals (m/s):\n"
        f"  before: {_summary(speed_before)}\n"
        f"  after:  {_summary(speed_after)}"
    )

    asset_to_motions: Dict[str, List[int]] = defaultdict(list)
    for m, asset_id in enumerate(motion_asset_ids):
        asset_to_motions[asset_id].append(m)
    pen_before_count = pen_after_count = total_frames = 0
    max_depth_before = max_depth_after = 0.0
    for asset_id, motion_ids in asset_to_motions.items():
        if asset_id not in asset_index or asset_id not in shapes_cache:
            continue
        idx = torch.cat(
            [
                torch.arange(
                    int(length_starts[m].item()),
                    int(length_starts[m].item()) + int(motion_num_frames[m].item()),
                    device=gts_after.device,
                )
                for m in motion_ids
            ]
        )
        shapes = shapes_cache[asset_id]
        lowest_before = compute_lowest_z_for_shape(
            gts_before[idx], grs_before[idx], shapes, body_name_to_idx
        )
        lowest_after = compute_lowest_z_for_shape(
            gts_after[idx], grs_after[idx], shapes, body_name_to_idx
        )
        pen_before_count += int((lowest_before < 0).sum().item())
        pen_after_count += int((lowest_after < 0).sum().item())
        total_frames += int(idx.numel())
        max_depth_before = max(
            max_depth_before, float((-lowest_before).clamp(min=0).max().item())
        )
        max_depth_after = max(
            max_depth_after, float((-lowest_after).clamp(min=0).max().item())
        )
    if total_frames:
        print(
            f"[penetration] all {len(asset_to_motions)} represented shapes: "
            f"frame fraction {pen_before_count / total_frames:.6f} -> "
            f"{pen_after_count / total_frames:.6f}; max depth "
            f"{max_depth_before:.4f} -> {max_depth_after:.4f} m"
        )

    for label, gts_ in (("before", gts_before), ("after", gts_after)):
        body_jerk, root_jerk = _per_motion_position_jerk(
            gts_, length_starts, motion_num_frames, motion_dt
        )
        if body_jerk.size:
            print(
                f"[jerk:{label}] per-motion mean, median/p95 (m/s^3): "
                f"all bodies={np.median(body_jerk):.2f}/{np.quantile(body_jerk, 0.95):.2f}, "
                f"root={np.median(root_jerk):.2f}/{np.quantile(root_jerk, 0.95):.2f}"
            )

    rotation_means, rotation_p95s, rotation_maxima = [], [], []
    root_xy_means, root_xy_p95s, root_z_means = [], [], []
    for _, s, e in motion_ranges(length_starts, motion_num_frames):
        before_q = axis_angle_to_quaternion(
            dps_before[s:e].reshape(e - s, -1, 3), w_last=True
        )
        after_q = axis_angle_to_quaternion(
            dps_after[s:e].reshape(e - s, -1, 3), w_last=True
        )
        angle_deg = torch.rad2deg(
            quat_diff_norm(before_q, after_q, w_last=True)
        ).reshape(-1)
        rotation_means.append(float(angle_deg.mean().item()))
        rotation_p95s.append(float(angle_deg.quantile(0.95).item()))
        rotation_maxima.append(float(angle_deg.max().item()))

        root_delta = gts_after[s:e, 0] - gts_before[s:e, 0]
        root_xy = root_delta[:, :2].norm(dim=-1)
        root_xy_means.append(float(root_xy.mean().item()))
        root_xy_p95s.append(float(root_xy.quantile(0.95).item()))
        root_z_means.append(float(root_delta[:, 2].abs().mean().item()))
    print(
        "[fidelity] local-rotation geodesic change, aggregate of per-motion stats: "
        f"mean={np.mean(rotation_means):.3f}deg "
        f"p95={np.quantile(rotation_p95s, 0.95):.3f}deg "
        f"max={np.max(rotation_maxima):.3f}deg"
    )
    print(
        "[root-change] aggregate of per-motion displacement: "
        f"XY mean={np.mean(root_xy_means):.4f}m "
        f"XY p95={np.quantile(root_xy_p95s, 0.95):.4f}m "
        f"|Z| mean={np.mean(root_z_means):.4f}m"
    )

    if motion_clip_ids is None:
        print("[shape-signal] no clip identity metadata; skipped")
        return
    try:
        knee_slice = get_dof_slice(kinematic_info, "L_Knee")
        grouped_before: Dict[str, List[float]] = defaultdict(list)
        grouped_after: Dict[str, List[float]] = defaultdict(list)
        for m, s, e in motion_ranges(length_starts, motion_num_frames):
            key = str(motion_clip_ids[m])
            before = dps_before[s:e, knee_slice].norm(dim=-1)
            after = dps_after[s:e, knee_slice].norm(dim=-1)
            grouped_before[key].append(float((before.max() - before.min()).item()))
            grouped_after[key].append(float((after.max() - after.min()).item()))
        variances_before, variances_after = [], []
        for key in grouped_before:
            if len(grouped_before[key]) < 2:
                continue
            variances_before.append(float(np.var(grouped_before[key])))
            variances_after.append(float(np.var(grouped_after[key])))
        if variances_before:
            print(
                f"[shape-signal] within-clip L_Knee range variance over "
                f"{len(variances_before)} clip groups, median: "
                f"before={np.median(variances_before):.6f} "
                f"after={np.median(variances_after):.6f}"
            )
        else:
            print("[shape-signal] no clip has multiple shape variants; skipped")
    except KeyError:
        print("[shape-signal] L_Knee not found in kinematic_info; skipped")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/smpl_mor"),
    )
    parser.add_argument("--out-motion-file", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--limit", type=int, default=-1, help="Only process the first N motions (for quick smoke-testing)."
    )
    parser.add_argument("--report", action="store_true", help="Print before/after metrics.")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[LOAD] {args.motion_file}")
    data = torch.load(args.motion_file, map_location="cpu", weights_only=False)

    num_motions = data["length_starts"].shape[0]
    if args.limit > 0 and args.limit < num_motions:
        n = args.limit
        frame_end = int((data["length_starts"][n - 1] + data["motion_num_frames"][n - 1]).item())
        # NOTE: `tensor[:k]` is a view sharing the ORIGINAL storage. torch.save on a
        # view serializes the full underlying buffer, not just the sliced range, so
        # `.clone()` is required here to actually shrink the file on disk.
        for key in ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs"):
            if data.get(key) is not None:
                data[key] = data[key][:frame_end].clone()
        for key in (
            "length_starts", "motion_lengths", "motion_dt", "motion_num_frames",
            "motion_weights", "motion_betas", "motion_gender_ids", "motion_source_id",
        ):
            if data.get(key) is not None:
                data[key] = data[key][:n].clone()
        for key in (
            "motion_asset_ids", "motion_beta_keys", "motion_genders", "motion_clip_ids",
            "motion_base_clip_ids", "motion_files", "motion_npz_files",
        ):
            if data.get(key) is not None:
                data[key] = data[key][:n]  # plain tuples: slicing already copies
        num_motions = n
        print(f"[LOAD] --limit applied: {num_motions} motions, {frame_end} frames")

    gts = data["gts"].to(device)
    grs = data["grs"].to(device)
    gvs = data["gvs"].to(device)
    gavs = data["gavs"].to(device)
    dps = data["dps"].to(device)
    dvs = data["dvs"].to(device)
    lrs = data["lrs"].to(device) if data.get("lrs") is not None else torch.zeros_like(grs)
    contacts = data.get("contacts")
    if contacts is not None:
        contacts = contacts.to(device)
    length_starts = data["length_starts"].to(device)
    motion_num_frames = data["motion_num_frames"].to(device)
    motion_dt = data["motion_dt"]
    motion_asset_ids = tuple(data["motion_asset_ids"])

    if args.report:
        gts_before = gts.clone()
        grs_before = grs.clone()
        dps_before = dps.clone()
    else:
        gts_before = grs_before = dps_before = None

    asset_index = discover_assets(args.asset_root)

    # Body topology (name->index and foot indices) is identical across smpl_mor
    # shapes; morphology-specific offsets and geometry are still loaded per asset.
    any_xml = next(iter(asset_index.values())).xml_path
    ref_kinematic_info = extract_kinematic_info(str(any_xml))
    body_name_to_idx = {
        name: i for i, name in enumerate(ref_kinematic_info.body_names)
    }
    left_foot_ids = build_body_ids_tensor(
        ref_kinematic_info.body_names, ["L_Ankle", "L_Toe"], device
    )
    right_foot_ids = build_body_ids_tensor(
        ref_kinematic_info.body_names, ["R_Ankle", "R_Toe"], device
    )
    foot_ids = torch.cat([left_foot_ids, right_foot_ids])

    print("[STEP1] validating/detecting contacts")
    foot_contact_masks, contacts = step1_contact_masks(
        gts, gvs, contacts, length_starts, motion_num_frames, foot_ids
    )

    step2_smooth_dps(
        dps,
        length_starts,
        motion_num_frames,
        foot_contact_masks,
        ref_kinematic_info,
    )

    kinematic_info_cache: Dict[str, KinematicInfo] = {}
    step3_rederive_tensors(
        gts, grs, gvs, gavs, dps, dvs, lrs,
        length_starts, motion_num_frames, motion_dt,
        motion_asset_ids, asset_index, kinematic_info_cache,
        recompute_velocities=False,
    )

    step4_root_drift_correction(
        gts,
        length_starts,
        motion_num_frames,
        foot_contact_masks,
        foot_ids,
    )

    shapes_cache: Dict[str, list] = {}
    step5_ground_penetration_correction(
        gts, grs, length_starts, motion_num_frames,
        motion_asset_ids, asset_index, body_name_to_idx, shapes_cache,
    )

    step3_rederive_tensors(
        gts, grs, gvs, gavs, dps, dvs, lrs,
        length_starts, motion_num_frames, motion_dt,
        motion_asset_ids, asset_index, kinematic_info_cache,
        recompute_velocities=True,
    )

    for tensor_name, tensor in (
        ("gts", gts), ("grs", grs), ("gvs", gvs), ("gavs", gavs),
        ("dps", dps), ("dvs", dvs), ("lrs", lrs),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Refinement produced non-finite values in {tensor_name}")

    if args.report:
        motion_clip_ids = data.get("motion_base_clip_ids")
        if motion_clip_ids is None:
            motion_clip_ids = data.get("motion_clip_ids")
        if motion_clip_ids is not None:
            motion_clip_ids = tuple(motion_clip_ids)
        print_report(
            args.motion_file.name,
            gts_before, gts, grs_before, grs, dps_before, dps,
            foot_ids, foot_contact_masks,
            shapes_cache, asset_index, body_name_to_idx,
            motion_asset_ids, motion_clip_ids,
            length_starts, motion_num_frames, motion_dt,
            ref_kinematic_info,
        )

    data["gts"] = gts.cpu()
    data["grs"] = grs.cpu()
    data["gvs"] = gvs.cpu()
    data["gavs"] = gavs.cpu()
    data["dps"] = dps.cpu()
    data["dvs"] = dvs.cpu()
    data["lrs"] = lrs.cpu()
    data["contacts"] = contacts.cpu()

    out_path = args.out_motion_file or args.motion_file.with_name(
        f"{args.motion_file.stem}_refined{args.motion_file.suffix}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(data, tmp_path)
    tmp_path.replace(out_path)
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
