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
contact-segmented joint-angle smoothing. No physics simulator, no RL tracker, no
naive uniform low-pass filtering. Design rationale and validated implementation
details: /home/hlz/.claude/plans/shiny-mapping-thimble.md.

Pipeline position: takes the output of `convert_amass_to_motionlib_with_morphology.py`
(Phase 11) and REPLACES `compute_humos_frame0_offsets.py` (Phase 12) — this tool's
per-frame ground-penetration correction subsumes that script's frame-0-only version.

Algorithm (see plan file for full justification):
  1. Contact detection (per foot, from gts/gvs; reuse `contacts` if present/valid).
  2. Footskate correction: per contact window, lock the foot's XY position by
     shifting `gts[..., :2]` for the whole motion (root-translation-only correction,
     no FK/IK needed).
  3. Ground-penetration correction: per shape, batched pure-tensor lowest-collision-Z
     query (protomotions.components.collision_geometry) against every frame, lifting
     `gts[..., 2]` to clear the floor.
  4. Jitter smoothing: Savitzky-Golay filter on `dps` (DOF angles), segmented at
     contact-state-change frames so smoothing never spans a foot-plant transition.
  5. Re-derive gts/grs/gvs/gavs/dvs from [corrected root pos, original root rot,
     smoothed dps] via a single per-shape FK pass, so the whole tensor set stays
     internally consistent.
  6. Save: mutate the original loaded dict in place, `torch.save` the same object
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
from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix

# First-guess defaults, documented as unvalidated in the plan file.
CONTACT_VEL_THRESH = 0.15
CONTACT_HEIGHT_THRESH = 0.1
HYSTERESIS_MAX_GAP = 2
HYSTERESIS_MIN_LEN = 3
CROSSFADE_FRAMES = 5
SAVGOL_WINDOW = 9
SAVGOL_POLYORDER = 3
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


def step1_contact_windows(
    gts: torch.Tensor,
    gvs: torch.Tensor,
    contacts: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    left_foot_ids: torch.Tensor,
    right_foot_ids: torch.Tensor,
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
    """Per-motion contact windows for the left and right foot (ankle OR toe)."""
    if contacts is None or not bool(contacts.any()):
        print("[STEP1] recomputing contacts (none present / all-zero)")
        contacts = compute_contact_labels_from_pos_and_vel(
            gts, gvs, vel_thres=CONTACT_VEL_THRESH, height_thresh=CONTACT_HEIGHT_THRESH
        )

    left_contact = contacts[:, left_foot_ids].any(dim=-1).cpu().numpy()
    right_contact = contacts[:, right_foot_ids].any(dim=-1).cpu().numpy()

    left_windows: Dict[int, List[Tuple[int, int]]] = {}
    right_windows: Dict[int, List[Tuple[int, int]]] = {}

    for m, s, e in motion_ranges(length_starts, motion_num_frames):
        left_m = clean_contact_mask(left_contact[s:e], HYSTERESIS_MAX_GAP, HYSTERESIS_MIN_LEN)
        right_m = clean_contact_mask(right_contact[s:e], HYSTERESIS_MAX_GAP, HYSTERESIS_MIN_LEN)
        left_windows[m] = contact_windows(left_m)
        right_windows[m] = contact_windows(right_m)

    return left_windows, right_windows


def step2_footskate_correction(
    gts: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    left_windows: Dict[int, List[Tuple[int, int]]],
    right_windows: Dict[int, List[Tuple[int, int]]],
    left_anchor_id: int,
    right_anchor_id: int,
) -> None:
    """In-place: shift gts[..., :2] per motion to lock each foot during contact."""
    for m, s, e in tqdm(
        list(motion_ranges(length_starts, motion_num_frames)), desc="[STEP2] footskate"
    ):
        T = e - s
        raw_delta = np.zeros((T, 2), dtype=np.float32)
        counts = np.zeros((T,), dtype=np.float32)

        motion_gts = gts[s:e].cpu().numpy()

        for windows, anchor_id in (
            (left_windows[m], left_anchor_id),
            (right_windows[m], right_anchor_id),
        ):
            for w_start, w_end in windows:
                actual_xy = motion_gts[w_start:w_end, anchor_id, :2]
                locked_xy = np.median(actual_xy, axis=0)
                raw_delta[w_start:w_end] += locked_xy - actual_xy
                counts[w_start:w_end] += 1.0

        nonzero = counts > 0
        raw_delta[nonzero] /= counts[nonzero, None]

        smoothed_delta = np.stack(
            [smooth_1d(raw_delta[:, 0], CROSSFADE_FRAMES), smooth_1d(raw_delta[:, 1], CROSSFADE_FRAMES)],
            axis=-1,
        )

        delta_t = torch.as_tensor(smoothed_delta, dtype=gts.dtype, device=gts.device)
        gts[s:e, :, :2] += delta_t.unsqueeze(1)


def step3_ground_penetration_correction(
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
        asset_to_motions.items(), desc="[STEP3] ground penetration"
    ):
        if asset_id not in asset_index:
            print(f"[STEP3][WARN] missing asset XML for {asset_id}, skipping")
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

        lowest_z = compute_lowest_z_for_shape(body_pos, body_rot, shapes, body_name_to_idx)
        lift = torch.clamp(TARGET_Z - lowest_z, min=0.0)

        gts[frame_idx, :, 2] += lift.unsqueeze(-1)


def step4_smooth_dps(
    dps: torch.Tensor,
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    left_windows: Dict[int, List[Tuple[int, int]]],
    right_windows: Dict[int, List[Tuple[int, int]]],
) -> None:
    """In-place: Savitzky-Golay smooth dps, segmented at contact-state changes."""
    for m, s, e in tqdm(
        list(motion_ranges(length_starts, motion_num_frames)), desc="[STEP4] smoothing"
    ):
        T = e - s
        left_state = np.zeros(T, dtype=bool)
        right_state = np.zeros(T, dtype=bool)
        for w_start, w_end in left_windows[m]:
            left_state[w_start:w_end] = True
        for w_start, w_end in right_windows[m]:
            right_state[w_start:w_end] = True

        combined = left_state.astype(np.int8) * 2 + right_state.astype(np.int8)
        change_points = np.where(np.diff(combined) != 0)[0] + 1
        boundaries = [0] + change_points.tolist() + [T]

        motion_dps = dps[s:e].cpu().numpy()

        for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
            seg_len = seg_end - seg_start
            window = min(SAVGOL_WINDOW, seg_len if seg_len % 2 == 1 else seg_len - 1)
            if window < SAVGOL_POLYORDER + 2:
                continue
            motion_dps[seg_start:seg_end] = savgol_filter(
                motion_dps[seg_start:seg_end], window_length=window,
                polyorder=SAVGOL_POLYORDER, axis=0,
            )

        dps[s:e] = torch.as_tensor(motion_dps, dtype=dps.dtype, device=dps.device)


def step5_rederive_tensors(
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
) -> None:
    """In-place: regenerate gts/grs (all bodies) + gvs/gavs/dvs/lrs from
    [corrected root pos, original root rot, smoothed dps], per shape."""
    asset_to_motions: Dict[str, List[int]] = defaultdict(list)
    for m in range(length_starts.shape[0]):
        asset_to_motions[motion_asset_ids[m]].append(m)

    for asset_id, motion_ids in tqdm(asset_to_motions.items(), desc="[STEP5] FK re-derive"):
        if asset_id not in asset_index:
            continue

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
            kinematic_info, root_pos, joint_rot_mats, fps=None, compute_velocities=False
        )

        gts[frame_idx] = state.rigid_body_pos
        grs[frame_idx] = state.rigid_body_rot
        lrs[frame_idx] = matrix_to_quaternion(joint_rot_mats, w_last=True)

        # Velocities need per-motion contiguous sequences (cross-motion boundaries
        # in `frame_idx` would otherwise produce bogus finite-difference velocities).
        for m, s, e in ranges:
            fps_m = round(1.0 / motion_dt[m].item())

            gvs[s:e] = compute_cartesian_velocity(gts[s:e], fps_m, velocity_max_horizon=3)

            grs_mat_m = quaternion_to_matrix(grs[s:e], w_last=True)
            gavs[s:e] = compute_angular_velocity(grs_mat_m, fps_m, velocity_max_horizon=3)

            local_rot_mats_m = extract_transforms_from_qpos_non_root(
                kinematic_info, dps[s:e], qpos_is_exp_map_on_3dof_joints=True
            )
            ang_vel = compute_angular_velocity(local_rot_mats_m[:, 1:, :, :], fps=fps_m)
            dvs[s:e] = ang_vel.reshape(e - s, -1)


def print_report(
    name: str,
    gts_before: torch.Tensor,
    gts_after: torch.Tensor,
    grs_before: torch.Tensor,
    grs_after: torch.Tensor,
    gvs_before: torch.Tensor,
    gvs_after: torch.Tensor,
    contacts_before: torch.Tensor,
    left_foot_ids: torch.Tensor,
    right_foot_ids: torch.Tensor,
    shapes_cache: Dict[str, list],
    asset_index: Dict[str, "AssetEntry"],  # noqa: F821
    body_name_to_idx: Dict[str, int],
    motion_asset_ids: Tuple[str, ...],
    length_starts: torch.Tensor,
    motion_num_frames: torch.Tensor,
    dps_before: torch.Tensor,
    dps_after: torch.Tensor,
    kinematic_info: KinematicInfo,
) -> None:
    print(f"\n===== {name} =====")

    foot_ids = torch.cat([left_foot_ids, right_foot_ids])
    if contacts_before is not None:
        # Fixed ORIGINAL contact mask, applied to both before/after velocities, so
        # this measures whether the frames we identified as stance got quieter.
        contact_mask = contacts_before[:, foot_ids].bool()
        if contact_mask.any():
            speed_before = gvs_before[:, foot_ids, :2].norm(dim=-1)[contact_mask]
            speed_after = gvs_after[:, foot_ids, :2].norm(dim=-1)[contact_mask]
            print(
                f"[footskate] stance-foot horizontal speed during ORIGINAL contact frames: "
                f"before mean={speed_before.mean().item():.4f} p95={speed_before.quantile(0.95).item():.4f} m/s, "
                f"after mean={speed_after.mean().item():.4f} p95={speed_after.quantile(0.95).item():.4f} m/s"
            )
    else:
        print("[footskate] no input `contacts` tensor available, skipped")

    # Ground penetration: sample a handful of shapes for a quick before/after check
    # (using the SAME asset's grs before/after, since step 3/5 only translate gts;
    # rotation isn't touched by penetration correction).
    sample_assets = list(shapes_cache.keys())[:8] if shapes_cache else []
    pen_before_fracs, pen_after_fracs = [], []
    for asset_id in sample_assets:
        if asset_id not in asset_index:
            continue
        motion_ids = [m for m in range(len(motion_asset_ids)) if motion_asset_ids[m] == asset_id]
        if not motion_ids:
            continue
        idx = torch.cat(
            [
                torch.arange(
                    int(length_starts[m].item()),
                    int(length_starts[m].item()) + int(motion_num_frames[m].item()),
                )
                for m in motion_ids
            ]
        )
        shapes = shapes_cache[asset_id]
        lowest_before = compute_lowest_z_for_shape(gts_before[idx], grs_before[idx], shapes, body_name_to_idx)
        lowest_after = compute_lowest_z_for_shape(gts_after[idx], grs_after[idx], shapes, body_name_to_idx)
        pen_before_fracs.append((lowest_before < 0).float().mean().item())
        pen_after_fracs.append((lowest_after < 0).float().mean().item())

    if pen_before_fracs:
        print(
            f"[penetration] fraction of frames with lowest-Z < 0, sampled over "
            f"{len(pen_before_fracs)} shapes: before={np.mean(pen_before_fracs):.4f} "
            f"after={np.mean(pen_after_fracs):.4f}"
        )

    print("[jerk] third finite-difference of gts, mean over all bodies/frames:")
    for label, gts_ in (("before", gts_before), ("after", gts_after)):
        d3 = gts_[3:] - 3 * gts_[2:-1] + 3 * gts_[1:-2] - gts_[:-3]
        print(f"  {label}: {d3.norm(dim=-1).mean().item():.6f}")

    # Cross-shape variance of a knee-flexion-range summary stat (shape-signal check).
    try:
        l_knee_slice = get_dof_slice(kinematic_info, "L_Knee")
        per_motion_range_before = []
        per_motion_range_after = []
        for m, s, e in motion_ranges(length_starts, motion_num_frames):
            seg_before = dps_before[s:e, l_knee_slice].norm(dim=-1)
            seg_after = dps_after[s:e, l_knee_slice].norm(dim=-1)
            per_motion_range_before.append((seg_before.max() - seg_before.min()).item())
            per_motion_range_after.append((seg_after.max() - seg_after.min()).item())
        pb = np.asarray(per_motion_range_before)
        pa = np.asarray(per_motion_range_after)
        print(
            f"[shape-signal] cross-motion variance of L_Knee exp-map-norm range: "
            f"before={pb.var():.6f} after={pa.var():.6f} "
            f"(should NOT collapse toward 0)"
        )
    except KeyError:
        print("[shape-signal] L_Knee not found in kinematic_info, skipped")


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

    gts_before = gts.clone()
    grs_before = grs.clone()
    gvs_before = gvs.clone()
    dps_before = dps.clone()
    contacts_before = contacts.clone() if contacts is not None else None

    asset_index = discover_assets(args.asset_root)

    # Body topology (name->index, foot indices) is identical across all smpl_mor
    # shapes -- only bone lengths / geometry differ. Extract it once from any shape.
    any_xml = next(iter(asset_index.values())).xml_path
    ref_kinematic_info = extract_kinematic_info(str(any_xml))
    body_name_to_idx = {name: i for i, name in enumerate(ref_kinematic_info.body_names)}

    left_foot_ids = build_body_ids_tensor(
        ref_kinematic_info.body_names, ["L_Ankle", "L_Toe"], device
    )
    right_foot_ids = build_body_ids_tensor(
        ref_kinematic_info.body_names, ["R_Ankle", "R_Toe"], device
    )
    left_anchor_id = body_name_to_idx["L_Ankle"]
    right_anchor_id = body_name_to_idx["R_Ankle"]

    print("[STEP1] detecting contact windows")
    left_windows, right_windows = step1_contact_windows(
        gts, gvs, contacts, length_starts, motion_num_frames, left_foot_ids, right_foot_ids
    )

    step2_footskate_correction(
        gts, length_starts, motion_num_frames, left_windows, right_windows, left_anchor_id, right_anchor_id
    )

    shapes_cache: Dict[str, list] = {}
    step3_ground_penetration_correction(
        gts, grs, length_starts, motion_num_frames, motion_asset_ids, asset_index, body_name_to_idx, shapes_cache
    )

    step4_smooth_dps(dps, length_starts, motion_num_frames, left_windows, right_windows)

    kinematic_info_cache: Dict[str, KinematicInfo] = {}
    step5_rederive_tensors(
        gts, grs, gvs, gavs, dps, dvs, lrs,
        length_starts, motion_num_frames, motion_dt,
        motion_asset_ids, asset_index, kinematic_info_cache,
    )

    if args.report:
        print_report(
            args.motion_file.name,
            gts_before, gts, grs_before, grs, gvs_before, gvs, contacts_before,
            left_foot_ids, right_foot_ids,
            shapes_cache, asset_index, body_name_to_idx,
            motion_asset_ids, length_starts, motion_num_frames,
            dps_before, dps, ref_kinematic_info,
        )

    data["gts"] = gts.cpu()
    data["grs"] = grs.cpu()
    data["gvs"] = gvs.cpu()
    data["gavs"] = gavs.cpu()
    data["dps"] = dps.cpu()
    data["dvs"] = dvs.cpu()
    data["lrs"] = lrs.cpu()

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
