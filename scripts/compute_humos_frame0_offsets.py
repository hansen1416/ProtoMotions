#!/usr/bin/env python3
# scripts/compute_humos_frame0_offsets.py

"""
Apply IsaacGym-computed frame-0 grounding offsets directly to a packaged MotionLib.

Example:
python scripts/compute_humos_frame0_offsets.py \
    --motion-file /home/hlz/datasets/humos_proto/humos_128.pt \
    --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \
    --out-motion-file /home/hlz/datasets/humos_proto/humos_128_offset.pt \
    --limit -1 \
    --overwrite

Input:
    packaged MotionLib .pt file

Output:
    corrected MotionLib .pt file with shifted gts[:, :, 2]
"""

from __future__ import annotations

import argparse
import math
import re
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# IsaacGym must be imported before torch.
from isaacgym import gymapi, gymtorch  # type: ignore

import numpy as np
import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig


ASSET_RE = re.compile(r"^(male|female|neutral)_(.+)_smpl\.xml$")


@dataclass(frozen=True)
class AssetEntry:
    gender: str
    beta_key: str
    asset_id: str
    xml_path: Path


@dataclass
class LocalShape:
    body_name: str
    points: np.ndarray  # [N, 3], body-local points
    radius: float       # capsule/sphere radius; 0 for box


@dataclass
class MotionEntry:
    motion_id: int
    clip_id: str
    gender: str
    beta_key: str
    asset_id: str
    xml_path: Path


def parse_floats(text: str) -> np.ndarray:
    return np.asarray([float(v) for v in text.strip().split()], dtype=np.float32)


def parse_vec_attr(attrs: dict, name: str, default: str) -> np.ndarray:
    return parse_floats(attrs.get(name, default))


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-8)

    w, x, y, z = q

    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def parse_mjcf_collision_shapes(xml_path: Path) -> List[LocalShape]:
    """
    Parse simple collision geometry from MJCF.

    Supported:
        capsule, sphere, box, cylinder

    We attach each geom to its parent body name, then later transform it by
    IsaacGym's rigid-body pose.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    default_geom_attrs = {}
    default_geom = root.find("./default/geom")
    if default_geom is not None:
        default_geom_attrs.update(default_geom.attrib)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"Missing <worldbody> in {xml_path}")

    shapes: List[LocalShape] = []

    for body in worldbody.iter("body"):
        body_name = body.attrib.get("name")
        if body_name is None:
            continue

        for geom in body.findall("geom"):
            attrs = dict(default_geom_attrs)
            attrs.update(geom.attrib)

            geom_type = attrs.get("type", "sphere")

            if geom_type == "plane":
                continue

            if "size" not in attrs:
                continue

            # Skip explicitly non-colliding geoms.
            if attrs.get("contype") == "0" and attrs.get("conaffinity") == "0":
                continue

            size = parse_floats(attrs["size"])
            pos = parse_vec_attr(attrs, "pos", "0 0 0")

            # MJCF quat is wxyz.
            local_rot = quat_wxyz_to_matrix(parse_vec_attr(attrs, "quat", "1 0 0 0"))

            if geom_type == "capsule":
                radius = float(size[0])

                if "fromto" in attrs:
                    points = parse_floats(attrs["fromto"]).reshape(2, 3)
                else:
                    half_len = float(size[1]) if len(size) > 1 else 0.0
                    local_points = np.asarray(
                        [[0.0, 0.0, -half_len], [0.0, 0.0, half_len]],
                        dtype=np.float32,
                    )
                    points = pos.reshape(1, 3) + local_points @ local_rot.T

                shapes.append(
                    LocalShape(
                        body_name=body_name,
                        points=points.astype(np.float32),
                        radius=radius,
                    )
                )

            elif geom_type == "sphere":
                radius = float(size[0])
                points = pos.reshape(1, 3)

                shapes.append(
                    LocalShape(
                        body_name=body_name,
                        points=points.astype(np.float32),
                        radius=radius,
                    )
                )

            elif geom_type == "box":
                half = size[:3]

                corners = np.asarray(
                    [
                        [sx * half[0], sy * half[1], sz * half[2]]
                        for sx in (-1.0, 1.0)
                        for sy in (-1.0, 1.0)
                        for sz in (-1.0, 1.0)
                    ],
                    dtype=np.float32,
                )

                points = pos.reshape(1, 3) + corners @ local_rot.T

                shapes.append(
                    LocalShape(
                        body_name=body_name,
                        points=points.astype(np.float32),
                        radius=0.0,
                    )
                )

            elif geom_type == "cylinder":
                radius = float(size[0])
                half_len = float(size[1]) if len(size) > 1 else 0.0

                local_points = np.asarray(
                    [[0.0, 0.0, -half_len], [0.0, 0.0, half_len]],
                    dtype=np.float32,
                )

                points = pos.reshape(1, 3) + local_points @ local_rot.T

                shapes.append(
                    LocalShape(
                        body_name=body_name,
                        points=points.astype(np.float32),
                        radius=radius,
                    )
                )

    if len(shapes) == 0:
        raise RuntimeError(f"No collision shapes parsed from {xml_path}")

    return shapes


def discover_assets(asset_root: Path) -> Dict[Tuple[str, str], AssetEntry]:
    asset_index: Dict[Tuple[str, str], AssetEntry] = {}

    for xml_path in sorted(asset_root.glob("*_smpl.xml")):
        match = ASSET_RE.match(xml_path.name)
        if match is None:
            continue

        gender = match.group(1)
        beta_key = match.group(2)
        asset_id = f"{gender}_{beta_key}"

        asset_index[(gender, beta_key)] = AssetEntry(
            gender=gender,
            beta_key=beta_key,
            asset_id=asset_id,
            xml_path=xml_path,
        )

    if len(asset_index) == 0:
        raise RuntimeError(f"No morphology XML files found in {asset_root}")

    print(f"[ASSETS] found {len(asset_index)} assets")
    return asset_index


def quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) by quaternion.

    q: [4], xyzw
    v: [N, 3]
    """

    q = q / torch.clamp(torch.linalg.norm(q), min=1e-8)

    q_xyz = q[:3].view(1, 3).expand_as(v)
    q_w = q[3]

    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


class IsaacGymOffsetComputer:
    def __init__(
        self,
        motion_lib: MotionLib,
        motion_entries: List[MotionEntry],
        device: torch.device,
        target_z: float,
    ):
        self.motion_lib = motion_lib
        self.motion_entries = motion_entries
        self.device = device
        self.target_z = target_z

        self.num_envs = len(motion_entries)

        self.gym = gymapi.acquire_gym()
        self.sim = None

        self.envs = []
        self.actor_handles = []

        self.num_dof = None
        self.num_bodies = None

        self.body_names_per_env: List[List[str]] = []
        self.body_name_to_idx_per_env: List[Dict[str, int]] = []
        self.local_shapes_per_env: List[List[LocalShape]] = []

        self.root_states = None
        self.dof_states = None
        self.rigid_body_states = None

        self._create_sim()

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None

    def _create_asset_options(self) -> gymapi.AssetOptions:
        opts = gymapi.AssetOptions()

        opts.fix_base_link = False
        opts.disable_gravity = True
        opts.collapse_fixed_joints = True
        opts.replace_cylinder_with_capsule = True

        opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        opts.angular_damping = 0.01
        opts.linear_damping = 0.0
        opts.max_angular_velocity = 100.0
        opts.max_linear_velocity = 1000.0

        return opts

    def _create_sim(self):
        device_id = 0
        if self.device.type == "cuda":
            device_id = self.device.index if self.device.index is not None else 0

        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.use_gpu_pipeline = self.device.type == "cuda"

        sim_params.physx.use_gpu = self.device.type == "cuda"
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4

        self.sim = self.gym.create_sim(
            device_id,
            device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )

        if self.sim is None:
            raise RuntimeError("Failed to create IsaacGym sim")

        asset_options = self._create_asset_options()

        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(0.0, 0.0, 0.0)
        num_per_row = max(1, int(math.sqrt(self.num_envs)))

        for env_id, entry in enumerate(self.motion_entries):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            asset = self.gym.load_asset(
                self.sim,
                str(entry.xml_path.parent),
                entry.xml_path.name,
                asset_options,
            )

            if asset is None:
                raise RuntimeError(f"Failed to load asset: {entry.xml_path}")

            num_dof = self.gym.get_asset_dof_count(asset)
            num_bodies = self.gym.get_asset_rigid_body_count(asset)
            body_names = list(self.gym.get_asset_rigid_body_names(asset))

            if self.num_dof is None:
                self.num_dof = num_dof
            elif self.num_dof != num_dof:
                raise RuntimeError(
                    f"DOF mismatch: {entry.asset_id} has {num_dof}, "
                    f"expected {self.num_dof}"
                )

            if self.num_bodies is None:
                self.num_bodies = num_bodies
            elif self.num_bodies != num_bodies:
                raise RuntimeError(
                    f"Body mismatch: {entry.asset_id} has {num_bodies}, "
                    f"expected {self.num_bodies}"
                )

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            actor = self.gym.create_actor(
                env,
                asset,
                pose,
                "humanoid",
                env_id,
                1,
                0,
            )

            dof_props = self.gym.get_actor_dof_properties(env, actor)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_NONE)
            dof_props["stiffness"].fill(0.0)
            dof_props["damping"].fill(0.0)
            self.gym.set_actor_dof_properties(env, actor, dof_props)

            self.envs.append(env)
            self.actor_handles.append(actor)

            self.body_names_per_env.append(body_names)
            self.body_name_to_idx_per_env.append(
                {name: i for i, name in enumerate(body_names)}
            )
            self.local_shapes_per_env.append(parse_mjcf_collision_shapes(entry.xml_path))

        self.gym.prepare_sim(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(root_tensor).view(self.num_envs, 13)
        self.dof_states = gymtorch.wrap_tensor(dof_tensor).view(
            self.num_envs,
            self.num_dof,
            2,
        )
        self.rigid_body_states = gymtorch.wrap_tensor(rb_tensor).view(
            self.num_envs,
            self.num_bodies,
            13,
        )

        print(
            f"[SIM] envs={self.num_envs}, bodies={self.num_bodies}, dofs={self.num_dof}"
        )

    def _set_frame0_states(self):
        """
        Set each IsaacGym actor to MotionLib frame 0.

        This is the critical part:
        we use MotionLib.gts/grs/dps directly, because this is exactly what
        the visualizer later uses.
        """

        for env_id, entry in enumerate(self.motion_entries):
            motion_id = entry.motion_id
            start = int(self.motion_lib.length_starts[motion_id].item())

            root_pos = self.motion_lib.gts[start, 0]
            root_rot = self.motion_lib.grs[start, 0]
            dof_pos = self.motion_lib.dps[start]

            self.root_states[env_id, 0:3] = root_pos
            self.root_states[env_id, 3:7] = root_rot
            self.root_states[env_id, 7:13] = 0.0

            self.dof_states[env_id, :, 0] = dof_pos
            self.dof_states[env_id, :, 1] = 0.0

        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
        )

        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
        )

        # Force IsaacGym to propagate root/dof states into rigid-body FK.
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # Same idea as ProtoMotions reset: set tensors, then refresh.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _lowest_collision_z(self, env_id: int) -> float:
        body_name_to_idx = self.body_name_to_idx_per_env[env_id]
        shapes = self.local_shapes_per_env[env_id]

        lowest = float("inf")
        used = 0

        for shape in shapes:
            if shape.body_name not in body_name_to_idx:
                continue

            body_idx = body_name_to_idx[shape.body_name]

            body_pos = self.rigid_body_states[env_id, body_idx, 0:3]
            body_rot = self.rigid_body_states[env_id, body_idx, 3:7]  # xyzw

            local_points = torch.as_tensor(
                shape.points,
                dtype=torch.float32,
                device=self.device,
            )

            world_points = body_pos.view(1, 3) + quat_rotate_xyzw(body_rot, local_points)

            shape_lowest = float(world_points[:, 2].min().item()) - float(shape.radius)

            lowest = min(lowest, shape_lowest)
            used += 1

        if used == 0:
            raise RuntimeError(f"No usable shapes for env_id={env_id}")

        return float(lowest)

    def compute_offsets(self) -> Dict[int, float]:
        self._set_frame0_states()

        offsets: Dict[int, float] = {}

        for env_id, entry in enumerate(self.motion_entries):
            lowest_z = self._lowest_collision_z(env_id)
            offset = self.target_z - lowest_z

            offsets[entry.motion_id] = float(offset)

            print(
                f"[OFFSET] motion_id={entry.motion_id:04d} "
                f"clip={entry.clip_id} "
                f"asset={entry.asset_id} "
                f"lowest_z={lowest_z:.6f} "
                f"offset={offset:.6f}"
            )

        return offsets


def default_offset_motion_file(motion_file: Path) -> Path:
    return motion_file.with_name(f"{motion_file.stem}_offset{motion_file.suffix}")

def save_corrected_motion_file(
    source_motion_file: Path,
    motion_lib: MotionLib,
    output_motion_file: Path,
    overwrite: bool,
):
    if output_motion_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output motion file already exists: {output_motion_file}\n"
            f"Use --overwrite to replace it."
        )

    output_motion_file.parent.mkdir(parents=True, exist_ok=True)

    # Load original dict to preserve all metadata fields, including custom ones
    # e.g. motion_clip_ids, motion_asset_ids, motion_npz_files.
    data = torch.load(source_motion_file, map_location="cpu", weights_only=False)

    # Only gts changes. Velocities do not change because this is a constant Z shift.
    data["gts"] = motion_lib.gts.detach().cpu()

    tmp_path = output_motion_file.with_suffix(output_motion_file.suffix + ".tmp")
    torch.save(data, tmp_path)
    tmp_path.replace(output_motion_file)

    print(f"[SAVE] corrected MotionLib: {output_motion_file}")

def apply_offsets_to_motion_lib(
    motion_lib: MotionLib,
    motion_entries: List[MotionEntry],
    offsets: Dict[int, float],
):
    for entry in motion_entries:
        motion_id = entry.motion_id
        offset = float(offsets[motion_id])

        start = int(motion_lib.length_starts[motion_id].item())
        num_frames = int(motion_lib.motion_num_frames[motion_id].item())
        end = start + num_frames

        motion_lib.gts[start:end, :, 2] += offset

        print(
            f"[APPLY] motion_id={motion_id:04d} "
            f"clip={entry.clip_id} "
            f"asset={entry.asset_id} "
            f"offset={offset:.6f}"
        )

def infer_clip_id_from_variant_stem(stem: str, gender: str) -> str:
    """
    Expected variant stem:
        000005_v00_male_0e26b88d

    Desired clip_id:
        000005
    """

    marker = f"_{gender}_"

    if marker not in stem:
        return stem

    prefix = stem.rsplit(marker, 1)[0]  # 000005_v00

    if "_v" not in prefix:
        return prefix

    return prefix.rsplit("_v", 1)[0]    # 000005


def get_clip_id(motion_lib: MotionLib, motion_id: int) -> str:
    # Preferred: explicit semantic clip id saved during conversion.
    if hasattr(motion_lib, "motion_clip_ids"):
        return str(motion_lib.motion_clip_ids[motion_id])

    # Fallback: infer from variant filename.
    if hasattr(motion_lib, "motion_files") and len(motion_lib.motion_files) > motion_id:
        stem = Path(motion_lib.motion_files[motion_id]).stem
        gender = str(motion_lib.motion_genders[motion_id])
        return infer_clip_id_from_variant_stem(stem, gender)

    return f"{motion_id:06d}"


def build_motion_entries(
    motion_lib: MotionLib,
    asset_index: Dict[Tuple[str, str], AssetEntry],
    limit: int,
) -> List[MotionEntry]:
    entries: List[MotionEntry] = []

    total = motion_lib.num_motions()

    for motion_id in range(total):
        clip_id = get_clip_id(motion_lib, motion_id)

        gender = str(motion_lib.motion_genders[motion_id])
        beta_key = str(motion_lib.motion_beta_keys[motion_id])
        asset_id = f"{gender}_{beta_key}"

        asset_key = (gender, beta_key)

        if asset_key not in asset_index:
            print(f"[WARN] missing asset XML for {asset_id}")
            continue

        asset = asset_index[asset_key]

        entries.append(
            MotionEntry(
                motion_id=motion_id,
                clip_id=clip_id,
                gender=gender,
                beta_key=beta_key,
                asset_id=asset_id,
                xml_path=asset.xml_path,
            )
        )

        if limit > 0 and len(entries) >= limit:
            break

    return entries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motion-file",
        type=Path,
        required=True,
        help="Packaged MotionLib .pt file, e.g. humos_8.pt",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(
            "/home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor"
        ),
    )
    parser.add_argument(
        "--out-motion-file",
        type=Path,
        default=None,
        help="Output corrected MotionLib .pt file. Default: <motion-file>_offset.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit processed motions. Use -1 for all. For final preprocessing, use -1.",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=0.005,
        help="Desired lowest collision point after applying offset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute offsets even if they already exist.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[MOTION] loading MotionLib: {args.motion_file}")

    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=str(args.motion_file)),
        device=device,
    )

    if not motion_lib.has_morphology_metadata():
        raise RuntimeError("MotionLib does not contain morphology metadata.")

    print(f"[MOTION] num_motions={motion_lib.num_motions()}")

    asset_index = discover_assets(args.asset_root)
    output_motion_file = (
        args.out_motion_file
        if args.out_motion_file is not None
        else default_offset_motion_file(args.motion_file)
    )

    motion_entries = build_motion_entries(
        motion_lib=motion_lib,
        asset_index=asset_index,
        limit=args.limit,
    )

    print(f"[TODO] process {len(motion_entries)} motions")

    if len(motion_entries) == 0:
        print("[DONE] nothing to process")
        return

    computer = None

    try:
        computer = IsaacGymOffsetComputer(
            motion_lib=motion_lib,
            motion_entries=motion_entries,
            device=device,
            target_z=args.target_z,
        )

        offsets = computer.compute_offsets()

        apply_offsets_to_motion_lib(
            motion_lib=motion_lib,
            motion_entries=motion_entries,
            offsets=offsets,
        )

        save_corrected_motion_file(
            source_motion_file=args.motion_file,
            motion_lib=motion_lib,
            output_motion_file=output_motion_file,
            overwrite=args.overwrite,
        )

        print("[DONE]")

    except Exception:
        traceback.print_exc()
        if args.stop_on_error:
            raise

    finally:
        if computer is not None:
            computer.close()


if __name__ == "__main__":
    main()