#!/usr/bin/env python3
# tools/compute_humos_frame0_offsets.py

"""
Apply IsaacGym-computed frame-0 grounding offsets directly to a packaged MotionLib.

Example:
python tools/compute_humos_frame0_offsets.py \
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
from tqdm import tqdm

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


def _cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross product along last dim with arbitrary leading batch dims."""
    return torch.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


def quat_rotate_per_env(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate per-env point sets by per-env quaternions.

    q : [N, 4]    xyzw, one quaternion per env
    v : [N, P, 3] P local points per env
    returns [N, P, 3]
    """
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)
    q_xyz = q[:, :3].unsqueeze(1)  # [N, 1, 3]  broadcasts over P
    q_w   = q[:, 3].view(-1, 1, 1) # [N, 1, 1]

    t = 2.0 * _cross3(q_xyz, v)                  # [N, P, 3]
    return v + q_w * t + _cross3(q_xyz, t)        # [N, P, 3]


class IsaacGymOffsetComputer:
    """
    Computes frame-0 grounding offsets using IsaacGym FK.

    Key optimisation over the naive version:
      - Creates ONE env per UNIQUE SMPL shape (~128) instead of one per motion
        (~8192).  The same envs are reused across rounds.
      - Uses IsaacGym's indexed set-tensor API so only the envs whose motion
        changes each round are touched.
      - Vectorises the lowest-collision-Z calculation across all envs at once
        on the GPU instead of a Python loop.

    Memory and creation time drop by ~64× (8192 / 128 unique shapes).
    """

    def __init__(
        self,
        motion_lib: MotionLib,
        motion_entries: List[MotionEntry],
        device: torch.device,
        target_z: float,
    ):
        self.motion_lib = motion_lib
        self.device = device
        self.target_z = target_z

        # ── group by unique asset, preserving first-seen order ──────────────
        self._asset_order: List[str] = []       # env_id → asset_id
        self._asset_xml:   List[Path] = []      # env_id → xml_path
        seen: Dict[str, int] = {}

        self._asset_to_motions: Dict[str, List[MotionEntry]] = {}

        for entry in motion_entries:
            aid = entry.asset_id
            if aid not in seen:
                seen[aid] = len(self._asset_order)
                self._asset_order.append(aid)
                self._asset_xml.append(entry.xml_path)
                self._asset_to_motions[aid] = []
            self._asset_to_motions[aid].append(entry)

        self.num_envs = len(self._asset_order)

        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.envs: List = []
        self.actor_handles: List = []

        self.num_dof:    int = 0
        self.num_bodies: int = 0

        self.body_name_to_idx: Dict[str, int] = {}  # same for all SMPL topologies
        self.local_shapes_per_env: List[List[LocalShape]] = []

        self.root_states       = None
        self.dof_states        = None
        self.rigid_body_states = None

        # Pre-built batched tensors for vectorised lowest-Z:
        # list of (body_idx, points[N,P,3], radii[N])
        self._batched_shapes: List[Tuple[int, torch.Tensor, torch.Tensor]] = []

        self._create_sim()
        self._build_batched_shapes()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None

    # ── IsaacGym setup ───────────────────────────────────────────────────────

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

        self.sim = self.gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create IsaacGym sim")

        asset_options = self._create_asset_options()
        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(0.0, 0.0, 0.0)
        num_per_row = max(1, int(math.sqrt(self.num_envs)))

        print(f"[SIM] creating {self.num_envs} envs (one per unique SMPL shape)")

        for env_id, xml_path in enumerate(self._asset_xml):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            asset = self.gym.load_asset(
                self.sim, str(xml_path.parent), xml_path.name, asset_options
            )
            if asset is None:
                raise RuntimeError(f"Failed to load asset: {xml_path}")

            num_dof    = self.gym.get_asset_dof_count(asset)
            num_bodies = self.gym.get_asset_rigid_body_count(asset)
            body_names = list(self.gym.get_asset_rigid_body_names(asset))

            if env_id == 0:
                self.num_dof    = num_dof
                self.num_bodies = num_bodies
                self.body_name_to_idx = {name: i for i, name in enumerate(body_names)}
            else:
                if num_dof != self.num_dof:
                    raise RuntimeError(f"DOF mismatch for {xml_path.name}")
                if num_bodies != self.num_bodies:
                    raise RuntimeError(f"Body mismatch for {xml_path.name}")

            pose  = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            actor = self.gym.create_actor(env, asset, pose, "humanoid", env_id, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env, actor)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_NONE)
            dof_props["stiffness"].fill(0.0)
            dof_props["damping"].fill(0.0)
            self.gym.set_actor_dof_properties(env, actor, dof_props)

            self.envs.append(env)
            self.actor_handles.append(actor)
            self.local_shapes_per_env.append(parse_mjcf_collision_shapes(xml_path))

        self.gym.prepare_sim(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor  = self.gym.acquire_dof_state_tensor(self.sim)
        rb_tensor   = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.root_states       = gymtorch.wrap_tensor(root_tensor).view(self.num_envs, 13)
        self.dof_states        = gymtorch.wrap_tensor(dof_tensor).view(self.num_envs, self.num_dof, 2)
        self.rigid_body_states = gymtorch.wrap_tensor(rb_tensor).view(self.num_envs, self.num_bodies, 13)

        print(f"[SIM] ready  bodies={self.num_bodies}  dofs={self.num_dof}")

    # ── pre-build batched shape tensors ──────────────────────────────────────

    def _build_batched_shapes(self):
        """
        Pre-build [N_envs, P, 3] tensors for every geom slot so that
        _compute_lowest_z_all_envs can run as pure batched GPU tensor ops.

        All SMPL topologies share the same body/geom count — only sizes differ.
        """
        N = self.num_envs
        template = self.local_shapes_per_env[0]

        for slot, tmpl_shape in enumerate(template):
            body_name = tmpl_shape.body_name
            if body_name not in self.body_name_to_idx:
                continue

            body_idx = self.body_name_to_idx[body_name]
            P = tmpl_shape.points.shape[0]

            all_pts  = []
            all_rads = []
            ok = True
            for env_id in range(N):
                env_shapes = self.local_shapes_per_env[env_id]
                if slot >= len(env_shapes) or env_shapes[slot].body_name != body_name:
                    ok = False
                    break
                s = env_shapes[slot]
                all_pts.append(torch.as_tensor(s.points, dtype=torch.float32, device=self.device))
                all_rads.append(s.radius)

            if not ok:
                continue

            pts_batch  = torch.stack(all_pts,  dim=0)                                      # [N, P, 3]
            rads_batch = torch.tensor(all_rads, dtype=torch.float32, device=self.device)   # [N]
            self._batched_shapes.append((body_idx, pts_batch, rads_batch))

        print(f"[SHAPES] {len(self._batched_shapes)} geom slots batched across {N} envs")

    # ── per-round simulation ─────────────────────────────────────────────────

    def _set_states_indexed(self, round_motions: List[Tuple[int, MotionEntry]]):
        """
        Set frame-0 root/DOF state for the active envs this round,
        using IsaacGym's indexed API so untouched envs are not disturbed.
        """
        env_ids = torch.tensor(
            [env_id for env_id, _ in round_motions],
            dtype=torch.int32,
            device=self.device,
        )

        for env_id, entry in round_motions:
            start = int(self.motion_lib.length_starts[entry.motion_id].item())
            self.root_states[env_id, 0:3]  = self.motion_lib.gts[start, 0].to(self.device)
            self.root_states[env_id, 3:7]  = self.motion_lib.grs[start, 0].to(self.device)
            self.root_states[env_id, 7:13] = 0.0
            self.dof_states[env_id, :, 0]  = self.motion_lib.dps[start].to(self.device)
            self.dof_states[env_id, :, 1]  = 0.0

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids),
            len(env_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids),
            len(env_ids),
        )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    # ── vectorised lowest-Z ──────────────────────────────────────────────────

    def _compute_lowest_z_all_envs(self) -> torch.Tensor:
        """
        GPU-vectorised: compute lowest world-Z collision point for every env.

        Returns tensor of shape [num_envs] on self.device.
        """
        body_pos = self.rigid_body_states[:, :, 0:3]   # [N, B, 3]
        body_rot = self.rigid_body_states[:, :, 3:7]   # [N, B, 4]  xyzw

        lowest = torch.full((self.num_envs,), float("inf"), device=self.device)

        for body_idx, pts_batch, rads_batch in self._batched_shapes:
            bpos = body_pos[:, body_idx, :]   # [N, 3]
            brot = body_rot[:, body_idx, :]   # [N, 4]

            # world_pts: [N, P, 3]
            world_pts = bpos.unsqueeze(1) + quat_rotate_per_env(brot, pts_batch)

            shape_lowest = world_pts[:, :, 2].min(dim=1).values - rads_batch  # [N]
            lowest = torch.minimum(lowest, shape_lowest)

        return lowest

    # ── public entry point ───────────────────────────────────────────────────

    def compute_offsets(self) -> Dict[int, float]:
        """
        Process all motions in rounds.  Each round reuses the same N_shapes
        environments; only the active envs' states are updated via the indexed API.
        """
        max_rounds = max(len(m) for m in self._asset_to_motions.values())
        total = sum(len(m) for m in self._asset_to_motions.values())

        offsets: Dict[int, float] = {}

        with tqdm(total=total, desc="offsets", unit="motion") as pbar:
            for round_idx in range(max_rounds):
                round_motions: List[Tuple[int, MotionEntry]] = []
                for env_id, asset_id in enumerate(self._asset_order):
                    motions = self._asset_to_motions[asset_id]
                    if round_idx < len(motions):
                        round_motions.append((env_id, motions[round_idx]))

                if not round_motions:
                    break

                self._set_states_indexed(round_motions)
                lowest_z = self._compute_lowest_z_all_envs()

                for env_id, entry in round_motions:
                    offsets[entry.motion_id] = float(self.target_z - lowest_z[env_id].item())
                    pbar.update(1)

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