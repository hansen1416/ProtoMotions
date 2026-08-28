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
"""Pure-tensor MJCF collision-geometry helpers (no IsaacGym / physics dependency).

Parses simple collision primitives (capsule/sphere/box/cylinder) out of an MJCF
file and computes the lowest world-Z collision point per frame given a batch of
per-body world positions/rotations. Used to detect/correct ground penetration
without running an actual physics simulator.

The geometry parsing and rotation math here were originally written for
`tools/compute_humos_frame0_offsets.py` (which only used them at frame 0, driven
by IsaacGym purely as an FK trigger). Factored out here so they can be reused
for a full-sequence, IsaacGym-free version.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

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
    radius: float  # capsule/sphere radius; 0 for box


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
    """Parse simple collision geometry from MJCF: capsule, sphere, box, cylinder.

    Each geom is attached to its parent body name; the caller transforms it by
    that body's world pose.
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


def discover_assets(asset_root: Path) -> Dict[str, AssetEntry]:
    """Scan an asset directory for `<gender>_<beta_key>_smpl.xml` files.

    Returns a dict keyed by `asset_id` ("male_0e26b88d"), matching the
    `motion_asset_ids` strings stored per-motion in a packaged MotionLib.
    """
    asset_index: Dict[str, AssetEntry] = {}

    for xml_path in sorted(asset_root.glob("*_smpl.xml")):
        match = ASSET_RE.match(xml_path.name)
        if match is None:
            continue

        gender = match.group(1)
        beta_key = match.group(2)
        asset_id = f"{gender}_{beta_key}"

        asset_index[asset_id] = AssetEntry(
            gender=gender,
            beta_key=beta_key,
            asset_id=asset_id,
            xml_path=xml_path,
        )

    if len(asset_index) == 0:
        raise RuntimeError(f"No morphology XML files found in {asset_root}")

    return asset_index


def cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross product along the last dim, broadcasting over leading dims."""
    return torch.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


def quat_rotate_broadcast(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a shared set of local points by a batch of quaternions.

    Args:
        q: [T, 4] xyzw quaternions (one per frame).
        v: [P, 3] local points, shared across all T frames.

    Returns:
        [T, P, 3] world-frame (pre-translation) points.
    """
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)
    q_xyz = q[:, :3].unsqueeze(1)  # [T, 1, 3]
    q_w = q[:, 3].view(-1, 1, 1)  # [T, 1, 1]
    v_b = v.unsqueeze(0)  # [1, P, 3] -> broadcasts to [T, P, 3]

    t = 2.0 * cross3(q_xyz, v_b)
    return v_b + q_w * t + cross3(q_xyz, t)


def compute_lowest_z_for_shape(
    body_pos: torch.Tensor,  # [T, Nbodies, 3]
    body_rot_xyzw: torch.Tensor,  # [T, Nbodies, 4]
    shapes: List[LocalShape],
    body_name_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Batched (over T frames), pure-PyTorch lowest-collision-Z query for one shape.

    Equivalent to `compute_humos_frame0_offsets.py`'s `_compute_lowest_z_all_envs`,
    but batched over time for a single body shape rather than over N one-frame envs.
    """
    T = body_pos.shape[0]
    lowest = torch.full((T,), float("inf"), device=body_pos.device, dtype=body_pos.dtype)

    for shape in shapes:
        if shape.body_name not in body_name_to_idx:
            continue

        body_idx = body_name_to_idx[shape.body_name]
        pts = torch.as_tensor(shape.points, dtype=body_pos.dtype, device=body_pos.device)

        bpos = body_pos[:, body_idx, :]  # [T, 3]
        brot = body_rot_xyzw[:, body_idx, :]  # [T, 4]

        world_pts = bpos.unsqueeze(1) + quat_rotate_broadcast(brot, pts)  # [T, P, 3]
        shape_lowest = world_pts[..., 2].min(dim=1).values - shape.radius
        lowest = torch.minimum(lowest, shape_lowest)

    return lowest
