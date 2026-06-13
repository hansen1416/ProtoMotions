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
Extract physics features from SMPL MuJoCo XML files.

15 features per body (z-scored across all training bodies):
  total_mass        — sum of all geom masses [kg]
  l_thigh_length    — femur length, left [m]
  l_shin_length     — tibia length, left [m]
  r_thigh_length    — femur length, right [m]
  r_shin_length     — tibia length, right [m]
  l_upper_arm_len   — humerus length, left [m]
  l_forearm_len     — ulna/radius length, left [m]
  r_upper_arm_len   — humerus length, right [m]
  r_forearm_len     — ulna/radius length, right [m]
  torso_height      — summed pelvis→chest segment lengths [m]
  neck_head_height  — summed chest→head segment lengths [m]
  hip_width         — L_Hip to R_Hip lateral distance [m]
  shoulder_width    — L_Shoulder to R_Shoulder lateral distance [m]
  leg_length        — l_thigh + l_shin [m]
  total_height      — leg_length + torso_height + neck_head_height [m]

Usage:
    python tools/extract_smpl_physics_features.py \
        --xml-dir protomotions/data/assets/mjcf/smpl_mor \
        --output protomotions/data/assets/mjcf/smpl_mor/physics_features.pt

Output .pt keys:
    features_raw  dict[asset_id -> tensor(15)]  — raw values in SI units
    features      dict[asset_id -> tensor(15)]  — z-scored across training bodies
    feature_names list[str]                     — 15 feature names
    mean          tensor(15)                    — per-feature mean
    std           tensor(15)                    — per-feature std (eps-floored)
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


FEATURE_NAMES = [
    "total_mass",
    "l_thigh_length",
    "l_shin_length",
    "r_thigh_length",
    "r_shin_length",
    "l_upper_arm_len",
    "l_forearm_len",
    "r_upper_arm_len",
    "r_forearm_len",
    "torso_height",
    "neck_head_height",
    "hip_width",
    "shoulder_width",
    "leg_length",
    "total_height",
]


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.strip().split()]


def _geom_mass(geom: ET.Element) -> float:
    """Compute mass of a MuJoCo geom from density and geometry."""
    gtype = geom.get("type", "sphere")
    density = float(geom.get("density", "1000"))
    size_str = geom.get("size", "0.05")
    size = _parse_floats(size_str)

    if gtype == "capsule":
        r = size[0]
        fromto_str = geom.get("fromto")
        if fromto_str:
            pts = _parse_floats(fromto_str)
            p1 = np.array(pts[:3])
            p2 = np.array(pts[3:])
            length = float(np.linalg.norm(p2 - p1))
        else:
            length = size[1] * 2.0 if len(size) > 1 else 0.0
        vol = math.pi * r * r * length + (4.0 / 3.0) * math.pi * r ** 3
    elif gtype == "box":
        sx, sy, sz = size[0], size[1], size[2]
        vol = 8.0 * sx * sy * sz
    elif gtype == "sphere":
        r = size[0]
        vol = (4.0 / 3.0) * math.pi * r ** 3
    elif gtype == "cylinder":
        r = size[0]
        h = size[1] * 2.0 if len(size) > 1 else 0.0
        vol = math.pi * r * r * h
    else:
        vol = 0.0

    return density * vol


def _vec3(s: str) -> np.ndarray:
    return np.array(_parse_floats(s), dtype=np.float64)


def _extract_body_info(root_el: ET.Element) -> Dict[str, dict]:
    """
    Walk the MuJoCo body tree and collect for each body:
      - pos_rel: 3-vector, relative offset from parent joint
      - global_pos: accumulated position from Pelvis origin
      - mass: sum of geom masses in this body
    Returns dict keyed by body name.
    """
    info: Dict[str, dict] = {}

    def walk(el: ET.Element, parent_global: np.ndarray):
        if el.tag != "body":
            for child in el:
                walk(child, parent_global)
            return

        name = el.get("name", "")
        pos_rel = _vec3(el.get("pos", "0 0 0"))
        global_pos = parent_global + pos_rel

        mass = sum(_geom_mass(g) for g in el if g.tag == "geom")
        info[name] = {
            "pos_rel": pos_rel,
            "global_pos": global_pos,
            "mass": mass,
        }

        for child in el:
            walk(child, global_pos)

    walk(root_el, np.zeros(3))
    return info


def _extract_features(xml_path: str) -> np.ndarray:
    """Extract 15 physics features from a single SMPL XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    info = _extract_body_info(root)

    def glen(body_name: str) -> float:
        return float(np.linalg.norm(info[body_name]["pos_rel"]))

    def gy(body_name: str) -> float:
        return float(info[body_name]["global_pos"][1])

    total_mass = sum(b["mass"] for b in info.values())

    l_thigh = glen("L_Knee")
    l_shin = glen("L_Ankle")
    r_thigh = glen("R_Knee")
    r_shin = glen("R_Ankle")

    l_upper_arm = glen("L_Elbow")
    l_forearm = glen("L_Wrist")
    r_upper_arm = glen("R_Elbow")
    r_forearm = glen("R_Wrist")

    torso_height = glen("Torso") + glen("Spine") + glen("Chest")
    neck_head_height = glen("Neck") + glen("Head")

    hip_width = abs(gy("L_Hip") - gy("R_Hip"))
    shoulder_width = abs(gy("L_Shoulder") - gy("R_Shoulder"))

    leg_length = l_thigh + l_shin
    total_height = leg_length + torso_height + neck_head_height

    return np.array(
        [
            total_mass,
            l_thigh, l_shin,
            r_thigh, r_shin,
            l_upper_arm, l_forearm,
            r_upper_arm, r_forearm,
            torso_height,
            neck_head_height,
            hip_width,
            shoulder_width,
            leg_length,
            total_height,
        ],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract physics features from SMPL XMLs.")
    parser.add_argument(
        "--xml-dir",
        default="protomotions/data/assets/mjcf/smpl_mor",
        help="Directory containing *_smpl.xml files",
    )
    parser.add_argument(
        "--output",
        default="protomotions/data/assets/mjcf/smpl_mor/physics_features.pt",
        help="Output .pt file path",
    )
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    xml_files = sorted(xml_dir.glob("*_smpl.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No *_smpl.xml files found in {xml_dir}")

    print(f"Processing {len(xml_files)} XML files from {xml_dir}")

    features_raw: Dict[str, np.ndarray] = {}
    for xml_path in xml_files:
        # filename: female_093098f0_smpl.xml → asset_id: female_093098f0
        stem = xml_path.stem  # e.g. female_093098f0_smpl
        asset_id = stem.replace("_smpl", "")
        features_raw[asset_id] = _extract_features(str(xml_path))

    asset_ids = sorted(features_raw.keys())
    matrix = np.stack([features_raw[a] for a in asset_ids], axis=0)  # [N, 15]

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    features_zscored = {a: (features_raw[a] - mean) / std for a in asset_ids}

    out = {
        "features_raw": {a: torch.tensor(features_raw[a]) for a in asset_ids},
        "features": {a: torch.tensor(features_zscored[a]) for a in asset_ids},
        "feature_names": FEATURE_NAMES,
        "mean": torch.tensor(mean),
        "std": torch.tensor(std),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(output_path))
    print(f"Saved physics features for {len(asset_ids)} assets → {output_path}")

    # Print summary statistics
    print("\nFeature summary (raw values):")
    print(f"{'Feature':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 64)
    for i, name in enumerate(FEATURE_NAMES):
        print(
            f"{name:<22} {mean[i]:10.4f} {std[i]:10.4f} "
            f"{matrix[:, i].min():10.4f} {matrix[:, i].max():10.4f}"
        )


if __name__ == "__main__":
    main()
