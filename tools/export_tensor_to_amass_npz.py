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

"""Convert HUMOS .tensor files → AMASS-style .npz with zeroed betas.

Uses the HHI valid_sorted.json to select the 20,951 physically-plausible,
difficulty-ordered motion IDs, then exports one .npz per clip at the
original actor's gender but with betas=0 (neutral SMPL body shape).

Output is compatible with convert_amass_to_motionlib_with_morphology.py
(Step 5 of the held-out pipeline).

Typical usage
-------------
    python tools/export_tensor_to_amass_npz.py \\
        --out-root /home/hlz/datasets/amass_neutral \\
        --skip-existing

Output layout
-------------
    <out-root>/
        HML3D/
            000000_v00_male_neutral.npz
            000002_v00_male_neutral.npz
            ...
        humanml3d_neutral_<N>.yaml          # motion config for downstream pipeline
        humanml3d_neutral_<N>_manifest.yaml # full metadata per clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import orjson
import torch
import yaml
from tqdm import tqdm


TENSOR_FPS = 20.0  # .tensor files are stored at 20 FPS


def gender_to_str(raw) -> str:
    """Convert numeric gender (1.0=male, -1.0=female) stored in .tensor to string."""
    val = float(np.asarray(raw).flat[0])
    if val > 0:
        return "male"
    if val < 0:
        return "female"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HUMOS .tensor files to AMASS-style .npz with zeroed betas."
    )
    parser.add_argument(
        "--valid-ids",
        type=Path,
        default=Path("/home/hlz/repos/hhi/data-processing/valid_sorted.json"),
        help="valid_sorted.json from HHI data-processing (keyid → difficulty score).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "/home/hlz/repos/humos/humos/annotations/humanml3d/annotations_processed.json"
        ),
        help="HUMOS annotations JSON (keyid → {path, duration, …}).",
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=Path("/home/hlz/repos/humos/datasets/humos3dfeats"),
        help="Root directory containing HUMOS .tensor files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root directory for .npz files and YAML.",
    )
    parser.add_argument(
        "--beta-key",
        type=str,
        default="neutral",
        help="Beta key embedded in output filenames (default: neutral).",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default=None,
        help="Override output YAML filename. Default: humanml3d_neutral_<N>.yaml.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose .npz already exists on disk (safe to resume).",
    )
    args = parser.parse_args()

    # --- load inputs ----------------------------------------------------------
    with open(args.valid_ids) as f:
        valid_sorted: dict = json.load(f)
    keyids = list(valid_sorted.keys())  # difficulty-ordered

    with open(args.annotations, "rb") as f:
        annotations: dict = orjson.loads(f.read())

    seq_dir = args.out_root / "HML3D"
    seq_dir.mkdir(parents=True, exist_ok=True)

    yaml_motions: list[dict] = []
    manifest: list[dict] = []
    n_skipped = 0
    n_missing = 0

    for keyid in tqdm(keyids, desc="Exporting", unit="clip"):
        if keyid not in annotations:
            tqdm.write(f"[WARN] no annotation for {keyid}, skipping")
            n_missing += 1
            continue

        tensor_path = args.tensor_dir / (annotations[keyid]["path"] + ".tensor")
        if not tensor_path.exists():
            tqdm.write(f"[WARN] missing .tensor for {keyid}: {tensor_path}")
            n_missing += 1
            continue

        data = torch.load(tensor_path, map_location="cpu", weights_only=False)

        root_orient = np.asarray(data["root_orient"], dtype=np.float32)  # (T, 3)
        pose_body = np.asarray(data["pose_body"], dtype=np.float32)       # (T, 63)
        trans = np.asarray(data["trans"], dtype=np.float32)               # (T, 3)
        gender = gender_to_str(data["gender"])
        T = root_orient.shape[0]

        # poses layout expected by convert_amass_to_proto.py:
        # root_orient (3) + 21 body joints (63) = 66 dims
        poses = np.concatenate([root_orient, pose_body[:, :63]], axis=1).astype(
            np.float32
        )

        out_stem = f"{keyid}_v00_{gender}_{args.beta_key}"
        npz_path = seq_dir / f"{out_stem}.npz"
        motion_rel = f"HML3D/{out_stem}.motion"
        duration = T / TENSOR_FPS

        if args.skip_existing and npz_path.exists():
            n_skipped += 1
        else:
            np.savez(
                npz_path,
                poses=poses,
                trans=trans,
                betas=np.zeros(10, dtype=np.float32),
                gender=np.array(gender),
                clip_id=np.array(keyid),
                mocap_framerate=np.array(TENSOR_FPS, dtype=np.float32),
            )

        yaml_motions.append(
            {
                "file": motion_rel,
                "fps": TENSOR_FPS,
                "weight": 1.0,
                "sub_motions": [{"timings": {"start": 0.0, "end": float(duration)}}],
            }
        )
        manifest.append(
            {
                "index": len(manifest),
                "clip_id": keyid,
                "gender": gender,
                "beta_key": args.beta_key,
                "difficulty_score": valid_sorted[keyid].get("score", None),
                "npz": str(npz_path),
                "motion": motion_rel,
                "duration": float(duration),
            }
        )

    total = len(yaml_motions)
    yaml_name = args.yaml_name or f"humanml3d_neutral_{total}.yaml"
    yaml_path = args.out_root / yaml_name
    manifest_path = args.out_root / yaml_name.replace(".yaml", "_manifest.yaml")

    with open(yaml_path, "w") as f:
        yaml.safe_dump({"motions": yaml_motions}, f, sort_keys=False)

    with open(manifest_path, "w") as f:
        yaml.safe_dump({"variants": manifest}, f, sort_keys=False)

    print(f"\nSaved YAML:        {yaml_path}")
    print(f"Saved manifest:    {manifest_path}")
    print(f"Total exported:    {total} clips")
    if n_skipped:
        print(f"Skipped existing:  {n_skipped}")
    if n_missing:
        print(f"Missing/skipped:   {n_missing}")


if __name__ == "__main__":
    main()
