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

"""Build canonical-AMASS + per-shape-FK reference motions (step 1 of note/README.note.md §63).

Instead of HUMOS's per-shape diffusion resample (a different SMPL pose sequence theta(t) per
body shape, which is what introduced the Group-A jitter identified in the failed-clip-overlap
diagnostic), this uses the single pre-HUMOS canonical AMASS pose sequence for a clip -- the same
theta(t) for all 128 shapes -- and lets the downstream converter's forward-kinematics step
retarget it onto each shape's own skeleton. This isolates exactly one variable against the
existing `discover.py` corpus: the reference-motion *source*. Reward, termination, architecture,
optimizer -- everything else in the eventual training run stays untouched.

Reuses the HUMOS repo's own pre-processing primitives (`swap_left_right` for mirrored [M-prefixed]
clips, `take_out_z_rotation` + first-frame-xy centering for canonicalization) rather than
reimplementing this geometry, so the output is processed the same way every other clip in this
project's pipeline already is. Needs the `humos_p310` conda env (this project's env doesn't have
`aitviewer`/`human_body_prior`, which `humos.prepare.tools` imports transitively):

    conda run -n humos_p310 python tools/build_canonical_amass_retarget.py \\
        --clip-ids-file data_cache/small150_128shape.clip_ids.txt \\
        --out-root /home/hlz/datasets/canonical_retarget_interm

Output matches `tools/export_humos_to_amass_npz.py`'s format exactly (poses/trans/betas/gender/
clip_id/mocap_framerate NPZs, `{clip_id}_v{idx:02d}_{gender}_{beta_key}.npz` naming, motions.yaml
+ manifest.yaml) so the existing, unmodified `tools/convert_amass_to_motionlib_with_morphology.py`
can be pointed at it directly -- no new conversion code, same as every clip already in the corpus.

Known gap: does NOT apply the frame-0 grounding offset (`tools/compute_humos_frame0_offsets.py`,
note/README.data-pipeline-chronological.md Phase 12) -- that needs IsaacGym, which needs a GPU;
this machine has none (`nvidia-smi` fails, no CUDA driver). Run that step on the pod before
training, same as every other shard in the data pipeline.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

HUMOS_REPO_DEFAULT = "/home/hlz/repos/humos"
POSE_DATA_NATIVE_FPS = 20.0  # verified: frames/duration ~= 20.0 for pose_data/*.npz (Phase 3)
TARGET_FPS = 30.0  # matches small150_128shape.pt's packaged fps (verified via motion_dt)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def load_clip_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def load_source_pose(pose_data_root: Path, rel_path: str) -> dict:
    npz_path = pose_data_root / f"{rel_path}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"pose_data file not found: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    return {
        "root_orient": d["root_orient"].astype(np.float32),  # [T, 3]
        "pose_body": d["pose_body"].astype(np.float32),  # [T, 63]
        "trans": d["transl"].astype(np.float32),  # [T, 3]
    }


def trim_to_annotation(data: dict, start: float, end: float, fps: float) -> dict:
    T = data["root_orient"].shape[0]
    f0 = max(0, int(round(start * fps)))
    f1 = min(T, int(round(end * fps)))
    if f1 <= f0:
        f0, f1 = 0, T  # malformed annotation window -- fall back to the full clip
    return {k: v[f0:f1] for k, v in data.items()}


def resample_to_fps(data: dict, native_fps: float, target_fps: float) -> dict:
    """Upsample via linear interpolation (native pose_data is ~20fps; the rest of this project's
    150-clip corpus is packaged at 30fps -- convert_amass_to_motionlib_with_morphology.py's own
    fps-matching only downsamples to an integer divisor of the source rate, so getting to 30fps
    from a 20fps source has to happen here, before that converter ever sees the data)."""
    T = data["root_orient"].shape[0]
    if T < 2:
        return data
    duration = (T - 1) / native_fps
    n_out = max(2, int(round(duration * target_fps)) + 1)
    t_src = np.arange(T) / native_fps
    t_dst = np.clip(np.arange(n_out) / target_fps, t_src[0], t_src[-1])
    out = {}
    for k, v in data.items():
        out[k] = np.stack(
            [np.interp(t_dst, t_src, v[:, d]) for d in range(v.shape[1])], axis=1
        ).astype(np.float32)
    return out


def process_clip(clip_id: str, annotations: dict, pose_data_root: Path, tools_mod) -> dict:
    ann = annotations[clip_id]
    path = ann["path"]
    is_mirror = path.startswith("M/")
    real_path = path[2:] if is_mirror else path

    data = load_source_pose(pose_data_root, real_path)

    entry = ann["annotations"][0]
    data = trim_to_annotation(data, entry["start"], entry["end"], POSE_DATA_NATIVE_FPS)

    if is_mirror:
        data = tools_mod.swap_left_right(data)

    # Canonicalize: remove initial yaw, center first-frame xy. No ground_the_human -- that
    # needs an SMPL body model forward pass just to find a floor offset that Phase 12's
    # IsaacGym-based frame-0 grounding step (run separately, on the pod) already redoes properly
    # using the *target* shape's own collision geometry -- redundant and shape-wrong to do it here
    # against a placeholder/source body.
    torch_data = {
        "root_orient": torch.from_numpy(data["root_orient"]),
        "trans": torch.from_numpy(data["trans"]),
    }
    root_orient, trans = tools_mod.take_out_z_rotation(torch_data["root_orient"], torch_data["trans"])
    trans = trans.clone()
    trans[:, :2] -= trans[0, :2].clone()

    canon = {
        "root_orient": root_orient.numpy().astype(np.float32),
        "pose_body": data["pose_body"],
        "trans": trans.numpy().astype(np.float32),
    }
    return resample_to_fps(canon, POSE_DATA_NATIVE_FPS, TARGET_FPS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip-ids-file", required=True, type=Path)
    parser.add_argument("--humos-repo", default=HUMOS_REPO_DEFAULT, type=Path)
    parser.add_argument(
        "--betas-file", default="protomotions/data/assets/all_betas.pt", type=Path
    )
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--genders", nargs="+", default=["male", "female"])
    parser.add_argument("--output-fps-tag", type=float, default=TARGET_FPS)
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip .npz files already on disk."
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.humos_repo))
    from humos.prepare import tools as humos_tools  # noqa: E402

    pose_data_root = args.humos_repo / "datasets" / "pose_data"
    annotations = __import__("json").load(
        open(args.humos_repo / "humos" / "annotations" / "humanml3d" / "annotations.json")
    )
    betas = torch.load(args.betas_file, map_location="cpu", weights_only=False)
    beta_keys = sorted(betas.keys(), key=natural_key)

    clip_ids = load_clip_ids(args.clip_ids_file)

    seq_dir = args.out_root / "CANONICAL"
    seq_dir.mkdir(parents=True, exist_ok=True)

    yaml_motions = []
    manifest = []
    global_idx = 0
    skipped_clips = []

    for clip_id in tqdm(clip_ids, desc="Clips", unit="clip"):
        if clip_id not in annotations:
            skipped_clips.append(clip_id)
            continue
        try:
            canon = process_clip(clip_id, annotations, pose_data_root, humos_tools)
        except FileNotFoundError as e:
            tqdm.write(f"[WARN] {clip_id}: {e}")
            skipped_clips.append(clip_id)
            continue

        T = canon["root_orient"].shape[0]
        poses = np.concatenate([canon["root_orient"], canon["pose_body"]], axis=1).astype(np.float32)
        trans = canon["trans"]

        local_idx = 0
        for beta_key in beta_keys:
            beta_vec = betas[beta_key].numpy().astype(np.float32)
            for gender in args.genders:
                out_name = f"{clip_id}_v{local_idx:02d}_{gender}_{beta_key}"
                npz_path = seq_dir / f"{out_name}.npz"

                if not (args.skip_existing and npz_path.exists()):
                    np.savez(
                        npz_path,
                        poses=poses,
                        trans=trans,
                        betas=beta_vec,
                        gender=np.array(gender),
                        clip_id=np.array(clip_id),
                        mocap_framerate=np.array(args.output_fps_tag, dtype=np.float32),
                    )

                duration = T / args.output_fps_tag
                motion_rel = f"CANONICAL/{out_name}.motion"

                yaml_motions.append(
                    {
                        "file": motion_rel,
                        "fps": float(args.output_fps_tag),
                        "weight": 1.0,
                        "sub_motions": [{"timings": {"start": 0.0, "end": float(duration)}}],
                    }
                )
                manifest.append(
                    {
                        "index": global_idx,
                        "clip_id": clip_id,
                        "gender": gender,
                        "beta_key": beta_key,
                        "npz": str(npz_path),
                        "motion": motion_rel,
                        "duration": float(duration),
                        "betas": beta_vec.tolist(),
                    }
                )
                global_idx += 1
                local_idx += 1

    total = global_idx
    yaml_path = args.out_root / f"canonical_{total}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"motions": yaml_motions}, f, sort_keys=False)

    manifest_path = args.out_root / f"canonical_{total}_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump({"variants": manifest}, f, sort_keys=False)

    print(f"\nSaved YAML:      {yaml_path}")
    print(f"Saved manifest:  {manifest_path}")
    print(f"Total exported:  {total} variants from {len(clip_ids) - len(skipped_clips)}/{len(clip_ids)} clips")
    if skipped_clips:
        print(f"Skipped clips ({len(skipped_clips)}): {skipped_clips}")


if __name__ == "__main__":
    main()
