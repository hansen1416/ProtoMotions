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
Render a reference motion clip (ground-truth mocap, NOT a trained policy rollout) to MP4
as an animated 3D stick figure, entirely on CPU -- no simulator, no GPU, no checkpoint.

Why this exists: `protomotions/record_video_mor.py` is the project's normal video tool,
but it hard-requires `--simulator isaacgym` (raises if anything else is passed -- see its
own `if args.simulator != "isaacgym": raise ...`) and IsaacGym's off-screen camera needs a
CUDA-capable GPU to actually step the sim, not just to import the bindings. Neither is
available on this machine (`torch.cuda.is_available()` is False here even though the
`isaacgym` python package itself imports fine). Since the goal was to sanity-check what a
specific *reference* clip looks like (e.g. the two near-static outliers in the frozen
hard-clip set, note/README.note.md §56-57) rather than to see a trained policy imitate it,
a policy/simulator isn't actually needed -- the reference pose sequence is already sitting
in the motion `.pt` file's `gts` (global joint positions) tensor, so this script just plots
that directly with matplotlib and writes frames out via ffmpeg (found locally at
`/usr/bin/ffmpeg`).

Bone connections use the standard 24-joint SMPL kinematic tree ordering (pelvis=0, the
usual hip/spine/limb chain) since this project's `gts` tensor follows that convention for
`smpl_mor` bodies. If a given robot config uses a different body ordering this would need
updating -- there's no `common_naming_to_robot_body_names`-based topology available at the
MotionLib level to derive it from directly, unlike a live `Simulator`/`PoseLib` instance.

Usage (local, CPU-only):
    python tools/visualize_motion_clip.py \\
        --motion-file data_cache/small150_128shape.pt \\
        --clip-id M002028 \\
        --output /tmp/M002028.mp4

    # Pick a specific shape variant instead of the first one found:
    python tools/visualize_motion_clip.py \\
        --motion-file data_cache/small150_128shape.pt \\
        --clip-id M002028 --asset-id female_30f6048e \\
        --output /tmp/M002028_female_30f6048e.mp4
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402

# Standard SMPL 24-joint parent chain (pelvis-rooted). See module docstring for caveats.
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--clip-id", required=True, help="e.g. M002028")
    parser.add_argument(
        "--asset-id",
        default=None,
        help="Specific body shape (e.g. female_30f6048e). Defaults to the first shape "
        "variant found for --clip-id.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading motion library from {args.motion_file} ...")
    motion_lib = MotionLib(config=MotionLibConfig(motion_file=args.motion_file), device="cpu")
    if not motion_lib.has_clip_identity_metadata():
        raise SystemExit("MotionLib has no clip identity metadata -- can't select by clip_id.")

    clip_to_motion_ids = motion_lib.build_clip_id_to_motion_ids()
    if args.clip_id not in clip_to_motion_ids:
        raise SystemExit(
            f"clip_id {args.clip_id!r} not found. Example available ids: "
            f"{list(clip_to_motion_ids)[:5]}"
        )
    motion_ids = clip_to_motion_ids[args.clip_id]

    if args.asset_id is not None:
        asset_ids = motion_lib.get_motion_asset_ids(motion_ids)
        matches = [mid for mid, aid in zip(motion_ids.tolist(), asset_ids) if aid == args.asset_id]
        if not matches:
            raise SystemExit(f"asset_id {args.asset_id!r} not found among {args.clip_id}'s shapes.")
        motion_id = matches[0]
    else:
        motion_id = motion_ids[0].item()
        asset_id = motion_lib.get_motion_asset_ids(torch.tensor([motion_id]))[0]
        print(f"No --asset-id given, using first shape variant: {asset_id}")

    length_starts = motion_lib.length_starts[motion_id].item()
    num_frames = motion_lib.motion_num_frames[motion_id].item()
    dt = motion_lib.motion_dt[motion_id].item()
    gts = motion_lib.gts[length_starts : length_starts + num_frames]  # [T, 24, 3]
    print(f"Clip {args.clip_id}: {num_frames} frames @ {1.0 / dt:.1f} fps")

    xyz = gts.numpy()
    x_min, x_max = xyz[..., 0].min(), xyz[..., 0].max()
    y_min, y_max = xyz[..., 1].min(), xyz[..., 1].max()
    z_min, z_max = xyz[..., 2].min(), xyz[..., 2].max()
    center = xyz.reshape(-1, 3).mean(axis=0)
    span = max(x_max - x_min, y_max - y_min, z_max - z_min, 0.5) / 2 + 0.2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(frame_idx):
        ax.cla()
        pts = xyz[frame_idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="tab:blue", s=20)
        for j, parent in enumerate(SMPL_PARENTS):
            if parent < 0:
                continue
            xs = [pts[j, 0], pts[parent, 0]]
            ys = [pts[j, 1], pts[parent, 1]]
            zs = [pts[j, 2], pts[parent, 2]]
            ax.plot(xs, ys, zs, c="tab:orange", linewidth=2)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(max(0.0, center[2] - span), center[2] + span)
        ax.set_title(f"{args.clip_id}  frame {frame_idx}/{num_frames}")
        ax.set_box_aspect([1, 1, 1])

    anim = animation.FuncAnimation(fig, draw_frame, frames=num_frames, interval=1000 / args.fps)
    writer = animation.FFMpegWriter(fps=args.fps)
    anim.save(args.output, writer=writer)
    plt.close(fig)
    print(f"-> Saved {args.output}")


if __name__ == "__main__":
    main()
