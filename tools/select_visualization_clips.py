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
Pick N persistently-failing and N persistently-succeeding clips from a run's own eval
history, and emit a ready-to-run bash script that renders each to an MP4 via
record_video_mor.py.

Why clip-level, not exact (clip, shape) motion_id: MimicEvaluator._save_failed_motions
logs the exact global motion_id (one specific body shape) that failed each eval epoch.
But with eval_one_shape_per_motion=True (this project's default), each of the ~150 clips
is evaluated against exactly ONE randomly-sampled shape per epoch -- a specific (clip,
shape) pair is really only revisited a handful of times over a whole run, so its raw fail
count is mostly noise. The CLIP itself, though, gets evaluated on some shape every single
epoch, so aggregating fail events by clip_id over all ~n_epochs gives a real signal: a
clip with 0 events over 50 epochs genuinely almost-always succeeds regardless of shape; a
clip with a high fraction of epochs failed is genuinely hard. See
tools/analyze_shape_failure_correlation.py's STATIC mode docstring for the same
read_failed_motion_events()/build_clip_id_to_motion_ids() approach -- this script reuses
that reasoning but selects concrete --motion-index values for record_video_mor.py instead
of just printing a ranked table.

For the failed bucket, the representative clip is identified via the shape that was
*actually observed* failing most often for that clip (not an arbitrary shape). For the
success bucket, since the clip essentially never failed regardless of shape, any shape
variant of it can be used to identify the clip.

Each emitted video renders the SAME set of --num-shapes body shapes (default 16),
selected once, deterministically, from the motion file itself (every 128/--num-shapes'th
asset id in sorted order) and passed to record_video_mor.py via --target-asset-ids. This
is deterministic given the same --motion-file, so running this script separately against
different checkpoints/experiments that share the same --motion-file (e.g. comparing
several ablation runs) automatically renders the exact same 16 body shapes in every
video -- no shared state needs to be passed between invocations.

Each command also gets --compact-spawn-spacing (default 2.0m), which lays the humanoids
out in a tight grid (see protomotions/envs/base_env/env.py's compact_spawn_spacing logic)
instead of record_video_mor.py's default wider spawn layout. Without it, the fixed camera
(_setup_fixed_camera) has to zoom out far enough to fit every actor in frame, which makes
each one tiny -- --spawn-spacing controls this directly if 2.0 is still too tight/loose.

Usage (run on the pod, where results/<experiment>/failed_motions/ and the motion .pt
file live -- CPU only, no GPU/simulator needed for this selection step):

    python tools/select_visualization_clips.py \\
        --results-dir results/hhi_wide_150motion_128shape_discover \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --checkpoint results/hhi_wide_150motion_128shape_discover/last.ckpt \\
        --num-failed 4 --num-success 4 --num-shapes 16 --spawn-spacing 2.0 \\
        --script-output results/hhi_wide_150motion_128shape_discover/render_visualize.sh

Then, still on the pod (this part needs the GPU/IsaacGym):

    bash results/hhi_wide_150motion_128shape_discover/render_visualize.sh
"""

from __future__ import annotations

import argparse
import re
import stat
from collections import Counter
from pathlib import Path

from protomotions.components.motion_lib import MotionLib, MotionLibConfig

FAILED_FILE_RE = re.compile(r"failed_motions_epoch_(\d+)_rank_(\d+)\.txt")


def read_failed_motion_events(failed_motions_dir: Path) -> tuple[Counter, int]:
    """Counter[motion_id] = number of eval epochs that motion_id showed up as failed."""
    counts: Counter = Counter()
    n_epochs = 0
    for f in sorted(failed_motions_dir.iterdir()):
        if not FAILED_FILE_RE.match(f.name):
            continue
        n_epochs += 1
        ids = [int(line) for line in f.read_text().splitlines() if line.strip()]
        counts.update(ids)
    return counts, n_epochs


def select_canonical_shapes(motion_lib, num_shapes: int) -> list:
    """Deterministically pick `num_shapes` asset ids, evenly strided over the sorted,
    unique asset ids present in `motion_lib`. Deterministic given the same motion
    file, so separate invocations of this script against the same --motion-file (e.g.
    different checkpoints/experiments) always pick the exact same shapes."""
    asset_id_to_motion_ids = motion_lib.build_asset_id_to_motion_ids()
    all_asset_ids = sorted(asset_id_to_motion_ids.keys())
    if num_shapes >= len(all_asset_ids):
        return all_asset_ids
    stride = len(all_asset_ids) / num_shapes
    return [all_asset_ids[int(i * stride)] for i in range(num_shapes)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-failed", type=int, default=4)
    parser.add_argument("--num-success", type=int, default=4)
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=16,
        help="Number of body shapes rendered side-by-side in each video.",
    )
    parser.add_argument(
        "--spawn-spacing",
        type=float,
        default=2.0,
        help=(
            "Compact grid spacing (metres) between humanoids, passed to "
            "record_video_mor.py's --compact-spawn-spacing. Without this, envs use "
            "the default (much wider) spawn layout, which pushes the fixed camera "
            "back far enough to fit everyone in frame and makes each actor tiny."
        ),
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/visualize_videos",
    )
    parser.add_argument(
        "--script-output",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/render_visualize.sh",
    )
    args = parser.parse_args()

    video_output_dir = args.video_output_dir or (args.results_dir / "visualize_videos")
    script_output = args.script_output or (args.results_dir / "render_visualize.sh")

    failed_dir = args.results_dir / "failed_motions"
    if not failed_dir.exists():
        raise SystemExit(f"No failed_motions dir at {failed_dir}")

    print(f"Loading motion library from {args.motion_file} ...")
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=str(args.motion_file)), device="cpu"
    )
    if not motion_lib.has_clip_identity_metadata():
        raise SystemExit("MotionLib has no motion_clip_ids -- can't group by clip.")

    canonical_shapes = select_canonical_shapes(motion_lib, args.num_shapes)
    target_asset_ids_str = ",".join(canonical_shapes)
    print(
        f"Canonical {len(canonical_shapes)} body shapes for every video "
        f"(deterministic from {args.motion_file}):\n  {canonical_shapes}\n"
    )

    fail_counts, n_epochs = read_failed_motion_events(failed_dir)
    print(f"Read {n_epochs} eval epochs, {len(fail_counts)} unique failed motion ids.\n")

    clip_to_motion_ids = motion_lib.build_clip_id_to_motion_ids()
    clip_fail_events = {
        clip_id: sum(fail_counts.get(int(mid), 0) for mid in mids)
        for clip_id, mids in clip_to_motion_ids.items()
    }
    ranked = sorted(clip_fail_events.items(), key=lambda kv: -kv[1])  # hardest first

    failed_bucket = ranked[: args.num_failed]
    success_bucket = list(reversed(ranked[-args.num_success :]))  # easiest first

    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated by tools/select_visualization_clips.py from "
        f"{n_epochs} eval epochs in {failed_dir}",
        "set -euo pipefail",
        f'mkdir -p "{video_output_dir}"',
        "",
    ]

    print(f"=== Failed bucket: top {len(failed_bucket)} clips by total fail events ===")
    print(f"{'clip_id':<40} {'fail_events':>12} {'/':>3} {n_epochs:<6} epochs")
    for rank, (clip_id, events) in enumerate(failed_bucket, start=1):
        mids = clip_to_motion_ids[clip_id]
        # Representative shape: the one actually observed failing most often for this clip.
        rep_mid = max(mids, key=lambda mid: fail_counts.get(int(mid), 0)).item()
        print(f"{clip_id:<40} {events:>12} {'/':>3} {n_epochs:<6}  motion_id={rep_mid}")
        out = video_output_dir / f"failed_{rank:02d}_{clip_id}_events{events}.mp4"
        lines.append(
            f'python protomotions/record_video_mor.py --checkpoint "{args.checkpoint}" '
            f'--simulator isaacgym --motion-file "{args.motion_file}" '
            f'--motion-index {rep_mid} --num-envs {len(canonical_shapes)} --same-motion '
            f'--target-asset-ids "{target_asset_ids_str}" '
            f'--compact-spawn-spacing {args.spawn_spacing} --output "{out}"'
        )

    lines.append("")
    print(f"\n=== Success bucket: {len(success_bucket)} clips with fewest fail events ===")
    print(f"{'clip_id':<40} {'fail_events':>12} {'/':>3} {n_epochs:<6} epochs")
    for rank, (clip_id, events) in enumerate(success_bucket, start=1):
        rep_mid = clip_to_motion_ids[clip_id][0].item()
        print(f"{clip_id:<40} {events:>12} {'/':>3} {n_epochs:<6}  motion_id={rep_mid}")
        out = video_output_dir / f"success_{rank:02d}_{clip_id}_events{events}.mp4"
        lines.append(
            f'python protomotions/record_video_mor.py --checkpoint "{args.checkpoint}" '
            f'--simulator isaacgym --motion-file "{args.motion_file}" '
            f'--motion-index {rep_mid} --num-envs {len(canonical_shapes)} --same-motion '
            f'--target-asset-ids "{target_asset_ids_str}" '
            f'--compact-spawn-spacing {args.spawn_spacing} --output "{out}"'
        )

    script_output.parent.mkdir(parents=True, exist_ok=True)
    script_output.write_text("\n".join(lines) + "\n")
    script_output.chmod(script_output.stat().st_mode | stat.S_IEXEC)
    print(f"\nWrote {len(failed_bucket) + len(success_bucket)}-command render script "
          f"→ {script_output}")
    print(f"Run it with: bash {script_output}")


if __name__ == "__main__":
    main()
