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
Diagnose whether shape-variation training failures on the 150motion/128shape ablation
correlate with body-shape "extremity" (pointing at kinematic infeasibility of specific
shapes) or are roughly shape-independent (pointing at optimization interference from
sharing one policy trunk across 128 body shapes, since jerk-reduction levers -- mass-
scaled gains, per-segment gains, soft tracking, AMP -- all plateau in the same success-
rate band regardless of how much they cut control instability; see note/README.note.md
§44-46).

Reads every results/<experiment>/failed_motions/failed_motions_epoch_*_rank_*.txt file.
Each line is already a GLOBAL motion_lib id -- MimicEvaluator._save_failed_motions writes
global ids directly (mapped from local eval-subset indices before the file is written, see
mimic_evaluator.py:230-239/261-263), so no rank-based index arithmetic is needed here,
unlike tools/analyse_failed_clip_overlap.py's older per-rank clip-major layout assumption
for a different (1024-clip, multi-GPU sharded) dataset.

Loads the same MotionLib .pt file used for training and reports:

1. Per-shape (128 body assets) failure-event count vs. that shape's beta L2-norm
   ("extremity") -- Pearson r. High |r| => certain shapes are genuinely harder
   (kinematic infeasibility of the reference for that body). Near-zero r => shape
   extremity does not explain failures.
2. Per-clip failure-event count, ranked -- lets you eyeball whether the same small set
   of "hard" clips (crawl/kneel/squat/backward, per the single-shape neutral analysis in
   note.md §17) dominates here too (motion-content-driven, not shape-driven), or whether
   a much broader/different set of clips is failing that were fine in the single-shape
   case.

Usage (run on the pod, where both the failed_motions/ logs and the motion .pt file live):
    python tools/analyze_shape_failure_correlation.py \\
        --results-dir results/hhi_wide_150motion_128shape_seggain \\
        --motion-file /workspace/motion_cache/small150_128shape.pt
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import torch

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


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom == 0:
        return float("nan")
    return (x @ y / denom).item()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="e.g. results/hhi_wide_150motion_128shape_seggain",
    )
    parser.add_argument(
        "--motion-file",
        type=Path,
        required=True,
        help="Same .pt file passed as --motion-file for this run",
    )
    parser.add_argument("--top-n-clips", type=int, default=30)
    args = parser.parse_args()

    failed_dir = args.results_dir / "failed_motions"
    if not failed_dir.exists():
        raise SystemExit(f"No failed_motions dir at {failed_dir}")

    print(f"Loading motion library from {args.motion_file} ...")
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=str(args.motion_file)), device="cpu"
    )
    if not motion_lib.has_morphology_metadata():
        raise SystemExit(
            "MotionLib has no morphology metadata (betas/gender/asset ids) -- "
            "can't do shape correlation."
        )

    fail_counts, n_epochs = read_failed_motion_events(failed_dir)
    total_events = sum(fail_counts.values())
    print(
        f"Read {n_epochs} eval epochs, {len(fail_counts)} unique motion ids, "
        f"{total_events} total failure events.\n"
    )

    # ---- 1. per-shape (asset) extremity correlation ----
    asset_to_motion_ids = motion_lib.build_asset_id_to_motion_ids()
    asset_ids = sorted(asset_to_motion_ids.keys())

    shape_extremity = []
    shape_fail_events = []
    for asset_id in asset_ids:
        motion_ids = asset_to_motion_ids[asset_id]
        betas = motion_lib.get_motion_betas(motion_ids[:1])[0]  # same body, every clip
        shape_extremity.append(betas.norm().item())
        shape_fail_events.append(
            sum(fail_counts.get(int(mid), 0) for mid in motion_ids)
        )

    r = pearson_r(torch.tensor(shape_extremity), torch.tensor(shape_fail_events).float())

    print("=== Per-shape failure-count vs. beta-L2-norm ('extremity') ===")
    print(f"Pearson r = {r:.3f}  (n={len(asset_ids)} shapes)")
    print(
        "High |r| => harder shapes are genuinely more failure-prone (kinematic "
        "infeasibility).\nNear-zero r => shape extremity does not explain failures.\n"
    )

    order = sorted(range(len(asset_ids)), key=lambda i: -shape_fail_events[i])
    print(f"{'asset_id':<20} {'fail_events':>12} {'beta_L2':>10}")
    for i in order[:15]:
        print(f"{asset_ids[i]:<20} {shape_fail_events[i]:>12} {shape_extremity[i]:>10.2f}")
    print("...")
    for i in order[-15:]:
        print(f"{asset_ids[i]:<20} {shape_fail_events[i]:>12} {shape_extremity[i]:>10.2f}")

    # ---- 2. per-clip failure ranking ----
    if motion_lib.has_clip_identity_metadata():
        clip_to_motion_ids = motion_lib.build_clip_id_to_motion_ids()
        clip_fail_events = {
            clip_id: sum(fail_counts.get(int(mid), 0) for mid in mids)
            for clip_id, mids in clip_to_motion_ids.items()
        }
        clip_n_shapes = {clip_id: len(mids) for clip_id, mids in clip_to_motion_ids.items()}
        ranked_clips = sorted(clip_fail_events.items(), key=lambda kv: -kv[1])

        print(
            f"\n=== Top {args.top_n_clips} clips by total failure events "
            "(summed over all their shape variants) ==="
        )
        print(f"{'clip_id':<40} {'fail_events':>12} {'per_shape':>10}")
        for clip_id, events in ranked_clips[: args.top_n_clips]:
            n_shapes = clip_n_shapes[clip_id]
            print(f"{clip_id:<40} {events:>12} {events / n_shapes:>10.3f}")
    else:
        print("\nMotionLib has no clip identity metadata -- skipping per-clip ranking.")


if __name__ == "__main__":
    main()
