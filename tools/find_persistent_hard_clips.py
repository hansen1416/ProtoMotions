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
Freeze the hard-clip tail for the `discover` lever lineage: intersect the persistently-failing
clips across every lever tried on the 150motion/128shape corpus (see note/README.note.md §55).

Each of `discover`, `_sharpen`, `_explore`, `_historical`, `_lookahead`, `_historical_lookahead`,
`_window_match` ruled out one independent hypothesis for the ~75-80% plateau (reward shaping,
PPO exploration, backward/forward observation context, phase alignment) without moving it. A
clip only belongs in the frozen hard set if it persistently failed under EVERY one of them --
that is a much stricter bar than any single run's own failure list, and it's the set the next
round of levers (see plan item 3) should be scored against so iteration is fast and comparable.

Motion ids in failed_motions/ logs are already global (MimicEvaluator._save_failed_motions maps
local eval-subset indices to global ids before writing -- see mimic_evaluator.py:230-239/261-263),
so no per-rank arithmetic is needed here (unlike tools/analyse_failed_clip_overlap.py's older
per-rank clip-major layout, which was for a different 1024-clip/4-GPU dataset). Clip identity
comes from MotionLib.build_clip_id_to_motion_ids(), so results are keyed by stable clip_id
strings (e.g. "M010318"), not by shape-specific motion_id -- consistent with the near-zero
shape-extremity correlation already found (tools/analyze_shape_failure_correlation.py): a clip
counts as "failed this epoch" if ANY of its 128 shape variants failed.

Run on the pod, where results/<run>/failed_motions/ for the full lineage and the motion .pt file
both live (only `discover`/`seggain`/`softtrack` are synced locally as of 2026-08-14; the lever
variants live on the pod only):

    python tools/find_persistent_hard_clips.py \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --runs hhi_wide_150motion_128shape_discover \\
               hhi_wide_150motion_128shape_discover_sharpen \\
               hhi_wide_150motion_128shape_discover_explore \\
               hhi_wide_150motion_128shape_discover_historical \\
               hhi_wide_150motion_128shape_discover_lookahead \\
               hhi_wide_150motion_128shape_discover_historical_lookahead \\
               hhi_wide_150motion_128shape_discover_window_match \\
        --output results/analysis/hard_clips_discover_lineage.md

Writes two files:
  --output                              human-readable per-run + intersection report
  <output>.clip_ids.txt (sidecar)       the frozen clip_id list, one per line -- feed this
                                         straight into tools/build_small_multishape_subset.py's
                                         --clip-ids-file to materialize the actual small
                                         hard-motion dataset.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig

FAILED_FILE_RE = re.compile(r"failed_motions_epoch_(\d+)_rank_(\d+)\.txt")


def read_epoch_clip_sets(
    results_dir: Path, motion_id_to_clip_id: Dict[int, str]
) -> Dict[int, Set[str]]:
    """Return {epoch: {clip_id, ...}} -- union across ranks, clip-deduped within an epoch."""
    failed_dir = results_dir / "failed_motions"
    if not failed_dir.exists():
        return {}

    epoch_motion_ids: Dict[int, Set[int]] = defaultdict(set)
    for f in failed_dir.iterdir():
        m = FAILED_FILE_RE.match(f.name)
        if not m:
            continue
        epoch = int(m.group(1))
        ids = (int(line) for line in f.read_text().splitlines() if line.strip())
        epoch_motion_ids[epoch].update(ids)

    epoch_clip_ids: Dict[int, Set[str]] = {}
    for epoch, motion_ids in epoch_motion_ids.items():
        clips = {
            motion_id_to_clip_id[mid]
            for mid in motion_ids
            if mid in motion_id_to_clip_id
        }
        epoch_clip_ids[epoch] = clips
    return epoch_clip_ids


def persistent_clips(
    epoch_clip_ids: Dict[int, Set[str]], window_epochs: int, threshold_fraction: float
) -> Set[str]:
    """Clips present in >= threshold_fraction of the last `window_epochs` eval epochs."""
    epochs = sorted(epoch_clip_ids)[-window_epochs:]
    if not epochs:
        return set()
    freq: Dict[str, int] = defaultdict(int)
    for ep in epochs:
        for clip_id in epoch_clip_ids[ep]:
            freq[clip_id] += 1
    threshold = threshold_fraction * len(epochs)
    return {clip_id for clip_id, count in freq.items() if count >= threshold}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--motion-file", required=True, help="Same .pt file used by every run")
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--runs", nargs="+", required=True, help="Experiment names under --results-root"
    )
    parser.add_argument("--window-epochs", type=int, default=10)
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=0.5,
        help="Per-run 'persistent' bar: fraction of the last --window-epochs eval epochs a clip "
        "must fail in. The final frozen set additionally requires this in EVERY run listed.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading motion library from {args.motion_file} ...")
    motion_lib = MotionLib(config=MotionLibConfig(motion_file=args.motion_file), device="cpu")
    if not motion_lib.has_clip_identity_metadata():
        raise SystemExit("MotionLib has no clip identity metadata -- can't group by clip_id.")

    clip_to_motion_ids = motion_lib.build_clip_id_to_motion_ids()
    motion_id_to_clip_id: Dict[int, str] = {}
    for clip_id, mids in clip_to_motion_ids.items():
        for mid in mids.tolist():
            motion_id_to_clip_id[int(mid)] = clip_id
    all_clip_ids = set(clip_to_motion_ids.keys())
    print(f"Motion library: {len(all_clip_ids)} clips, {len(motion_id_to_clip_id)} motion ids\n")

    results_root = Path(args.results_root)
    per_run_persistent: Dict[str, Set[str]] = {}
    per_run_latest_epoch: Dict[str, int] = {}
    missing_runs: List[str] = []

    for run in args.runs:
        run_dir = results_root / run
        epoch_clip_ids = read_epoch_clip_sets(run_dir, motion_id_to_clip_id)
        if not epoch_clip_ids:
            missing_runs.append(run)
            continue
        per_run_persistent[run] = persistent_clips(
            epoch_clip_ids, args.window_epochs, args.threshold_fraction
        )
        per_run_latest_epoch[run] = max(epoch_clip_ids)

    if missing_runs:
        print(
            f"WARNING: {len(missing_runs)} run(s) had no failed_motions/ dir under "
            f"{results_root} -- skipped (sync from the pod to include them):"
        )
        for run in missing_runs:
            print(f"  - {run}")
        print()

    if not per_run_persistent:
        raise SystemExit("No runs with failed_motions/ data found -- nothing to intersect.")

    intersection = set.intersection(*per_run_persistent.values())

    lines = []
    lines.append("# Frozen Hard-Clip Set — `discover` Lever Lineage")
    lines.append("")
    lines.append(
        f"Window: last {args.window_epochs} eval epochs per run; persistence threshold: "
        f">= {args.threshold_fraction:.0%} of those epochs."
    )
    lines.append(
        f"A clip is in the frozen set only if it is persistent in ALL {len(per_run_persistent)} "
        "run(s) below."
    )
    lines.append("")
    lines.append("## Per-run summary")
    lines.append("")
    lines.append("| Run | Latest epoch | Persistent clips |")
    lines.append("|---|---|---|")
    for run in args.runs:
        if run in missing_runs:
            lines.append(f"| {run} | — | *(missing, not synced)* |")
        else:
            lines.append(
                f"| {run} | {per_run_latest_epoch[run]} | {len(per_run_persistent[run])} |"
            )
    lines.append("")
    lines.append(f"## Frozen intersection: {len(intersection)} clips")
    lines.append("")
    if missing_runs:
        lines.append(
            "**Note:** the above is a partial intersection over the runs actually present -- "
            f"{len(missing_runs)} run(s) were missing. Re-run with all runs synced before "
            "treating this as final."
        )
        lines.append("")
    for clip_id in sorted(intersection):
        lines.append(f"- {clip_id}")
    lines.append("")

    output = "\n".join(lines)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(output)
    print(f"\n-> Report written to {out_path}")

    sidecar_path = out_path.with_suffix("").with_suffix(".clip_ids.txt")
    sidecar_path.write_text("\n".join(sorted(intersection)) + ("\n" if intersection else ""))
    print(f"-> Frozen clip_id list written to {sidecar_path}")


if __name__ == "__main__":
    main()
