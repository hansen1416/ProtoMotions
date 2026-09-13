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
Physical analysis (paper section 8), aggregation step. See
note/README.physical-analysis-plan.md section 3.

Input: the per-(clip_id, asset_id) table from tools/analyze_morphology_motion_interaction.py
(mean torque, power, jerk, tracking success, ...) plus the clip -> semantic-class labels
from tools/classify_clip_semantics.py.

For each metric and each semantic class: the cross-shape distribution of that metric,
summarized by coefficient of variation (CV = std/mean) per clip, pooled across every clip
in the class. Not a single correlation number -- the paper's own framing ("A concise
result would state that relationship together with per-class distributions, not a single
correlation coefficient"). CV (rather than raw std) is used because it normalizes out the
fact that dynamic motions may simply have higher absolute torque *and* higher variance for
trivial reasons -- CV asks "how much does shape matter relative to this motion's own
baseline demand," closer to the paper's actual claim.

Failed rollouts (tracking failure during replay) are excluded before computing spread
statistics -- a fallen-over body's torque trace reflects a physics failure, not the
motion's control demand on that shape.

Usage:
    python tools/aggregate_morphology_motion_interaction.py \\
        --table data_cache/morphology_motion_interaction.csv \\
        --clip-classes data_cache/clip_semantic_classes.json \\
        --output data_cache/morphology_motion_interaction.per_class_cv.csv
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

METRICS = [
    "mean_abs_torque_raw",
    "mean_abs_torque_normalized",
    "mean_power_raw",
    "mean_power_mass_normalized",
    "normalized_jerk",
]


def read_table(path: Path) -> List[dict]:
    import csv

    with open(path) as f:
        return list(csv.DictReader(f))


def coefficient_of_variation(values: List[float]) -> float:
    mean = statistics.mean(values)
    if mean == 0:
        return float("nan")
    stdev = statistics.pstdev(values)
    return stdev / abs(mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=Path("data_cache/morphology_motion_interaction.csv"))
    parser.add_argument(
        "--clip-classes", type=Path, default=Path("data_cache/clip_semantic_classes.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data_cache/morphology_motion_interaction.per_class_cv.csv")
    )
    parser.add_argument("--min-shapes-per-clip", type=int, default=2)
    args = parser.parse_args()

    rows = read_table(args.table)
    clip_classes: Dict[str, List[str]] = json.loads(args.clip_classes.read_text())

    total_rows = len(rows)
    failed_rows = [r for r in rows if int(r["failed"]) == 1]
    rows = [r for r in rows if int(r["failed"]) == 0]
    print(f"Loaded {total_rows} rows, excluding {len(failed_rows)} failed rollouts "
          f"({100 * len(failed_rows) / total_rows:.1f}%), {len(rows)} remain.")

    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip_id"]].append(r)

    # per-clip CV, per metric
    per_clip_cv: Dict[str, Dict[str, float]] = {}
    per_clip_n_shapes: Dict[str, int] = {}
    for clip_id, clip_rows in by_clip.items():
        if len(clip_rows) < args.min_shapes_per_clip:
            continue
        per_clip_n_shapes[clip_id] = len(clip_rows)
        per_clip_cv[clip_id] = {}
        for metric in METRICS:
            values = [float(r[metric]) for r in clip_rows]
            per_clip_cv[clip_id][metric] = coefficient_of_variation(values)

    dropped_clips = set(by_clip.keys()) - set(per_clip_cv.keys())
    if dropped_clips:
        print(f"Dropped {len(dropped_clips)} clips with < {args.min_shapes_per_clip} surviving "
              f"shapes after failure exclusion: {sorted(dropped_clips)}")

    unclassified_clips = set(per_clip_cv.keys()) - set(clip_classes.keys())
    if unclassified_clips:
        print(f"WARNING: {len(unclassified_clips)} clips in the table have no semantic-class "
              f"label: {sorted(unclassified_clips)[:10]}")

    # pool per-clip CVs into per-class distributions (a multi-class clip contributes to
    # every class it matched -- assign-to-all, see tools/classify_clip_semantics.py)
    per_class_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for clip_id, cv_by_metric in per_clip_cv.items():
        classes = clip_classes.get(clip_id, ["other"])
        for cls in classes:
            for metric in METRICS:
                per_class_values[cls][metric].append(cv_by_metric[metric])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out_f:
        header = ["semantic_class", "n_clips"] + [
            f"{metric}_cv_{stat}" for metric in METRICS for stat in ("mean", "median", "min", "max")
        ]
        out_f.write(",".join(header) + "\n")

        print(f"\n{'class':12s} {'n_clips':8s}", "  ".join(f"{m:28s}" for m in METRICS))
        for cls in sorted(per_class_values.keys()):
            row = [cls, str(len(per_class_values[cls][METRICS[0]]))]
            summary_strs = []
            for metric in METRICS:
                vals = [v for v in per_class_values[cls][metric] if v == v]  # drop NaN
                if not vals:
                    row += ["", "", "", ""]
                    summary_strs.append(f"{'n/a':28s}")
                    continue
                mean_v = statistics.mean(vals)
                median_v = statistics.median(vals)
                min_v, max_v = min(vals), max(vals)
                row += [f"{mean_v:.4f}", f"{median_v:.4f}", f"{min_v:.4f}", f"{max_v:.4f}"]
                summary_strs.append(f"mean={mean_v:.3f} med={median_v:.3f} [{min_v:.3f},{max_v:.3f}]")
            out_f.write(",".join(row) + "\n")
            print(f"{cls:12s} {len(per_class_values[cls][METRICS[0]]):<8d}", "  ".join(summary_strs))

    print(f"\nPer-class CV summary written to {args.output}")
    print(
        "\nHypothesis check: dynamic classes (running/jumping/kicking) should show higher "
        "cross-shape CV than quasi-static classes (sitting/other-mostly-standing) -- read "
        "mean_abs_torque_normalized_cv and normalized_jerk_cv columns above."
    )


if __name__ == "__main__":
    main()
