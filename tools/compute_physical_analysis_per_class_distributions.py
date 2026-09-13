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
Physical analysis (paper Section 7), per-class distribution figure. Follow-up to
tools/aggregate_morphology_motion_interaction.py and
tools/test_morphology_motion_variance_significance.py, whose dynamic-vs-quasi-static
2-group comparison (Table 13 in the paper) pools all classes into just two buckets. The
paper's own hypothesis framing asks for per-class distributions, not a pooled number, so
this script produces the per-clip statistic underlying Figure 11: for each clip, the
standard deviation of a metric across its 128 trained-shape replicates (absolute units,
same quantity Table 13 pools, just kept at the per-clip level and grouped by all 8
semantic classes instead of 2).

Usage:
    python tools/compute_physical_analysis_per_class_distributions.py \\
        --table data_cache/morphology_motion_interaction.csv \\
        --clip-classes data_cache/clip_semantic_classes.json \\
        --output-prefix data_cache/physical_analysis_per_class
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

METRICS = ["mean_abs_torque_normalized", "mean_power_mass_normalized", "normalized_jerk"]
CLASS_ORDER = ["walking", "running", "jumping", "kicking", "dancing", "sitting", "standing", "other"]


def percentile(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    idx = min(n - 1, max(0, round(p * (n - 1))))
    return sorted_vals[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=Path("data_cache/morphology_motion_interaction.csv"))
    parser.add_argument("--clip-classes", type=Path, default=Path("data_cache/clip_semantic_classes.json"))
    parser.add_argument("--output-prefix", type=Path, default=Path("data_cache/physical_analysis_per_class"))
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(args.table)) if int(r["failed"]) == 0]
    clip_classes: Dict[str, List[str]] = json.loads(args.clip_classes.read_text())

    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip_id"]].append(r)

    # --- per-clip table: one row per clip, its classes, and its cross-shape std per metric ---
    per_clip_rows = []
    for clip_id, clip_rows in sorted(by_clip.items()):
        classes = clip_classes.get(clip_id, ["other"])
        row = {"clip_id": clip_id, "classes": ";".join(classes), "n_shapes": len(clip_rows)}
        for metric in METRICS:
            vals = [float(r[metric]) for r in clip_rows]
            row[f"std_{metric}"] = statistics.pstdev(vals)
        per_clip_rows.append(row)

    per_clip_path = Path(str(args.output_prefix) + ".per_clip.csv")
    with open(per_clip_path, "w", newline="") as f:
        fieldnames = ["clip_id", "classes", "n_shapes"] + [f"std_{m}" for m in METRICS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_clip_rows)
    print(f"Per-clip cross-shape std written to {per_clip_path} ({len(per_clip_rows)} clips).")

    # --- per-class box-plot summary (the exact numbers hardcoded into Figure 11) ---
    per_class_vals = {m: defaultdict(list) for m in METRICS}
    for row in per_clip_rows:
        for cls in row["classes"].split(";"):
            for metric in METRICS:
                per_class_vals[metric][cls].append(row[f"std_{metric}"])

    summary_rows = []
    print(f"\n{'class':10s} {'metric':30s} {'n':>4s} {'lw':>10s} {'q1':>10s} {'median':>10s} {'q3':>10s} {'uw':>10s}")
    for metric in METRICS:
        for cls in CLASS_ORDER:
            vals = sorted(per_class_vals[metric].get(cls, []))
            if not vals:
                continue
            lw, q1, med, q3, uw = (
                percentile(vals, 0.05), percentile(vals, 0.25), percentile(vals, 0.50),
                percentile(vals, 0.75), percentile(vals, 0.95),
            )
            summary_rows.append({
                "metric": metric, "class": cls, "n": len(vals),
                "lower_whisker": lw, "q1": q1, "median": med, "q3": q3, "upper_whisker": uw,
            })
            print(f"{cls:10s} {metric:30s} {len(vals):4d} {lw:10.5g} {q1:10.5g} {med:10.5g} {q3:10.5g} {uw:10.5g}")

    summary_path = Path(str(args.output_prefix) + ".boxplot_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "class", "n", "lower_whisker", "q1", "median", "q3", "upper_whisker"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nPer-class box-plot summary (quartiles, 5th/95th percentile whiskers) written to {summary_path}")


if __name__ == "__main__":
    main()
