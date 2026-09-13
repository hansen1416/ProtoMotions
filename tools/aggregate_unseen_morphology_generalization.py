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
Unseen-morphology generalization, aggregation step. See
tools/analyze_unseen_morphology_generalization.py and note/README.heldout-pipeline.md.

Joins the per-(clip, unseen-shape) replay table against:
1. Clip split membership (train/validation/test of hhi_stage2_v1) -- only TRAIN-split clips
   isolate the shape axis cleanly (the policy has seen the motion, never this body). Clips
   whose ids fall in validation/test are reported separately since for those, both the motion
   and the shape are unseen, confounding the two axes.
2. Euclidean distance in raw 10-D beta space from each unseen shape to its nearest trained
   shape of the same gender (128 trained shapes in protomotions/data/assets/mjcf/smpl_mor/).

Reports success rate and component errors as a function of that distance (paper's own planned
readout, Section 6.5.1), and flags any single unseen shape whose aggregate looks like an outlier
relative to the other 31 (e.g. a degenerate asset) before it contaminates the headline claim.

Usage:
    python tools/aggregate_unseen_morphology_generalization.py \\
        --table data_cache/unseen_morphology_generalization.csv \\
        --trained-assets-yaml protomotions/data/assets/mjcf/smpl_mor/assets.yaml \\
        --unseen-assets-yaml protomotions/data/assets/mjcf/smpl_mor_interp/assets.yaml \\
        --train-ids data_cache/hhi_stage2_v1_train_ids.txt \\
        --validation-ids data_cache/hhi_stage2_v1_validation_ids.txt \\
        --test-ids data_cache/hhi_stage2_v1_test_ids.txt \\
        --output-prefix data_cache/unseen_morphology_generalization
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def load_betas(assets_yaml: Path) -> Dict[str, dict]:
    entries = yaml.safe_load(assets_yaml.read_text())
    return {e["asset_id"]: e for e in entries}


def nearest_trained_distance(unseen_betas: np.ndarray, unseen_gender: str, trained: Dict[str, dict]) -> float:
    same_gender = [
        np.array(e["betas"], dtype=np.float64)
        for e in trained.values()
        if e["gender"] == unseen_gender
    ]
    dists = [float(np.linalg.norm(unseen_betas - b)) for b in same_gender]
    return min(dists)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=Path("data_cache/unseen_morphology_generalization.csv"))
    parser.add_argument(
        "--trained-assets-yaml", type=Path,
        default=Path("protomotions/data/assets/mjcf/smpl_mor/assets.yaml"),
    )
    parser.add_argument(
        "--unseen-assets-yaml", type=Path,
        default=Path("protomotions/data/assets/mjcf/smpl_mor_interp/assets.yaml"),
    )
    parser.add_argument("--train-ids", type=Path, default=Path("data_cache/hhi_stage2_v1_train_ids.txt"))
    parser.add_argument("--validation-ids", type=Path, default=Path("data_cache/hhi_stage2_v1_validation_ids.txt"))
    parser.add_argument("--test-ids", type=Path, default=Path("data_cache/hhi_stage2_v1_test_ids.txt"))
    parser.add_argument("--num-distance-bins", type=int, default=4)
    parser.add_argument("--output-prefix", type=Path, default=Path("data_cache/unseen_morphology_generalization"))
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.table)))
    print(f"Loaded {len(rows)} (clip, shape) rows.")

    trained = load_betas(args.trained_assets_yaml)
    unseen = load_betas(args.unseen_assets_yaml)
    print(f"{len(trained)} trained shapes, {len(unseen)} unseen shapes.")

    asset_distance: Dict[str, float] = {}
    for asset_id, entry in unseen.items():
        betas = np.array(entry["betas"], dtype=np.float64)
        asset_distance[asset_id] = nearest_trained_distance(betas, entry["gender"], trained)

    train_ids = set(l.strip() for l in args.train_ids.read_text().splitlines() if l.strip())
    val_ids = set(l.strip() for l in args.validation_ids.read_text().splitlines() if l.strip())
    test_ids = set(l.strip() for l in args.test_ids.read_text().splitlines() if l.strip())

    def split_of(clip_id: str) -> str:
        if clip_id in train_ids:
            return "train"
        if clip_id in val_ids:
            return "validation"
        if clip_id in test_ids:
            return "test"
        return "unknown"

    for r in rows:
        r["distance_to_nearest_trained"] = asset_distance[r["asset_id"]]
        r["clip_split"] = split_of(r["clip_id"])

    split_counts = defaultdict(int)
    for r in rows:
        split_counts[r["clip_split"]] += 1
    print("Row counts by clip split:", dict(split_counts))

    # --- per-shape outlier check, across ALL rows regardless of split ---
    print("\n=== Per-shape aggregate (all clips, sanity/outlier check) ===")
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset_id"]].append(r)
    per_asset_summary = []
    for asset_id, ars in sorted(by_asset.items(), key=lambda kv: asset_distance[kv[0]]):
        succ = sum(int(r["success"]) for r in ars) / len(ars)
        mean_gt = statistics.mean(float(r["mean_gt_error"]) for r in ars)
        per_asset_summary.append((asset_id, asset_distance[asset_id], succ, mean_gt, len(ars)))
        print(f"  {asset_id:16s} dist={asset_distance[asset_id]:.3f}  n={len(ars):4d}  "
              f"success={100*succ:5.1f}%  mean_gt_err={mean_gt:.4f}")

    succ_rates = [s[2] for s in per_asset_summary]
    med_succ = statistics.median(succ_rates)
    mad = statistics.median([abs(s - med_succ) for s in succ_rates]) or 1e-6
    outliers = [s for s in per_asset_summary if abs(s[2] - med_succ) / mad > 5]
    if outliers:
        print(f"\nOutlier shape(s) (success rate >5 MAD from median {100*med_succ:.1f}%):")
        for o in outliers:
            print(f"  {o[0]}: success={100*o[2]:.1f}%")
    else:
        print(f"\nNo shape's success rate is a >5-MAD outlier from the median ({100*med_succ:.1f}%).")

    # --- headline result: TRAIN-split clips only (clean shape-only generalization) ---
    for split_name, split_rows in [
        ("train (shape-only generalization)", [r for r in rows if r["clip_split"] == "train"]),
        ("validation+test (shape AND motion unseen)",
         [r for r in rows if r["clip_split"] in ("validation", "test")]),
    ]:
        if not split_rows:
            continue
        n = len(split_rows)
        succ = sum(int(r["success"]) for r in split_rows) / n
        mean_gt = statistics.mean(float(r["mean_gt_error"]) for r in split_rows)
        mean_gr = statistics.mean(float(r["mean_gr_error"]) for r in split_rows)
        print(f"\n=== {split_name}: n={n} ===")
        print(f"  success rate: {100*succ:.2f}%")
        print(f"  mean gt_err:  {mean_gt:.4f}")
        print(f"  mean gr_err:  {mean_gr:.4f}")

    # --- distance-stratified result on train-split clips ---
    train_rows = [r for r in rows if r["clip_split"] == "train"]
    distances = sorted(set(asset_distance.values()))
    dmin, dmax = min(distances), max(distances)
    n_bins = args.num_distance_bins
    bin_edges = [dmin + (dmax - dmin) * i / n_bins for i in range(n_bins + 1)]

    def bin_of(d):
        for i in range(n_bins):
            if d <= bin_edges[i + 1] or i == n_bins - 1:
                return i
        return n_bins - 1

    binned = defaultdict(list)
    for r in train_rows:
        binned[bin_of(r["distance_to_nearest_trained"])].append(r)

    print(f"\n=== Distance-stratified result (train-split clips only, {n_bins} bins) ===")
    csv_rows = []
    for i in range(n_bins):
        brs = binned.get(i, [])
        if not brs:
            continue
        n = len(brs)
        succ = sum(int(r["success"]) for r in brs) / n
        mean_gt = statistics.mean(float(r["mean_gt_error"]) for r in brs)
        mean_gr = statistics.mean(float(r["mean_gr_error"]) for r in brs)
        lo, hi = bin_edges[i], bin_edges[i + 1]
        print(f"  dist [{lo:.2f},{hi:.2f}): n={n:5d}  success={100*succ:5.1f}%  "
              f"mean_gt_err={mean_gt:.4f}  mean_gr_err={mean_gr:.4f}")
        csv_rows.append({
            "dist_lo": lo, "dist_hi": hi, "n": n,
            "success_rate": succ, "mean_gt_error": mean_gt, "mean_gr_error": mean_gr,
        })

    out_csv = Path(str(args.output_prefix) + ".distance_bins.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dist_lo", "dist_hi", "n", "success_rate", "mean_gt_error", "mean_gr_error"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nDistance-bin table written to {out_csv}")

    # Also dump the full per-row joined table for the paper/plotting.
    out_full = Path(str(args.output_prefix) + ".joined.csv")
    fieldnames = list(rows[0].keys())
    with open(out_full, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Full joined per-row table written to {out_full}")


if __name__ == "__main__":
    main()
