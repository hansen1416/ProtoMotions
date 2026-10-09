"""Compare complete replay CSVs on exactly the same unseen clip/body pairs.

Uses only the Python standard library. Split labels from the original joined
CSV describe the ORIGINAL checkpoint's exposure, not the full-data checkpoint.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["clip_id"], row["asset_id"]
            if key in rows:
                raise ValueError(f"Duplicate pair in {path}: {key}")
            for field in ("mean_gt_error", "mean_gr_error", "max_gt_error"):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Non-finite {field} in {path}: {key}")
            if row["success"] not in ("0", "1"):
                raise ValueError(f"Invalid success in {path}: {key}")
            rows[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("Pair sets differ: "
                         f"missing={len(baseline.keys() - candidate.keys())}, "
                         f"extra={len(candidate.keys() - baseline.keys())}")
    clips = {k[0] for k in baseline}
    assets = {k[1] for k in baseline}
    if len(clips) != 1024 or len(assets) != 32 or len(baseline) != 32768:
        raise ValueError("Expected the complete 1,024 x 32 baseline panel")
    groups = defaultdict(list)
    for key, old in baseline.items():
        new = candidate[key]
        if old["episode_len_steps"] != new["episode_len_steps"]:
            raise ValueError(f"Episode horizons differ: {key}")
        pair = old, new
        groups[("all", "all")].append(pair)
        groups[("asset", key[1])].append(pair)
        if "clip_split" in old:
            groups[("original_motion_split", old["clip_split"])].append(pair)
    summaries = []
    for (kind, group), pairs in sorted(groups.items()):
        old_success = statistics.mean(int(a["success"]) for a, b in pairs) * 100
        new_success = statistics.mean(int(b["success"]) for a, b in pairs) * 100
        summary = dict(group_type=kind, group=group, pairs=len(pairs),
                       baseline_success_pct=old_success,
                       candidate_success_pct=new_success,
                       success_change_pp=new_success - old_success,
                       recovered=sum(a["success"] == "0" and b["success"] == "1" for a, b in pairs),
                       regressed=sum(a["success"] == "1" and b["success"] == "0" for a, b in pairs))
        for field in ("mean_gt_error", "mean_gr_error", "max_gt_error"):
            summary["baseline_" + field] = statistics.mean(float(a[field]) for a, b in pairs)
            summary["candidate_" + field] = statistics.mean(float(b[field]) for a, b in pairs)
        summaries.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Validated {len(baseline)} matched pairs; wrote {args.output}")
    print("Original split labels are historical: full-data training includes all three splits.")


if __name__ == "__main__":
    main()
