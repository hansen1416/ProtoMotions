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
Paired checkpoint comparison on the (clip, shape) replay tables written by
tools/analyze_unseen_morphology_generalization.py. See
note/README.unseen-morphology-checkpoint-comparison.md for the full protocol.

Every CSV passed in must come from the SAME motion file and the SAME evaluation settings; only the
checkpoint differs. The comparison is paired over clips: each bootstrap resample draws clips with
replacement and applies the same draw to the baseline and to the other run, so the CI reflects
clip-level pairing and not independent samples.

Runs:
  --baseline  NAME=csv   the frozen checkpoint everything is measured against
  --candidate NAME=csv   the checkpoint under test (e.g. the fulldata polish)
  --reference NAME=csv   (repeatable) runs that define the noise band: a re-run of the baseline
                         checkpoint (simulator nondeterminism) and other checkpoints from the same
                         lineage (checkpoint-to-checkpoint variation). Each is compared against
                         the baseline exactly like the candidate.

Groups of shapes (--group NAME=asset_id,asset_id,... or NAME=path/to/file.txt, repeatable). Shapes
not in any user group form the automatic group "rest"; "all" is always added; --bottom-k adds the
K shapes with the lowest BASELINE success. All group numbers are pooled over the shapes in the group
and averaged per clip before bootstrapping, so a clip is the resampling unit.

Usage:
    python tools/compare_checkpoints_unseen_morphology.py \\
        --baseline refined_run1=results/analysis/x/unseen_refined_run1.csv \\
        --reference refined_run2=results/analysis/x/unseen_refined_run2.csv \\
        --candidate fulldata=results/analysis/x/unseen_fulldata.csv \\
        --clip-ids-file data/splits/hhi_stage2_v1/train_ids.txt \\
        --group flagged4=male_a83170f5,male_cf3d5ee7,female_4d1b9df1,female_1d4da893 \\
        --group mild=male_95749fe2 \\
        --output-prefix results/analysis/x/compare_unseen_train932

Self-test (no GPU, synthetic data):
    python tools/compare_checkpoints_unseen_morphology.py --self-test
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

# Order matches the last axis of the per-run grids.
METRICS = ("success", "mean_gt_error", "mean_gr_error")


def read_ids(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def split_named(spec):
    if "=" not in spec:
        raise SystemExit(f"Expected NAME=value, got {spec!r}")
    name, value = spec.split("=", 1)
    return name.strip(), value.strip()


def load_run(csv_path):
    rows = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["clip_id"], row["asset_id"])] = (
                float(row["success"]),
                float(row["mean_gt_error"]),
                float(row["mean_gr_error"]),
            )
    if not rows:
        raise SystemExit(f"{csv_path} has no rows")
    return rows


def build_grids(runs, clip_filter):
    """Intersect (clip, asset) pairs across runs and return dense [clip, asset, metric] grids."""
    common = set.intersection(*[set(rows.keys()) for rows in runs.values()])
    if clip_filter is not None:
        common = {key for key in common if key[0] in clip_filter}
    if not common:
        raise SystemExit("No (clip, shape) pairs are shared by all runs after filtering")
    assets = sorted({asset for _, asset in common})
    assets_per_clip = {}
    for clip, asset in common:
        assets_per_clip.setdefault(clip, set()).add(asset)
    clips = sorted(c for c, present in assets_per_clip.items() if len(present) == len(assets))
    dropped = len(assets_per_clip) - len(clips)
    if not clips:
        raise SystemExit("No clip has every shape in every run")
    grids = {}
    for name, rows in runs.items():
        grid = np.empty((len(clips), len(assets), len(METRICS)), dtype=np.float64)
        for ci, clip in enumerate(clips):
            for ai, asset in enumerate(assets):
                grid[ci, ai] = rows[(clip, asset)]
        grids[name] = grid
    return clips, assets, grids, dropped


def paired_delta(base_vec, other_vec, iters, rng):
    """Mean of (other - base) over clips with a clip-level paired bootstrap 95% CI."""
    diff = other_vec - base_vec
    draws = rng.integers(0, diff.size, size=(iters, diff.size))
    boot = diff[draws].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compare_group(base, other, asset_idx, iters, rng):
    result = {}
    for m, metric in enumerate(METRICS):
        base_vec = base[:, asset_idx, m].mean(axis=1)
        other_vec = other[:, asset_idx, m].mean(axis=1)
        delta, lo, hi = paired_delta(base_vec, other_vec, iters, rng)
        result[metric] = (float(base_vec.mean()), float(other_vec.mean()), delta, lo, hi)
    base_ok = base[:, asset_idx, 0] > 0.5
    other_ok = other[:, asset_idx, 0] > 0.5
    result["fixed"] = int((~base_ok & other_ok).sum())
    result["broken"] = int((base_ok & ~other_ok).sum())
    return result


def resolve_groups(group_specs, assets, base_grid, bottom_k):
    groups = {}
    for spec in group_specs:
        name, value = split_named(spec)
        if len(value.split(",")) == 1 and Path(value).is_file():
            members = read_ids(value)
        else:
            members = [item.strip() for item in value.split(",") if item.strip()]
        unknown = [m for m in members if m not in assets]
        if unknown:
            print(f"WARNING group {name}: {len(unknown)} ids not in the data: {unknown[:6]}")
        groups[name] = [assets.index(m) for m in members if m in assets]
    if groups:
        taken = {i for members in groups.values() for i in members}
        rest = [i for i in range(len(assets)) if i not in taken]
        if rest:
            groups["rest"] = rest
    if bottom_k > 0:
        order = np.argsort(base_grid[:, :, 0].mean(axis=0))[:bottom_k]
        groups[f"bottom{bottom_k}_by_baseline"] = [int(i) for i in order]
    groups["all"] = list(range(len(assets)))
    return {name: idx for name, idx in groups.items() if idx}


def pct(x):
    return f"{100.0 * x:6.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", help="NAME=csv")
    parser.add_argument("--candidate", help="NAME=csv")
    parser.add_argument("--reference", action="append", default=[], help="NAME=csv, repeatable")
    parser.add_argument("--clip-ids-file", type=Path, default=None, help="Restrict to these clip ids")
    parser.add_argument("--group", action="append", default=[], help="NAME=id,id,... or NAME=file")
    parser.add_argument("--bottom-k", type=int, default=0)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-shapes", type=int, default=40, help="Rows of the per-shape table to print")
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.baseline or not args.candidate:
        raise SystemExit("--baseline and --candidate are required")

    base_name, base_path = split_named(args.baseline)
    cand_name, cand_path = split_named(args.candidate)
    others = [(cand_name, cand_path)] + [split_named(spec) for spec in args.reference]
    if len({base_name} | {n for n, _ in others}) != 1 + len(others):
        raise SystemExit("Run names must be unique")

    runs = {base_name: load_run(base_path)}
    for name, path in others:
        runs[name] = load_run(path)

    clip_filter = set(read_ids(args.clip_ids_file)) if args.clip_ids_file else None
    clips, assets, grids, dropped = build_grids(runs, clip_filter)
    print(f"Runs: {list(runs)}")
    print(f"Paired grid: {len(clips)} clips x {len(assets)} shapes (clips dropped for incomplete rows: {dropped})")
    rng = np.random.default_rng(args.seed)

    base = grids[base_name]
    groups = resolve_groups(args.group, assets, base, args.bottom_k)

    # ---- group-level table: candidate and every reference vs baseline -------------------------
    group_rows = []
    print("\n=== Group-level paired deltas vs baseline (percentage points for success) ===")
    header = (
        f"{'group':26s} {'run':16s} {'shapes':>6s} {'base%':>7s} {'run%':>7s} {'d_succ':>8s} "
        f"{'95% CI':>17s} {'fixed':>6s} {'broke':>6s} {'d_gt(m)':>9s} {'d_gr(rad)':>10s}"
    )
    print(header)
    for group_name, idx in groups.items():
        for run_name, _ in others:
            r = compare_group(base, grids[run_name], idx, args.iters, rng)
            s, g, rr = r["success"], r["mean_gt_error"], r["mean_gr_error"]
            tag = "CANDIDATE" if run_name == cand_name else "reference"
            print(
                f"{group_name:26s} {run_name:16s} {len(idx):6d} {pct(s[0])}  {pct(s[1])}  "
                f"{100 * s[2]:+7.2f}  [{100 * s[3]:+6.2f},{100 * s[4]:+6.2f}] "
                f"{r['fixed']:6d} {r['broken']:6d} {g[2]:+9.4f} {rr[2]:+10.4f}  {tag}"
            )
            group_rows.append(
                {
                    "group": group_name, "run": run_name, "role": tag, "n_shapes": len(idx),
                    "n_clips": len(clips), "base_success": s[0], "run_success": s[1],
                    "d_success": s[2], "d_success_lo": s[3], "d_success_hi": s[4],
                    "fixed_pairs": r["fixed"], "broken_pairs": r["broken"],
                    "d_gt_error": g[2], "d_gt_lo": g[3], "d_gt_hi": g[4],
                    "d_gr_error": rr[2], "d_gr_lo": rr[3], "d_gr_hi": rr[4],
                }
            )

    # ---- noise-band summary: is the candidate delta larger than any reference delta? ----------
    print("\n=== Candidate vs noise band (success delta, percentage points) ===")
    for group_name in groups:
        cand = next(r for r in group_rows if r["group"] == group_name and r["role"] == "CANDIDATE")
        refs = [r for r in group_rows if r["group"] == group_name and r["role"] == "reference"]
        ci_excludes_zero = cand["d_success_lo"] > 0 or cand["d_success_hi"] < 0
        if refs:
            band = max(abs(r["d_success"]) for r in refs)
            beyond = ci_excludes_zero and abs(cand["d_success"]) > band
            print(
                f"{group_name:26s} cand {100 * cand['d_success']:+7.2f}  max|ref| {100 * band:6.2f}  "
                f"CI excludes 0: {ci_excludes_zero!s:5s}  beyond noise band: {beyond}"
            )
        else:
            print(f"{group_name:26s} cand {100 * cand['d_success']:+7.2f}  (no --reference runs: no noise band)")

    # ---- per-shape table, candidate vs baseline -----------------------------------------------
    member_of = {}
    for group_name, idx in groups.items():
        if group_name in ("all",) or group_name.startswith("bottom"):
            continue
        for i in idx:
            member_of.setdefault(i, group_name)
    cand_grid = grids[cand_name]
    order = np.argsort(base[:, :, 0].mean(axis=0))
    shape_rows = []
    for i in order:
        r = compare_group(base, cand_grid, [int(i)], args.iters, rng)
        s, g, rr = r["success"], r["mean_gt_error"], r["mean_gr_error"]
        shape_rows.append(
            {
                "asset_id": assets[i], "group": member_of.get(int(i), ""), "n_clips": len(clips),
                "base_success": s[0], "cand_success": s[1], "d_success": s[2],
                "d_success_lo": s[3], "d_success_hi": s[4],
                "fixed_pairs": r["fixed"], "broken_pairs": r["broken"],
                "d_gt_error": g[2], "d_gt_lo": g[3], "d_gt_hi": g[4],
                "d_gr_error": rr[2], "d_gr_lo": rr[3], "d_gr_hi": rr[4],
            }
        )
    print(f"\n=== Per-shape, {cand_name} vs {base_name} (lowest baseline success first, "
          f"showing {min(args.print_shapes, len(shape_rows))} of {len(shape_rows)}) ===")
    print(f"{'shape':22s} {'group':10s} {'base%':>7s} {'cand%':>7s} {'d_succ':>8s} {'95% CI':>17s} {'fixed':>6s} {'broke':>6s}")
    for row in shape_rows[: args.print_shapes]:
        print(
            f"{row['asset_id']:22s} {row['group']:10s} {pct(row['base_success'])}  {pct(row['cand_success'])}  "
            f"{100 * row['d_success']:+7.2f}  [{100 * row['d_success_lo']:+6.2f},{100 * row['d_success_hi']:+6.2f}] "
            f"{row['fixed_pairs']:6d} {row['broken_pairs']:6d}"
        )
    regress = [r for r in shape_rows if r["d_success_hi"] < -0.02]
    print(f"\nShapes whose success CI lies entirely below -2 pp (regression candidates): {len(regress)}")
    for row in regress:
        print(f"  {row['asset_id']}: {100 * row['d_success']:+.2f} pp  [{100 * row['d_success_lo']:+.2f},{100 * row['d_success_hi']:+.2f}]")

    if args.output_prefix is not None:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix, rows in ((".groups.csv", group_rows), (".per_shape.csv", shape_rows)):
            path = Path(str(args.output_prefix) + suffix)
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {path}")
    return 0


def self_test():
    rng = np.random.default_rng(1)
    clips = [f"clip{i}" for i in range(80)]
    assets = [f"shape{i}" for i in range(6)]
    fields = ["clip_id", "asset_id", "motion_id", "episode_len_steps", "success", "terminated",
              "max_gt_error", "mean_gt_error", "mean_gr_error"]

    def write(path, p_success):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for c in clips:
                for a_i, a in enumerate(assets):
                    ok = int(rng.random() < p_success[a_i])
                    writer.writerow({
                        "clip_id": c, "asset_id": a, "motion_id": 0, "episode_len_steps": 100,
                        "success": ok, "terminated": 0, "max_gt_error": 0.2 if ok else 0.9,
                        "mean_gt_error": 0.08 if ok else 0.6, "mean_gr_error": 0.2 if ok else 1.0,
                    })

    base_p = [0.30, 0.99, 0.99, 0.99, 0.99, 0.99]
    cand_p = [0.85, 0.99, 0.99, 0.99, 0.99, 0.99]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write(tmp / "base.csv", base_p)
        write(tmp / "ref.csv", base_p)
        write(tmp / "cand.csv", cand_p)
        (tmp / "ids.txt").write_text("\n".join(clips[:70]) + "\n")
        main([
            "--baseline", f"base={tmp / 'base.csv'}",
            "--reference", f"rerun={tmp / 'ref.csv'}",
            "--candidate", f"cand={tmp / 'cand.csv'}",
            "--clip-ids-file", str(tmp / "ids.txt"),
            "--group", "flagged=shape0",
            "--bottom-k", "2", "--iters", "500",
            "--output-prefix", str(tmp / "out" / "cmp"),
        ])
        groups_csv = list(csv.DictReader(open(tmp / "out" / "cmp.groups.csv")))
        shape_csv = list(csv.DictReader(open(tmp / "out" / "cmp.per_shape.csv")))
        flagged = next(r for r in groups_csv if r["group"] == "flagged" and r["role"] == "CANDIDATE")
        assert float(flagged["d_success_lo"]) > 0.2, "improved shape should show a clearly positive CI"
        assert shape_csv[0]["asset_id"] == "shape0", "lowest-baseline shape should be listed first"
        assert len(shape_csv) == len(assets)
    print("\nSELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
