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

"""Failed clip overlap analysis across all 150-clip/128-shape "discover" lineage runs.

Reads failed_motion log files (single rank, rank_0 only -- these are all single-GPU runs)
from every run directory in `results/` matching the 150-clip/128-shape ablation set, maps
per-run motion IDs -> global clip IDs, computes a per-run "persistent failure" set (>=50%
of the last WINDOW_EPOCHS eval epochs), then cross-tabulates: for each of the 150 clips, in
how many of the N runs is it a persistent failure. Clips persistent across (nearly) every
run -- regardless of architecture (flat/MoE/attention), reward shaping (amp/seggain/
softtrack/relaxed_rh), or temporal features (historical/lookahead/window_match/sharpen/
explore) -- are the intrinsic-difficulty class the reward-function/capacity levers can't
reach; clips persistent in only a few runs are lever-sensitive.

Motion ID layout (single rank, clip-major ordering, matches build_small_multishape_subset.py):
  global_clip_idx = motion_id // SHAPES_PER_CLIP   (0-149)
  clip_name       = clip_ids[global_clip_idx]      (from data_cache/small150_128shape.clip_ids.txt)
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis")
CLIP_IDS_FILE = Path("data_cache/small150_128shape.clip_ids.txt")

SHAPES_PER_CLIP = 128
TOTAL_CLIPS = 150

WINDOW_EPOCHS = 10
PERSISTENT_THRESHOLD = WINDOW_EPOCHS // 2  # >=5/10

RUN_DIRS = sorted(RESULTS_DIR.glob("hhi_wide_150motion*"))


def read_failed_motions_dir(run_dir: Path) -> dict[int, list[int]]:
    """Return {epoch: [global_clip_idx, ...]} for rank_0 failed-motion logs."""
    fm_dir = run_dir / "failed_motions"
    if not fm_dir.exists():
        return {}
    pattern = re.compile(r"failed_motions_epoch_(\d+)_rank_0\.txt")
    result: dict[int, list[int]] = {}
    for f in fm_dir.iterdir():
        m = pattern.match(f.name)
        if not m:
            continue
        epoch = int(m.group(1))
        local_ids = [int(line.strip()) for line in f.read_text().splitlines() if line.strip()]
        global_clip_ids = sorted({mid // SHAPES_PER_CLIP for mid in local_ids})
        result[epoch] = global_clip_ids
    return result


def failure_frequency(epoch_clips: dict[int, list[int]], window: int) -> dict[int, int]:
    epochs = sorted(epoch_clips)[-window:]
    freq: dict[int, int] = defaultdict(int)
    for ep in epochs:
        for clip_idx in epoch_clips[ep]:
            freq[clip_idx] += 1
    return dict(freq), epochs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clip_names = [l.strip() for l in CLIP_IDS_FILE.read_text().splitlines() if l.strip()]
    assert len(clip_names) == TOTAL_CLIPS, f"expected {TOTAL_CLIPS} clip names, got {len(clip_names)}"

    per_run_persistent: dict[str, set[int]] = {}
    per_run_freq: dict[str, dict[int, int]] = {}
    per_run_meta: dict[str, dict] = {}

    for run_dir in RUN_DIRS:
        name = run_dir.name
        epoch_clips = read_failed_motions_dir(run_dir)
        if not epoch_clips:
            print(f"WARNING: no failed_motions data for {name}, skipping")
            continue
        freq, window_epochs = failure_frequency(epoch_clips, WINDOW_EPOCHS)
        persistent = {c for c, f in freq.items() if f >= PERSISTENT_THRESHOLD}
        per_run_persistent[name] = persistent
        per_run_freq[name] = freq
        latest_epoch = max(epoch_clips)
        per_run_meta[name] = {
            "latest_epoch": latest_epoch,
            "latest_failed_count": len(epoch_clips[latest_epoch]),
            "num_eval_points": len(epoch_clips),
            "window_epochs": window_epochs,
        }

    run_names = sorted(per_run_persistent)
    n_runs = len(run_names)

    # Cross-run tally: for each clip, how many runs is it persistent in
    clip_run_count: dict[int, int] = defaultdict(int)
    clip_runs: dict[int, list[str]] = defaultdict(list)
    for name, persistent in per_run_persistent.items():
        for c in persistent:
            clip_run_count[c] += 1
            clip_runs[c].append(name)

    universal = {c for c, n in clip_run_count.items() if n == n_runs}
    near_universal = {c for c, n in clip_run_count.items() if n >= n_runs - 1 and n < n_runs}
    majority = {c for c, n in clip_run_count.items() if n >= (n_runs + 1) // 2}
    never_persistent = set(range(TOTAL_CLIPS)) - set(clip_run_count)

    lines = []
    lines.append("# Failed Clip Overlap Analysis — 150-Clip/128-Shape Discover Lineage")
    lines.append("")
    lines.append(f"Runs analyzed ({n_runs}):")
    for name in run_names:
        m = per_run_meta[name]
        lines.append(
            f"- `{name}`: latest epoch {m['latest_epoch']}, "
            f"{m['latest_failed_count']}/150 failed at latest eval, "
            f"{len(per_run_persistent[name])} persistent (>= {PERSISTENT_THRESHOLD}/{WINDOW_EPOCHS} "
            f"of last {len(m['window_epochs'])} evals)"
        )
    lines.append("")
    lines.append(
        f"Persistence threshold: a clip counts as a run's \"persistent failure\" if it appears "
        f"in >= {PERSISTENT_THRESHOLD} of that run's last {WINDOW_EPOCHS} eval checkpoints "
        f"(controls for single-epoch eval noise; runs stopped at different epoch counts, so this "
        f"compares each run's own converged/plateaued regime, not a fixed global step)."
    )
    lines.append("")
    lines.append("## Cross-run overlap summary")
    lines.append("")
    lines.append(f"| Category | Count | Fraction of 150 |")
    lines.append("|---|---|---|")
    lines.append(f"| Persistent in ALL {n_runs} runs (universal hard class) | **{len(universal)}** | {len(universal)/TOTAL_CLIPS:.1%} |")
    lines.append(f"| Persistent in {n_runs-1}/{n_runs} runs (near-universal) | **{len(near_universal)}** | {len(near_universal)/TOTAL_CLIPS:.1%} |")
    lines.append(f"| Persistent in >= half the runs | **{len(majority)}** | {len(majority)/TOTAL_CLIPS:.1%} |")
    lines.append(f"| Never a persistent failure in ANY run | **{len(never_persistent)}** | {len(never_persistent)/TOTAL_CLIPS:.1%} |")
    lines.append("")

    lines.append(f"## Universal hard class (n={len(universal)}) — persistent in every one of {n_runs} runs")
    lines.append("")
    lines.append("clip_idx | clip_name")
    lines.append("---|---")
    for c in sorted(universal):
        lines.append(f"{c:3d} | {clip_names[c]}")
    lines.append("")

    lines.append(f"## Near-universal ({n_runs-1}/{n_runs} runs, n={len(near_universal)})")
    lines.append("")
    lines.append("clip_idx | clip_name | missing_from")
    lines.append("---|---|---")
    for c in sorted(near_universal):
        missing = sorted(set(run_names) - set(clip_runs[c]))
        lines.append(f"{c:3d} | {clip_names[c]} | {', '.join(missing)}")
    lines.append("")

    lines.append("## Full per-clip run-count distribution")
    lines.append("")
    lines.append("How many of the 150 clips are persistent in exactly K of the runs:")
    lines.append("")
    for k in range(n_runs, -1, -1):
        n = sum(1 for c in range(TOTAL_CLIPS) if clip_run_count.get(c, 0) == k)
        if n:
            lines.append(f"  {k:2d}/{n_runs} runs: {n:3d} clips {'█' * min(n, 60)}")
    lines.append("")

    lines.append("## Per-clip detail (every clip that is persistent in >=1 run)")
    lines.append("")
    lines.append("clip_idx | clip_name | #runs | runs")
    lines.append("---|---|---|---")
    for c in sorted(clip_run_count, key=lambda c: -clip_run_count[c]):
        lines.append(f"{c:3d} | {clip_names[c]} | {clip_run_count[c]}/{n_runs} | {', '.join(sorted(clip_runs[c]))}")
    lines.append("")

    output = "\n".join(lines)
    out_path = OUTPUT_DIR / "T_failed_clip_overlap_150.md"
    out_path.write_text(output)
    print(output)
    print(f"\n-> Written to {out_path}")


if __name__ == "__main__":
    main()
