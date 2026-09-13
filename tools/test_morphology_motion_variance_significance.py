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
Physical analysis (paper section 8), significance check. Follow-up to
tools/aggregate_morphology_motion_interaction.py after its per-clip CV comparison turned
out to be misleading: CV inflates for near-zero-mean quasi-static clips (small
denominator), and raw per-clip std pooled across only 4-66 clips/class was too small a
sample to call anything either way by eye. This script asks the question properly.

Method (CPU-only, reuses the existing 19,200-row table -- no new GPU replay):
1. Per-clip normalize: residual = value / clip's own across-128-shape mean. This removes
   each clip's baseline demand level (the near-zero-mean artifact that broke plain CV) while
   keeping relative cross-shape sensitivity, and it's what lets us pool rows *within* a class
   instead of being stuck comparing 4-66 clip-level summary numbers.
2. Pool all (clip, shape) residuals belonging to a class (a clip contributes to every class
   it matched, same as the CV aggregation). This turns e.g. "sitting" from 4 numbers into
   4 x 128 = 512 residuals -- much more power to estimate that class's *typical* cross-shape
   spread, though see the caveat below.
3. Test whether "dynamic" (running + jumping + kicking) and "quasi-static"
   (standing + sitting) pooled residuals have different variance: Levene's test (median-
   centered / Brown-Forsythe, robust to non-normality), the direct paper hypothesis test.
   Also run the omnibus Levene's test across all classes for completeness.
4. Bootstrap CI on each group's pooled std, resampling at the CLIP level (not the row level)
   with replacement -- rows within a clip aren't independent draws of "how shape-sensitive
   this class is," they're 128 shape-views of the *same* clip, so resampling rows directly
   would understate the true uncertainty. Resampling clips (then taking all their rows) is
   the valid way to reflect that only ~10 (or 4) independent clips actually define a small
   class, even though each is measured precisely.

Caveat this does NOT fix: per-clip normalization removes each clip's baseline level, but a
class's evidence is still ultimately "however many clips matched it." A class with 4 clips
can have its *typical* spread estimated precisely (512 residuals), but whether those 4
clips are representative of "sitting motions in general" is a sample-size-of-4 question no
amount of row pooling can answer -- the clip-level bootstrap CI is what surfaces that
honestly (wide CIs for small classes) rather than hiding it.

Usage:
    python tools/test_morphology_motion_variance_significance.py \\
        --table data_cache/morphology_motion_interaction.csv \\
        --clip-classes data_cache/clip_semantic_classes.json \\
        --output data_cache/morphology_motion_interaction.variance_significance.txt
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

METRICS = ["mean_abs_torque_normalized", "mean_power_mass_normalized", "normalized_jerk"]
DYNAMIC_CLASSES = {"running", "jumping", "kicking"}
QUASI_STATIC_CLASSES = {"standing", "sitting"}
N_BOOTSTRAP = 10000
RNG_SEED = 0


def read_table(path: Path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_clip_residuals(rows: List[dict], metric: str, kind: str) -> Dict[str, np.ndarray]:
    """clip_id -> array of per-shape residuals, either:
    - 'ratio': value / clip_mean (scale-free, but blows up for near-zero-mean clips --
      mathematically the same failure mode as plain CV)
    - 'demeaned': value - clip_mean (keeps physical units, no small-denominator issue, but
      not scale-free -- a clip with genuinely larger absolute demand can have a larger
      absolute spread for reasons unrelated to "shape sensitivity" per se)
    Both are reported so a significant 'ratio' result can be checked against whether it
    survives in absolute terms, or is a near-zero-baseline artifact."""
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip_id"]].append(float(r[metric]))
    residuals = {}
    for clip_id, values in by_clip.items():
        arr = np.array(values, dtype=np.float64)
        mean = arr.mean()
        if kind == "ratio":
            if mean == 0:
                continue
            residuals[clip_id] = arr / mean
        elif kind == "demeaned":
            residuals[clip_id] = arr - mean
        else:
            raise ValueError(kind)
    return residuals


def pooled_residuals_for_classes(
    clip_residuals: Dict[str, np.ndarray], clip_classes: Dict[str, List[str]], classes: set
) -> np.ndarray:
    member_clips = [c for c, labels in clip_classes.items() if set(labels) & classes and c in clip_residuals]
    if not member_clips:
        return np.array([])
    return np.concatenate([clip_residuals[c] for c in member_clips])


def bootstrap_std_ci(
    clip_residuals: Dict[str, np.ndarray], member_clips: List[str], rng: np.random.Generator, n_boot: int
) -> tuple:
    """Resample CLIPS with replacement (not rows) -- respects that 128 shape-rows per clip
    aren't independent evidence about the class, only the clip is."""
    if not member_clips:
        return float("nan"), float("nan"), float("nan")
    boot_stds = np.empty(n_boot)
    n = len(member_clips)
    for b in range(n_boot):
        sampled = rng.choice(member_clips, size=n, replace=True)
        pooled = np.concatenate([clip_residuals[c] for c in sampled])
        boot_stds[b] = pooled.std()
    point = np.concatenate([clip_residuals[c] for c in member_clips]).std()
    lo, hi = np.percentile(boot_stds, [2.5, 97.5])
    return point, lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=Path("data_cache/morphology_motion_interaction.csv"))
    parser.add_argument("--clip-classes", type=Path, default=Path("data_cache/clip_semantic_classes.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data_cache/morphology_motion_interaction.variance_significance.txt")
    )
    args = parser.parse_args()

    rows = read_table(args.table)
    rows = [r for r in rows if int(r["failed"]) == 0]
    clip_classes: Dict[str, List[str]] = json.loads(args.clip_classes.read_text())
    rng = np.random.default_rng(RNG_SEED)

    lines = []

    def emit(s: str = ""):
        print(s)
        lines.append(s)

    for metric in METRICS:
        emit(f"\n{'=' * 70}\n{metric}\n{'=' * 70}")

        for kind in ("ratio", "demeaned"):
            emit(f"\n--- residual kind: {kind} "
                 f"{'(scale-free, but near-zero-mean-sensitive -- same failure mode as CV)' if kind == 'ratio' else '(absolute units, not scale-free)'} ---")
            clip_residuals = build_clip_residuals(rows, metric, kind)

            dynamic_clips = [c for c, labels in clip_classes.items() if set(labels) & DYNAMIC_CLASSES and c in clip_residuals]
            quasi_clips = [c for c, labels in clip_classes.items() if set(labels) & QUASI_STATIC_CLASSES and c in clip_residuals]
            emit(f"dynamic (running+jumping+kicking): {len(dynamic_clips)} clips")
            emit(f"quasi-static (standing+sitting):    {len(quasi_clips)} clips")

            dyn_pooled = np.concatenate([clip_residuals[c] for c in dynamic_clips])
            qs_pooled = np.concatenate([clip_residuals[c] for c in quasi_clips])

            # Levene's test (median-centered = Brown-Forsythe), the paper's direct hypothesis:
            # does cross-shape variance differ between dynamic and quasi-static motions?
            stat, p_value = stats.levene(dyn_pooled, qs_pooled, center="median")
            emit(f"\nLevene's test (dynamic vs. quasi-static), Brown-Forsythe (median-centered):")
            emit(f"  W = {stat:.3f}, p = {p_value:.4g}"
                 f"  {'-- SIGNIFICANT at p<0.05' if p_value < 0.05 else '-- NOT significant at p<0.05'}")

            dyn_point, dyn_lo, dyn_hi = bootstrap_std_ci(clip_residuals, dynamic_clips, rng, N_BOOTSTRAP)
            qs_point, qs_lo, qs_hi = bootstrap_std_ci(clip_residuals, quasi_clips, rng, N_BOOTSTRAP)
            emit(f"\nBootstrap 95% CI on pooled residual std (resampled at the clip level, "
                 f"n_boot={N_BOOTSTRAP}):")
            emit(f"  dynamic:      std = {dyn_point:.4f}  CI = [{dyn_lo:.4f}, {dyn_hi:.4f}]  (n_clips={len(dynamic_clips)})")
            emit(f"  quasi-static: std = {qs_point:.4f}  CI = [{qs_lo:.4f}, {qs_hi:.4f}]  (n_clips={len(quasi_clips)})")
            overlap = not (dyn_hi < qs_lo or qs_hi < dyn_lo)
            emit(f"  CIs {'OVERLAP -- no evidence of a real difference' if overlap else 'DO NOT OVERLAP -- evidence of a real difference'}")

            # Omnibus Levene's across all 8 classes, for completeness.
            all_classes = sorted({c for labels in clip_classes.values() for c in labels})
            groups = []
            group_names = []
            for cls in all_classes:
                member_clips = [c for c, labels in clip_classes.items() if cls in labels and c in clip_residuals]
                if len(member_clips) < 2:
                    continue
                groups.append(np.concatenate([clip_residuals[c] for c in member_clips]))
                group_names.append(f"{cls}(n={len(member_clips)})")
            stat_all, p_all = stats.levene(*groups, center="median")
            emit(f"\nOmnibus Levene's test across all classes [{', '.join(group_names)}]:")
            emit(f"  W = {stat_all:.3f}, p = {p_all:.4g}"
                 f"  {'-- SIGNIFICANT' if p_all < 0.05 else '-- NOT significant'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
