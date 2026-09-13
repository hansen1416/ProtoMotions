# Physical Analysis Per-Class Distribution Plot (2026-09-13)

Follow-up to the physical analysis in `paper/sections/08_physical_analysis.tex` Section 7
(Table 13, dynamic-vs-quasi-static pooled comparison). The section's own hypothesis framing
calls for "per-class distributions," but Table 13 pools all clips into just two groups
(dynamic: running+jumping+kicking, quasi-static: standing+sitting). This note documents the
per-class breakdown added as Figure 11.

## What was computed

For each of the 150 development clips, the standard deviation of each metric across its 128
trained-shape replicates (absolute units -- the same per-clip-demeaned-residual quantity Table
13 already uses, just kept at the per-clip level instead of pooled into 2 groups). Grouped by
all 8 semantic classes (assign-to-all for multi-class clips, same as everywhere else in this
analysis), not just the 2 pooled ones.

No new GPU replay needed -- this is a pure re-aggregation of
`data_cache/morphology_motion_interaction.csv` (already computed and committed) against
`data_cache/clip_semantic_classes.json`.

Script: `tools/compute_physical_analysis_per_class_distributions.py`.

## Result

No class or pair of classes stands out as consistently more shape-sensitive on either metric
plotted (normalized torque, normalized jerk) -- heavy overlap across all 8 classes, consistent
with (not additional independent evidence beyond) the null result already reported in Table 13.
The visibly wider boxes for sitting (n=4), kicking (n=7), and dancing (n=8) track their small
clip counts rather than a distinct physical effect -- their upper whiskers are driven by one or
two clips each, not a genuine population spread.

Full per-class quartile/whisker numbers (median normalized-jerk std, as one representative
column):

| Class | n | Median jerk std |
|---|---:|---:|
| walking | 66 | 370.6 |
| standing | 36 | 319.3 |
| other | 40 | 208.8 |
| kicking | 7 | 411.6 |
| dancing | 8 | 359.8 |
| sitting | 4 | 495.3 |
| running | 10 | 266.4 |
| jumping | 10 | 231.7 |

Note the ordering does not separate dynamic (running/jumping/kicking) from quasi-static
(sitting/standing) classes at all -- walking (the largest, best-powered class) sits at the top,
sitting (smallest class, n=4) is second, and running/jumping (dynamic) are near the bottom. This
is the same non-monotonic, non-separating pattern already established by the formal Levene's
test in Table 13; the plot is a visual complement, not a new statistical claim.

## Files

- `data_cache/physical_analysis_per_class.per_clip.csv` -- one row per clip: its semantic
  class(es), shape count, and cross-shape std for all 3 metrics (torque, power, jerk).
- `data_cache/physical_analysis_per_class.boxplot_summary.csv` -- the per-class
  quartile/whisker numbers, one row per (class, metric) -- this is the exact table the LaTeX
  figure's `boxplot prepared` values were copied from (cross-checked: matches to the shown
  precision).
- `tools/compute_physical_analysis_per_class_distributions.py` -- the script.
- Paper: `paper/sections/08_physical_analysis.tex`, Figure 11.
