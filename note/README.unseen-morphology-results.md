# Unseen-Morphology Generalization: Results (2026-09-13)

Executed the evaluation specified in `note/README.heldout-pipeline.md` and
`paper/sections/07_results_and_failure_analysis.tex` Section 6.5.1. This
resolves the "planned unseen-morphology evaluation" placeholder that was
sitting unrun in the paper.

## Data provenance

- **32 held-out shapes**: `protomotions/data/assets/mjcf/smpl_mor_interp/` (16
  beta vectors x 2 genders), sampled in the same `[-3,3]^10` range as the 128
  trained shapes but never used in training. Assets + `assets.yaml` were
  already generated and checked into the repo (git-lfs) prior to this
  session -- Steps 1-2 of `README.heldout-pipeline.md`.
- **Motion source**: R2 `humos_unseen_morphology_offset/` -- 128 shards
  (`humos_32768_NNNN_offset.pt`), 8 clips x 32 shapes = 256 motions/shard,
  128 x 256 = 32,768 motions total. Merged into one file with
  `tools/merge_motion_shards.py --num-shards 1` ->
  `data_cache/unseen_morphology_merged_0.pt` (14.5 GB).
- **Critical check before trusting this data**: an earlier HUMOS inference
  run for this same interp shape set was known-bad (`README.note.md` search
  "717" / "infer.py concatenates train+val+test splits") -- it confounded
  motion-OOD with shape-OOD and was explicitly marked not usable. This R2
  dataset is a **different, later run**: 1024 clips (not 717), and verified
  against `data/hhi_stage2_splits/hhi_stage2_v1/{train,validation,test}_ids.txt`
  before use:
  - 932/1024 clip ids are in the **training** partition -- the checkpoint has
    seen the motion before, only the body is new. This is the clean
    shape-only generalization test.
  - 47 in validation, 45 in test (92 total) -- both motion and shape are
    unseen simultaneously for these; reported separately, not mixed into the
    headline number.
  - 0/1024 unmatched.
- **Caveat carried into the paper**: this motion set only went through the
  frame-zero grounding offset (`compute_humos_frame0_offsets.py`, Step 5b of
  the heldout pipeline), not the fuller multi-stage refinement (contact-aware
  quaternion filtering, root XY correction) that the main training corpus
  went through. Possible small domain gap between what the policy trained on
  and what it's evaluated on here; not expected to explain the size of the
  effect found below (see diagnostic).

## Method

- `tools/analyze_unseen_morphology_generalization.py`: same batched
  one-motion-per-env deterministic replay pattern as
  `tools/analyze_morphology_motion_interaction.py` (physical analysis), but
  simplified to the metrics the paper's main tables already use: success
  (`max_gt_error <= 0.5 m` over the full ~198-step episode), mean position
  error, mean rotation error. Robot config's `asset.asset_folder_name` /
  `asset_info_file` repointed at `smpl_mor_interp/` instead of `smpl_mor/`,
  `selected_asset_ids=None` so all 32 new shapes get discovered.
  `reference_body_masses` (mass-scaled PD gains) is untouched -- it's a
  concrete dict already resolved at training time, not re-derived from the
  asset folder at eval time.
- 4096 envs (32 shapes x 128 clip-replicas/shape), 8 rounds, full 1024 clips
  x 32 shapes = 32,768 pairs. Checkpoint: same
  `hhi_wide_stage2_discover_attention_slot_type_refined` (epoch 28,170) used
  throughout the rest of the paper.
- `tools/aggregate_unseen_morphology_generalization.py`: joins clip split
  membership + Euclidean distance (raw 10-D beta space, same-gender nearest
  neighbor) from each unseen shape to its nearest trained shape
  (`protomotions/data/assets/mjcf/smpl_mor/assets.yaml`).

## Result

**28 of 32 unseen shapes generalize at essentially the trained-body level,
with no decline across the full evaluated distance range (2.58-5.05):**

| Population (train-split clips unless noted) | n | Success | mean gt_err | mean gr_err |
|---|---:|---:|---:|---:|
| Clean shapes (28/32), train-split clips | 26,096 | 99.75% | 0.093 m | 0.200 rad |
| Clean shapes (28/32), validation+test clips (both motion & shape unseen) | 2,576 | 99.61% | 0.093 m | 0.204 rad |
| Trained-body validation/test, for reference (Table 7 in paper) | -- | 98.47% / 99.14% | 0.070 / 0.068 m | 0.174 / 0.172 rad |

This is the strong form of the paper's own stated hypothesis test: performance
stays flat with distance from the training set rather than declining,
supporting morphology-conditioned control over memorization of the 128
trained configurations.

**4 of 32 shapes fail severely** (pooled: 35.25% success, mean gt_err 0.76 m,
mean gr_err 1.10 rad, train-split clips):

| Beta key | Failing variant | Same-beta other-gender variant | Distance to nearest trained |
|---|---:|---:|---:|
| a83170f5 | male: 58.3% | female: 100.0% | 3.24 |
| cf3d5ee7 | male: 46.8% | female: 99.8% | 3.53 |
| 4d1b9df1 | female: 35.8% | male: 99.8% | 3.53 |
| 1d4da893 | female: 0.1% | male: 99.8% | 3.67 |

## Diagnostic: why these 4 are flagged as a likely asset artifact, not a real shape effect

For every one of the 4 failing shapes, the **identical beta vector applied to
the other gender's body template succeeds at 99.8-100%**. If the failure were
a genuine consequence of that region of shape space being dynamically hard
(e.g., an awkward limb-length ratio), both gender variants sharing the same
coefficients should be similarly affected -- SMPL beta effects are broadly
analogous in direction across the two gender templates. Instead we see a
clean binary split by gender at fixed coefficients, which points to a
gender-specific problem in that one MJCF asset (degenerate collision
geometry, a bad limb-length interaction with that specific base template,
etc.) rather than a controller limitation.

Things checked and ruled out as an *easy* explanation:
- **File corruption / truncation**: all 4 problem assets' XML files are
  normal size (~22.1-22.2 KB), same as the 28 clean ones. No obvious
  truncation.
- **Extreme individual beta components**: the failing shapes' beta vectors
  aren't more extreme componentwise than several *successful* shapes (e.g.
  `female_5c5b9586` has a component at 2.96, right at the sampling boundary,
  and succeeds 100%).
- **Simulator termination flag**: `terminated` is 0 for all 32,768 rows
  (eval mode disables training termination, as documented elsewhere in the
  paper) -- so the failure isn't a detected "fall," it's silent tracking
  divergence that the disabled-termination eval loop doesn't stop.
- **Not a distance effect**: the distance-stratified bins are non-monotonic
  -- e.g. bin [3.20, 3.82) has 67.5% pooled success while both neighboring
  bins have ~99.9% -- a signature of specific outliers contaminating one
  bin, not a smooth decline with distance.

**Not done** (would be needed to actually confirm root cause, left as
follow-up): direct visual/mesh inspection of the 4 flagged MJCF assets
(e.g. load in a viewer, check for self-intersecting collision capsules or a
degenerate joint limit), and a rendered rollout video of one of the 4
failing cases to see qualitatively what the body is doing (falling,
spinning, stuck).

## Files

- `data_cache/unseen_morphology_generalization.csv` -- raw per-(clip, shape)
  replay output, 32,768 rows.
- `data_cache/unseen_morphology_generalization.joined.csv` -- same, with
  `distance_to_nearest_trained` and `clip_split` (train/validation/test)
  joined in. This is the file the numbers above and the paper come from.
- `data_cache/unseen_morphology_generalization.distance_bins.csv` -- the
  4-bin distance-stratified summary.
- `tools/analyze_unseen_morphology_generalization.py` -- the replay script.
- `tools/aggregate_unseen_morphology_generalization.py` -- the join/aggregate
  script (also prints the full per-shape breakdown table and the outlier
  check).
- Paper: `paper/sections/07_results_and_failure_analysis.tex` Section 6.5.1,
  Table 11 (flagged shapes), Figure 10 (distance scatter).
- `data_cache/unseen_morphology_merged_0.pt` (14.5 GB, not committed --
  gitignored, local to whichever pod ran this) is the merged motion library
  if this needs to be re-run or extended.
