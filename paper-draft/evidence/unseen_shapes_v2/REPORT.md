# Unseen-shape v2 evaluation: `refined` vs `fulldata` on the `inside` and `outside` datasets

Run 2026-10-09 on a RunPod A40 pod, following `note/README.unseen-shapes-v2-evaluation-handoff.md`
(with the mechanics of `note/README.unseen-morphology-checkpoint-comparison.md`). The datasets were built
in the same session following `note/README.unseen-shapes-v2-protomotions-pod-handoff.md` (appendix A).
This is a **checkpoint comparison**: training length and clip coverage changed together between the two
checkpoints, so no difference below is attributed to either factor alone.

## 1. Answer

On the headline population (233 train-split clips x 128 unseen bodies = 29,824 pairs per dataset), the
frozen paper checkpoint `refined` succeeds on **97.28%** of pairs for `inside` bodies (betas in [-3,3])
and **97.15%** for `outside` bodies (max |beta| in (3,5]), against **99.29%** on the trained-body control
(150 clips x 128 trained bodies). Within `outside`, pooled success by tier is 99.03% / 97.76% / 95.33% /
96.47% (tiers 0 to 3), so it does not fall monotonically with beta distance. The lowest success sits
with the heaviest bodies (8 `outside` bodies above 144.4 kg: 79.0% averaged over shapes) and with a
handful of individual shapes in both datasets (lowest: `male_669a0fb3` 3.0% inside, `male_1c099619`
13.3% outside). Every failing row is silent drift (`terminated=0`), none are falls. The candidate
`fulldata` (`epoch_34000.ckpt`) differs from `refined` beyond the noise band in all three populations
but not in the same direction: **-0.37 pp** on `inside` [95% CI -0.57, -0.18], **+0.42 pp** on
`outside` [+0.22, +0.61] and **-0.17 pp** on the trained-body control [-0.37, -0.03], with the rerun
of `refined` moving by at most 0.04 pp. The `outside` gain is concentrated in the bodies outside the
training mass range (+7.68 pp, 10 shapes) and in tier 3 (+1.39 pp). Per shape the picture is mixed:
on the train-split clips 3 `inside` and 6 `outside` shapes lose more than 2 pp with confidence, while
5 `outside` shapes gain (two by more than 55 pp).

## 2. Provenance

| Item | Value |
|---|---|
| Repo | `/workspace/ProtoMotions`, branch `feature/hhi`, HEAD `75324d93e1a06be90a8960a29a0c21eca92ae3c2` at analysis time (it was `466e3e9` when the session started; the user pulled during the session). No tracked file modified by this work. |
| Hardware | NVIDIA A40 (driver 580.178.04), container memory limit 50 GB |
| Protocol | `tools/analyze_unseen_morphology_generalization.py` unchanged, deterministic replay, success = max position error <= 0.5 m within at most 210 steps (script defaults), `--envs-per-asset 32` for every run (128 x 32 = 4,096 envs) |

**Checkpoint identity** (frozen copies in `results/frozen/<name>/`, each with its own `resolved_configs_inference.pt`):

| Name | Source file | sha256 | Size (B) | Epoch | step_count | Motion manifest |
|---|---|---|---:|---:|---:|---|
| `refined` (run1, run2) | `hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt` | `4b83f20c93bf95fab389fc8935685ef8a2c7e46583f765e8c802421edb4479cb` | 615,057,590 | 28,170 | 11,076,894,720 | `train_manifest.jsonl` (+ validation manifest) |
| `fulldata` | `hhi_wide_stage2_discover_attention_slot_type_fulldata/epoch_34000.ckpt` | `7b57de8ec047c390c2881f8a630d7196a09da6dd08bfbae743f36d6542cd2192` | 615,057,590 | 34,000 | 15,661,793,280 | `full_manifest.jsonl` (no validation manifest) |

Both have robot asset folder `mjcf/smpl_mor/`. The user confirmed this table before any GPU run and chose
`epoch_34000` (the pre-registered file) over `last.ckpt` (34,050) and `score_based.ckpt`.

**Runs** (all `rc=0`, sequential on one GPU):

| Run | Rows | Wall-clock (s) |
|---|---:|---:|
| ctrl refined_run1 | 19,200 | 231 |
| inside refined_run1 | 32,768 | 374 |
| inside refined_run2 | 32,768 | 367 |
| outside refined_run1 | 32,768 | 370 |
| outside refined_run2 | 32,768 | 362 |
| ctrl refined_run2 | 19,200 | 232 |
| inside fulldata | 32,768 | 365 |
| outside fulldata | 32,768 | 380 |
| ctrl fulldata | 19,200 | 248 |

Commands: the runner is `/workspace/unseen_v2/scripts/run_eval.sh` (the handoff's `run_v2` plus the
10-06 note's `run_ctrl` with `--envs-per-asset 32`); the comparisons are
`tools/compare_checkpoints_unseen_morphology.py --baseline refined_run1=... --reference refined_run2=... --candidate fulldata=... --clip-ids-file <train233|valtest23> --group <files in groups/> --bottom-k 8 --print-shapes 128`.

**Deviations from the handoff**
1. Train/validation id lists are not on R2 any more (only manifests). I rebuilt `train_ids.txt` and
   `validation_ids.txt` from `train_manifest.jsonl` / `validation_manifest.jsonl`; both match the sha256
   in `split_metadata.json` exactly. All 150 control clips and all 233 headline clips are in the train
   split; none of the 23 val/test clips are.
2. The merged motion files were built from the local refined shards just before they were uploaded (and
   `rclone check` found 0 size differences against R2), instead of downloading them back from R2.
3. The first `inside` smoke test was OOM-killed because it ran while 4 refinement processes used the 50
   GB container limit; it was rerun alone and passed. Nothing else was affected.
4. No ladder checkpoints and no reproduction of the 2026-09-13 set (user's decision), so the noise band
   rests on the `refined` rerun alone: it measures simulator nondeterminism, not checkpoint-to-checkpoint
   variation.
5. The `mass_out` / `height_out` groups are built per dataset (`inside_*`, `outside_*`), since each
   comparison runs on one dataset.

## 3. Gates

| Gate | Result |
|---|---|
| S (merged files) | PASS for both: 32,768 motions, 256 clips, 128 shapes, all 32,768 (clip, shape) pairs, asset ids equal to `assets.yaml`, no non-finite values |
| R (sanity) | PASS: comparator `--self-test` passed; `refined` on the trained-body control 99.29% (19,200 rows) |
| N (rerun) | PASS with a note, see below |

Gate N, `refined_run1` vs `refined_run2`, all 256 clips:

| Dataset | Pairs | Same `success` | Pairs that differ | run1 | run2 |
|---|---:|---:|---:|---:|---:|
| inside | 32,768 | 99.33% | 220 | 97.27% | 97.25% |
| outside | 32,768 | 98.68% | 434 | 97.12% | 97.17% |
| ctrl | 19,200 | 99.96% | 8 | 99.29% | 99.31% |

Pooled success agrees within 0.05 pp everywhere. The simulator is not bitwise deterministic: individual
pairs flip, about twice as often on `outside` as on `inside`, and the largest per-pair change in
`max_gt_error` is several metres (a pair that drifts in one run and not the other). I judged this not
material at the pooled level and continued. It does mean per-shape differences of a few pairs are noise.

## 4. Headline tables (233 train-split clips)

Success in %, deltas in percentage points, 95% bootstrap CI over clips. "Beyond noise" = CI excludes 0
and |candidate delta| > |rerun delta| (interpretation rule 1).

### inside

| Group | Shapes | refined | fulldata | Delta [95% CI] | rerun Delta | Beyond noise | d gt err (m) |
|---|---:|---:|---:|---|---:|---|---:|
| all | 128 | 97.28 | 96.91 | -0.37 [-0.57, -0.18] | -0.02 | **yes** | +0.0050 |
| rest | 124 | 97.35 | 96.94 | -0.42 [-0.62, -0.24] | -0.01 | **yes** | +0.0051 |
| inside_mass_out | 1 | 91.42 | 86.27 | -5.15 [-9.44, -0.86] | -2.15 | **yes** | +0.0312 |
| inside_height_out | 3 | 96.42 | 99.43 | +3.00 [+1.86, +4.29] | +0.29 | **yes** | -0.0100 |
| bottom8_by_baseline | 8 | 62.77 | 63.47 | +0.70 [-0.70, +2.15] | -0.48 | no | +0.0465 |

### outside

| Group | Shapes | refined | fulldata | Delta [95% CI] | rerun Delta | Beyond noise | d gt err (m) |
|---|---:|---:|---:|---|---:|---|---:|
| all | 128 | 97.15 | 97.57 | +0.42 [+0.22, +0.61] | +0.04 | **yes** | +0.0024 |
| outside_tier0 | 32 | 99.03 | 99.46 | +0.43 [+0.15, +0.70] | +0.12 | **yes** | -0.0014 |
| outside_tier1 | 32 | 97.76 | 97.81 | +0.05 [-0.24, +0.35] | -0.05 | no | +0.0044 |
| outside_tier2 | 32 | 95.33 | 95.12 | -0.21 [-0.67, +0.23] | -0.05 | no | +0.0101 |
| outside_tier3 | 32 | 96.47 | 97.87 | +1.39 [+1.07, +1.72] | +0.16 | **yes** | -0.0038 |
| outside_ratio_out | 26 | 95.43 | 95.41 | -0.02 [-0.56, +0.53] | +0.08 | no | +0.0050 |
| outside_mass_out | 10 | 83.22 | 90.90 | +7.68 [+6.39, +9.01] | +0.43 | **yes** | -0.0086 |
| outside_height_out | 7 | 97.92 | 99.82 | +1.90 [+1.29, +2.58] | +0.18 | **yes** | -0.0088 |
| bottom8_by_baseline | 8 | 61.11 | 74.41 | +13.30 [+10.94, +15.77] | +0.43 | **yes** | +0.0054 |

The tier groups cover all 128 `outside` shapes, so there is no `rest` group; groups overlap (a shape can
be in a tier and in `mass_out`). `bottom8_by_baseline` is selected from the baseline run alone and is
therefore exposed to regression to the mean (the rerun moves it by only +0.43 pp, so the effect seems
small here, but conclusions should rest on `all` and the pre-registered groups). Mean position error
moves the other way from success in several groups (e.g. `outside` `all`: success up, mean error up by
2.4 mm), so the gain is in fewer drifting pairs rather than tighter tracking overall.

### Shapes whose success CI lies entirely below -2 pp (rule 2)

| Dataset | Shape | Tier | Mass (kg) | Height (m) | M/F mass ratio | refined | fulldata | Delta [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|---|
| inside | female_e7bc8c92 | - | 79.5 | 1.32 | 1.22 | 92.3 | 79.0 | -13.30 [-18.03, -9.01] |
| inside | male_6d81a8f6 | - | 56.3 | 1.29 | 0.59 | 97.9 | 85.0 | -12.88 [-18.03, -8.15] |
| inside | male_ec3bb7f8 | - | 42.7 | 1.41 | 1.06 | 19.3 | 10.7 | -8.58 [-13.30, -4.29] |
| outside | female_edcf2449 | 2 | 181.4 | 1.51 | 0.42 | 94.8 | 58.4 | -36.48 [-43.78, -29.18] |
| outside | male_a7f6b877 | 2 | 94.3 | 1.62 | 1.86 | 65.7 | 40.8 | -24.89 [-31.33, -19.30] |
| outside | male_edcf2449 | 2 | 76.3 | 1.14 | 0.42 | 82.0 | 67.0 | -15.02 [-21.46, -8.15] |
| outside | female_e706da88 | 3 | 75.3 | 1.63 | 0.41 | 94.4 | 79.8 | -14.59 [-18.88, -10.30] |
| outside | female_bd0ec1ce | 1 | 56.4 | 1.48 | 0.75 | 63.9 | 52.8 | -11.16 [-16.74, -6.01] |
| outside | female_f9bf3075 | 3 | 207.1 | 1.58 | 0.34 | 97.4 | 88.8 | -8.58 [-12.88, -4.29] |

Shapes whose CI lies entirely above +2 pp: `inside` `female_cfc560dc` (+9.01) and `male_669a0fb3`
(+5.58, 3.0% -> 8.6%); `outside` `female_98d4329d` (+65.24, 29.6% -> 94.8%), `male_1c099619` (+57.94,
13.3% -> 71.2%), `female_33cc784f` (+15.88), `female_1b5f1e4c` (+14.59), `female_1c099619` (+7.30).
Four of the six `outside` regressions are betas with a male/female mass ratio below 0.56; both
genders of beta `edcf2449` regress. Rule 5 applies: this is not evidence of an asset defect or a bad
HUMOS reference either way.

## 5. Per-shape tables and degradation summaries

Full tables: `per_shape_inside.csv`, `per_shape_outside.csv` (asset, tier, max |beta|, L2,
nearest-trained L2, mass, height, male/female mass ratio, and success and mean position error for both
checkpoints on the 233 clips). Binned summaries (`per_shape_summary.txt`), success averaged over shapes
in each bin, refined / fulldata:

| outside bin | Shapes | refined % | fulldata % |
|---|---:|---:|---:|
| tier 0, max\|beta\| (3, 3.5] | 32 | 99.03 | 99.46 |
| tier 1, (3.5, 4] | 32 | 97.76 | 97.81 |
| tier 2, (4, 4.5] | 32 | 95.33 | 95.12 |
| tier 3, (4.5, 5] | 32 | 96.47 | 97.87 |
| mass < 26.4 kg | 2 | 100.00 | 100.00 |
| mass 26.4 to 144.4 kg (training range) | 118 | 98.33 | 98.13 |
| mass > 144.4 kg | 8 | 79.02 | 88.62 |
| height 1.60 to 1.67 m | 14 | 90.53 | 91.81 |
| height outside 1.13 to 1.67 m | 7 | 97.91 | 99.82 |
| M/F mass ratio < 0.56 | 22 | 94.63 | 94.60 |
| M/F mass ratio 0.56 to 3.21 | 102 | 97.59 | 98.11 |
| M/F mass ratio > 3.21 | 4 | 99.78 | 99.89 |

| inside bin | Shapes | refined % | fulldata % |
|---|---:|---:|---:|
| max\|beta\| < 2.5 | 24 | 99.39 | 99.00 |
| max\|beta\| 2.5 to 3 | 104 | 96.80 | 96.43 |
| mass 40 to 60 kg | 29 | 93.86 | 93.22 |
| mass 100 to 144.4 kg | 21 | 95.01 | 95.18 |
| height 1.60 to 1.67 m | 9 | 88.75 | 89.22 |
| female / male | 64 / 64 | 97.95 / 96.61 | 97.85 / 95.98 |

(Rows spanning several bins are computed directly from `per_shape_outside.csv`.) Within the training mass and height ranges, success still
varies by bin and by individual shape: in both datasets the tallest in-range bin (1.60 to 1.67 m) is the
weakest height bin, and the lowest-success `inside` shapes are not extreme in mass or height (for
example `male_ec3bb7f8`, 42.7 kg, 1.41 m, 19.3%). Beta distance alone does not order the results.

## 6. Trained-body control (150 train-split clips x 128 trained shapes)

| Group | refined | fulldata | Delta [95% CI] | rerun Delta | Beyond noise | d gt err (m) |
|---|---:|---:|---|---:|---|---:|
| all | 99.29 | 99.12 | -0.17 [-0.37, -0.03] | +0.02 | **yes** | +0.0031 |
| bottom8_by_baseline | 98.67 | 98.83 | +0.17 [+0.00, +0.42] | +0.50 | no | +0.0074 |

No trained shape has a CI entirely below -2 pp. Rule 3 reading: on `inside` the candidate delta
(-0.37 pp, mean error +5.0 mm) has the same sign as on trained bodies (-0.17 pp, +3.1 mm) and about twice
the size, which reads as a broad tracking change that is somewhat larger on new bodies. On `outside` the
delta has the opposite sign (+0.42 pp) and is concentrated in the bodies beyond the training mass range
and in tier 3, which reads as a change concentrated on extreme new bodies. Neither reading attributes
the cause to training length or clip coverage.

## 7. Contaminated subset (23 validation/test clips; these clips were in `fulldata`'s training set)

Labelled contaminated for `fulldata`. Not pooled into the headline. 23 clips x 128 shapes = 2,944 pairs.

| Dataset | Group | refined | fulldata | Delta [95% CI] | rerun Delta | Beyond noise |
|---|---|---:|---:|---|---:|---|
| inside | all | 97.08 | 96.50 | -0.58 [-1.02, -0.14] | -0.03 | yes |
| inside | rest | 97.12 | 96.63 | -0.49 [-0.95, -0.04] | -0.04 | yes |
| outside | all | 96.81 | 96.91 | +0.10 [-0.51, +0.71] | +0.10 | no |
| outside | outside_mass_out | 81.74 | 89.57 | +7.83 [+3.91, +11.32] | -0.43 | yes |

With only 23 clips most group CIs are wide; full tables in `compare_<D>_valtest23.txt`. On these clips
`fulldata` does not do better than on the clean 233 clips, even though it trained on them.

## 8. Limitations

- One seed, one candidate checkpoint, no ladder: the noise band is simulator nondeterminism only, so
  "beyond noise" does not rule out ordinary checkpoint-to-checkpoint variation of the same size.
- The simulator flips 0.7 to 1.3% of pairs between identical runs (gate N); per-shape deltas of a few
  pairs are not meaningful.
- The 64 betas per dataset (and the 16 per tier) are not independent samples of shape space for any
  physical property; mass, height and mass-ratio groups are small (1 to 26 shapes) and overlap.
- HUMOS references for extreme bodies were refined (penetration to 0) but not otherwise validated; a
  low success can come from the reference or from the controller, and these runs cannot tell which.
- The control dataset and the unseen datasets come from different refinement runs; only within-dataset
  paired deltas are compared.

## 9. Files (`results/analysis/unseen_v2_2026-10-09/`, also on `r2:proto-data/unseen_shapes_v2/eval/`)

`{inside,outside,ctrl}_{refined_run1,refined_run2,fulldata}.{csv,log}`, `compare_*.{txt,groups.csv,per_shape.csv}`,
`per_shape_{inside,outside}.csv`, `per_shape_summary.txt`, `gateN.txt`, `failure_modes.txt`,
`run_times.txt`, `checkpoint_identity.txt`, `groups/*.txt`, `train233_ids.txt`, `valtest23_ids.txt`,
`hhi_stage2_v1_train_ids.txt`, `clips_256_keyids.json`, `beta_manifest.csv`, `physical_report.csv`,
`smoke_*.{csv,log}`, this report.

## Appendix A. Dataset build (pod handoff), summary

| Gate | inside | outside | Evidence |
|---|---|---|---|
| A preflight | pass | pass | 128 XMLs + `assets.yaml`, 256 raw clips each; `physics_features.pt` shows as modified only because git-lfs is not installed (its sha256 equals the LFS pointer oid) |
| B NPZ export | pass (pre-existing) | pass | 32,768 NPZ + YAML each; `outside` export took 30 s |
| C `.motion` | pass | pass | 32,768 non-empty files, 256 clips x 128 with consistent sizes |
| D packaging | pass | pass | 4 shards x 8,192, 64 clips per shard, 128 per clip, 64 beta keys x 2 genders, finite |
| E frame-0 offset | pass | pass | all 128 bodies loaded in IsaacGym, no load failure or NaN; z-only shift of every motion (inside -0.37 to +0.22 m, outside -0.44 to +0.33 m) |
| F refinement | pass | pass | same order, frame counts and keys as the offset shards; finite; penetration 46 to 50% of frames -> 0 in every shard |
| G upload | pass | pass | `rclone check` 0 differences |

Timing: `.motion` conversion went from about 3 files/s (single process) to about 100 files/s with 32
parallel workers (inside: 31,170 files in 316 s; outside: 32,768 in 337 s). Packaging 6.5 min (inside)
and 30 min (outside, while inside was refining). Offsets 15 to 20 s per shard. Refinement 4 shards in
parallel, 42 to 45 min per dataset. Uploads about 5 min per dataset.

R2 (each prefix was empty before the copy):

| Prefix | Objects | Bytes |
|---|---:|---:|
| `unseen_shapes_v2/inside_offset/` | 4 | 14,480,454,852 |
| `unseen_shapes_v2/inside_refined/` | 9 (4 shards, 4 reports, manifest) | 14,481,774,681 |
| `unseen_shapes_v2/outside_offset/` | 4 | 14,480,520,464 |
| `unseen_shapes_v2/outside_refined/` | 9 | 14,481,831,974 |

Build deviations: 1,598 `inside` `.motion` files existed at the start (the handoff said about 330); all
were kept after a per-clip size check. `outside` stages overlapped with `inside` stages where disk and
CPU allowed. Only `.motion` files were deleted from `npz/` (NPZ kept). The optional
`reference_quality_by_asset.csv` was not produced. tmux was installed with apt for the long jobs.
