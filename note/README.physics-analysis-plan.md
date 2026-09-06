# Shape-Conditioned Physical Analysis Plan

**Date:** 2026-09-01 (updated 2026-09-05)  
**Status:** Planned; run after selecting the final full-scale checkpoint.

## Update 2026-09-05 -- two evaluator bugs fixed, existing split infrastructure identified

Two bugs in the online training evaluator were found and fixed this week, both directly relevant
to this plan's instrumentation:

1. **Morphology/env-asset mismatch in eval batching.** The training-time `MimicEvaluator` assigned
   eval batches to sequential `env_ids = arange(0, N)` with no check that an env's fixed physical
   asset matched the shape the selected reference motion was generated for. Fixed by
   `MimicEvaluator._build_morphology_matched_eval_batches`, which groups by asset and asserts the
   match at runtime. On the running `hhi_wide_stage2_discover_attention_slot_type_refined` job,
   `eval_holdout/success_rate` jumped from ~78-82% to ~96-98% at the exact epoch the fix landed --
   confirming the mismatch was real and material, not a rounding error. **The offline paired
   physics evaluator this plan builds (Sec. 6) must implement the same explicit shape/env-asset
   assertion, not assume batch-order alignment** -- see the new item in Sec. 11.
2. **`GlobalClipPool` priority-selection numerical bug**, unrelated to physics measurement but worth
   knowing when interpreting which checkpoints saw which clips: the old softmax-over-rank sampler
   underflowed past ~rank 100 in float32, so it was already collapsing to top-K in practice. Now
   explicit deterministic top-K. Does not change this plan's design.

Separately, the versioned split infrastructure this plan needs already exists and is already
active on the current run -- see Sec. 3 below for the exact manifests/hashes. This replaces the
open-ended "select ~48-60 clips" language with a concrete, already-frozen source set.

## 1. Purpose

The goal is to determine whether one morphology-aware policy produces physically meaningful, shape-dependent control strategies while tracking the same motion content across different bodies.

The analysis should answer three questions:

1. How do mass, height, limb proportions, and shape extremity affect joint loading, mechanical work, contact forces, and stability?
2. Does the policy preserve tracking quality by adapting its torque and contact strategy to each body?
3. Does the actor causally use the beta observation, or do the apparent differences mainly come from the physical asset, reference motion, and mass-scaled PD controller?

The interesting result is not merely that a heavier body requires more raw torque. The stronger result is a systematic change in **normalized joint-effort distribution, contact behavior, or stability** while tracking quality remains comparable.

## 2. Hypotheses

- **H1 — physical scaling:** raw torque, ground-reaction force, and mechanical work increase with body mass and scale.
- **H2 — nontrivial adaptation:** after normalizing for mass and height, joint-load distribution and contact strategy still vary systematically with limb proportions.
- **H3 — preserved task quality:** the policy maintains similar success and tracking error over most of the shape range by changing its physical strategy.
- **H4 — causal beta use:** replacing the correct beta observation with a neutral or mismatched beta degrades tracking or changes the physical control profile.

## 3. Freeze the evaluation protocol

This project already has a versioned, hash-verified clip split -- `hhi_stage2_v1`
(`GlobalClipPoolConfig.split_metadata_name`), active by default on the current stage-2 experiment.
Per the resolved config actually logged for the running `hhi_wide_stage2_discover_attention_slot_type_refined`
job:

| Role | Manifest | Clip count | SHA-256 |
|---|---|---|---|
| Train | `splits/hhi_stage2_v1/train_manifest.jsonl` | 18,855 | `bddfd28b...` |
| Validation | `splits/hhi_stage2_v1/validation_manifest.jsonl` | 1,048 | `78c7e588...` |
| Test | *(never downloaded -- provenance only)* | 1,048 | `41b77af3...` |

The training pipeline records the test manifest's hash and count (`GlobalClipPoolConfig.test_manifest_sha256`,
`test_clip_count`) but never fetches it -- "no test manifest field in the training API" is already
enforced in code (`GlobalClipPool`). **Use this existing split rather than selecting a fresh clip
set:** the validation split is what `MimicEvaluator._evaluate_holdout` already scores during
training (`eval_holdout/*` in wandb) and is the right place to pilot/debug this plan's
instrumentation; the untouched test split is the only legitimate source for headline paper numbers.

Note that `eval_holdout/*` currently samples **one rotating shape per clip** (deterministic
per-clip hash panel, not the full clip x 128-shape matrix) -- it tells you aggregate trend, not
per-clip/per-shape structure. It cannot substitute for this plan's paired 128-shape panel.

After full-scale training:

1. Select the checkpoint using the existing fixed-holdout criterion (`eval_holdout/*` on the
   validation split above), and require the trend to be stable over several consecutive eval
   cycles post-fix, not a single reading -- `eval/*` (non-holdout) is known to oscillate with
   `GlobalClipPool` rebuild composition and is not a selection criterion.
2. Freeze the checkpoint, resolved configuration (including `split_version`/manifest hashes above),
   clip list, shape list, simulator settings, and random seeds.
3. Use deterministic actor actions for the main analysis.
4. Run in Isaac Gym, matching the training simulator.
5. Record the checkpoint identity and evaluation manifest with every output.
6. Do not download or open the test manifest until every instrumentation validation gate (Sec. 7 of
   the companion `README.failure-and-physics-analysis-report.md`) has passed on the validation
   split's pilot subset.

The motion and shape samples must be fixed before inspecting the physics results.

## 4. Motion and shape sampling

### 4.1 Main paired dataset

Select approximately **48–60 clips from the frozen, untouched `hhi_stage2_v1` test split** (1,048
clips total; see Sec. 3), stratified across:

- Walking and steady locomotion
- Running and dynamic locomotion
- Turning and directional changes
- Jumping and landing
- Squatting, kneeling, and rising
- Contact-rich or unusual motions such as crawling

**Gap to close first:** the clip manifest only carries `clip_id`/`file` -- there is no existing
motion-family/action-category label per clip anywhere in this repo (checked: no such field in
`GlobalClipPool`'s manifest schema, no category lookup table under `data/`). Stratification
requires building a `clip_id -> motion_family` mapping (e.g. from HumanML3D/AMASS text
annotations, or a manual/LLM labeling pass over the test split's 1,048 clip IDs) before selection.
Build and freeze this mapping from clip identity alone -- never from rollout results -- to keep
the test split's "no model-selection feedback" guarantee intact.

Evaluate every selected clip across all **128 training body shapes**. This gives approximately 6,000–8,000 paired rollouts and preserves the same-motion/across-shape structure.

Analyze two predefined groups:

- **Common-success group:** clips tracked successfully by at least 90–95% of shapes. Use this group for clean comparisons of physical strategy.
- **Boundary group:** clips whose success varies substantially across shapes. Use this group to explain morphology-related failure.

Failures remain part of the dataset. Low torque from a collapsed or poorly tracking episode must never be interpreted as efficient control.

### 4.2 Instrumentation pilot

Before production evaluation, test the complete pipeline on **8 clips drawn from the validation
split** (`splits/hhi_stage2_v1/validation_manifest.jsonl`, 1,048 clips) -- not the test split, so
that debugging the instrumentation never touches the held-out set:

- 8 clips
- 16 representative shapes
- Neutral, light/heavy, short/tall, and proportionally extreme bodies

The pilot should validate units, DOF ordering, contact-force signs, normalization, output size, and deterministic replay.

### 4.3 Optional robustness subset

For a smaller subset of clips and shapes, repeat rollouts under standardized initial-state perturbations. Keep this separate from the deterministic primary analysis.

## 5. Physical shape descriptors

Extract interpretable quantities from each SMPL-MOR asset:

- Total mass, \(M\)
- Standing height, \(H\)
- Leg length divided by height
- Arm length divided by height
- Shoulder width and hip width
- Torso-to-leg mass ratio
- Beta L2 norm as a secondary shape-extremity descriptor

Use these physical quantities as the primary explanatory variables. Beta L2 alone is not physically interpretable enough for the main analysis.

## 6. Per-step instrumentation

For every rollout, record:

- Clip ID, shape/asset ID, gender, beta vector, and motion phase
- Applied DOF torque and shape-specific effort limits
- Joint positions and velocities
- Actor action and PD target, if available
- Actual and reference rigid-body position, rotation, velocity, and angular velocity
- Per-body contact forces and binary contact flags
- Root and center-of-mass position and velocity
- Tracking errors, success/failure state, termination reason, and episode length

`tools/check_replay_torque_saturation.py` already provides the correct foundation for replaying a checkpoint in Isaac Gym, retrieving real PhysX-applied torque, and computing shape-scaled effort limits. Extend this approach into a general paired physics evaluator rather than relying on offline inverse dynamics.

**Required, not optional:** whatever code assigns each (clip, shape) rollout to a simulator env
must explicitly group by the env's fixed physical asset and assert the match at runtime, following
`MimicEvaluator._build_morphology_matched_eval_batches`'s pattern in
`protomotions/agents/evaluators/mimic_evaluator.py`. The training-time evaluator shipped for weeks
with exactly this class of bug (sequential env-id assignment silently ignoring per-env morphology),
and it produced plausible-looking but wrong numbers the whole time -- there was no crash, just
quietly wrong torque/tracking data. A paired physics evaluator doing 6,000-8,000 rollouts across
128 distinct physical assets is at least as exposed to this failure mode as the training evaluator
was.

Offline inverse dynamics should not be the principal method because earlier reference self-collisions produced physically meaningless torque spikes.

## 7. Core metrics

### 7.1 Tracking guardrails

- Success rate
- Mean and peak rigid-body position error
- Rotation error
- Episode completion fraction
- Fall rate

Every effort comparison must be accompanied by tracking quality.

### 7.2 Joint loading

For every DOF and grouped joint:

- RMS torque
- 95th-percentile and peak absolute torque
- Torque utilization: \(|\tau_d| / \tau_{\mathrm{limit},d}\)
- Saturated-frame fraction
- Dimensionless joint moment: \(\|\tau_j\| / (M g H)\)

For 3-DOF joints, retain axis-level values and also report the norm across the joint axes. Aggregate into hip, knee, ankle, torso, shoulder, and elbow groups.

### 7.3 Mechanical power and work

- Instantaneous mechanical power: \(P_d(t)=\tau_d(t)\dot q_d(t)\)
- Positive work: \(\int \max(P_d(t),0)\,dt\)
- Negative work: \(\int \min(P_d(t),0)\,dt\)
- Absolute work: \(\int |P_d(t)|\,dt\)
- Mean and peak power per unit mass
- Positive and absolute work per unit mass

Use the terms **mechanical work**, **actuator power**, or **control effort**. Do not describe these measurements as metabolic energy expenditure.

### 7.4 Contact behavior

For each foot:

- Peak vertical ground-reaction force normalized by body weight, \(F_z/(Mg)\)
- Horizontal/shear force
- Contact impulse
- Stance duration and duty factor
- Contact transition timing
- Contact-timing error relative to the reference
- Left/right force and timing symmetry where appropriate

Normalize time to 0–100% motion phase for contact-curve comparisons.

### 7.5 Stability

- Center-of-mass trajectory and velocity
- Stance-phase COM-to-support margin
- Fraction of relevant stance frames with negative support margin
- Root orientation and pelvis-height stability

Treat support-polygon measures as descriptive and restrict them to appropriate stance phases. COM projection outside the support polygon is not automatically instability during running, jumping, or flight. A dynamic extrapolated-COM/capture-point measure can be added later if required.

## 8. Main analyses

### 8.1 Morphology-to-physics relationship

For each metric, use the paired structure and control for motion content. A suitable model is:

\[
y_{c,s}=\alpha_c+\beta_1 M_s+\beta_2 H_s+\beta_3 R_{\mathrm{leg},s}+\epsilon_{c,s},
\]

where \(c\) is the clip, \(s\) is the shape, and \(\alpha_c\) controls for clip difficulty and dynamics.

Report standardized effect sizes and bootstrap confidence intervals. Use physical descriptors selected in advance and avoid overfitting correlated beta dimensions.

### 8.2 Joint-effort redistribution

Calculate the fraction of total lower-body effort assigned to the hip, knee, and ankle. Repeat for upper-body and torso groups where relevant.

This identifies nontrivial strategy changes even when total normalized effort is similar.

### 8.3 Contact adaptation

For locomotion and landing motions, compare force curves, impulse, stance duration, contact timing, and symmetry across body shapes. Analyze flight and stance phases separately.

### 8.4 Morphology-dependent failure

For the boundary group, relate success and tracking error to:

- Mass, height, and limb proportions
- Torque utilization and saturation
- Mechanical work
- Contact timing and peak forces
- Stability descriptors
- Motion category

This distinguishes actuator/load limitations from contact or tracking failures.

### 8.5 Statistical treatment

- Keep clip-shape observations paired.
- Use clip fixed effects or an equivalent mixed-effects formulation.
- Bootstrap at the clip level; use a crossed clip/shape bootstrap if practical.
- Report effect sizes and confidence intervals, not only p-values.
- Correct joint-wise multiple comparisons with false-discovery-rate control.
- Analyze success first and physical metrics conditional on acceptable tracking second.

## 9. Wrong-beta causal experiment

Keep the following fixed:

- Physical body asset
- Shape-specific reference motion
- Initial state
- Checkpoint and actor weights

Change only the morphology observation supplied to the actor:

1. Correct beta
2. Neutral beta
3. Beta shuffled from a morphologically distant body

Run this experiment on approximately 20–30 clips and a stratified 32-shape subset. Compare success, tracking error, torque utilization, mechanical work, contact timing, and joint-load distribution.

This is the cleanest test of whether the learned actor uses morphology. The critic is irrelevant during deterministic inference.

If incorrect beta inputs have little effect, do not claim that beta conditioning causes the observed strategies; those differences may instead come from body physics, reference kinematics, and mass-scaled control gains.

## 10. Separate body effects from reference effects

HUMOS produces slightly different kinematics for different body shapes. Physical differences can therefore arise from both the body and the requested reference.

For every shape-specific reference, record:

- Joint range of motion
- Joint angular speed
- Root speed
- Contact schedule
- Reference COM trajectory, where available

Use these as covariates or show that the morphology effects remain after accounting for reference-motion differences. The wrong-beta experiment provides the complementary causal test because the body and reference remain fixed.

## 11. Validation checks

Before trusting the production output:

1. Confirm applied torque matches `tools/check_replay_torque_saturation.py` on identical rollouts.
2. Verify simulator-to-common DOF ordering and joint grouping.
2b. For every rollout, assert the env's physical asset ID equals the selected shape's asset ID
   before recording any metric -- a direct regression test against the exact bug class fixed in
   `MimicEvaluator` on 2026-09-05 (see Update note at top of this file).
3. Confirm torque stays within shape-specific effort limits up to numerical tolerance.
4. Check that a static or slow stance produces vertical forces close to body weight.
5. Independently recompute \(\tau\dot q\) for several trajectories.
6. Confirm recorded success agrees with the standard evaluator.
7. Repeat a deterministic batch and confirm identical results.
8. Check all traces for NaNs, truncated episodes, and invalid contacts.
9. Inspect synchronized videos with torque/contact overlays for representative cases.

## 12. Paper outputs

### Main figures

1. **Joint-load heatmap:** joint groups versus body-shape or physical-feature quantiles.
2. **Paired morphology plot:** mass, height, or leg ratio versus normalized torque/work, controlling for clip.
3. **Wrong-beta ablation:** correct, neutral, and shuffled beta across tracking and physics metrics.

### Main table

Report success, tracking error, normalized work, peak normalized ground-reaction force, torque utilization, and saturation rate.

### Qualitative result

Render the same motion across approximately five representative shapes. Color joints by normalized torque and overlay foot-force vectors or contact state.

## 13. Interpretation rules

The strong claim is supported if:

- Tracking remains reasonably consistent across shapes.
- Normalized joint-load distribution changes systematically with morphology.
- Contact behavior adapts coherently to body proportions.
- Incorrect beta observations measurably degrade control.

If only raw torque scales with mass, the result is physically correct but largely expected and should be supplementary. If wrong beta has little effect, describe the measurements as morphology-dependent system behavior rather than evidence of learned beta-conditioned physical strategy.

## 14. Recommended execution order

0. Confirm `eval_holdout/*` (validation split) has stabilized over several consecutive post-fix
   eval cycles, and build the `clip_id -> motion_family` mapping needed for Sec. 4.1 stratification
   -- both are prerequisites, not part of the frozen-checkpoint step below.
1. Freeze the final checkpoint and evaluation manifest (train/validation/test hashes from Sec. 3).
2. Build and validate the 8-clip/16-shape pilot on the **validation** split.
3. Run the deterministic 48–60 clip × 128 shape paired evaluation on the **test** split.
4. Produce the common-success and boundary analyses.
5. Run the smaller wrong-beta causal evaluation.
6. Add robustness perturbations only if the primary findings warrant them.
7. Generate the final statistics, figures, table, and qualitative video.

This ordering preserves the project's compute-sensitive methodology: validate the measurement design cheaply, then spend GPU time only on the fixed production analysis.
