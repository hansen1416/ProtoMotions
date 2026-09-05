# Residual Motion Failure and Morphology-Dependent Physics Analysis

**Date:** 2026-09-03  
**Status:** Pre-registered analysis plan; final full-scale results are not yet available.  
**Purpose:** Complete the two remaining evidence-heavy parts of the paper: explain the residual
motion failures and show how physical control changes when the same motion content is executed by
different body shapes.

## Executive conclusion

The two studies should be built from one fixed, paired evaluation corpus rather than treated as
unrelated additions. For every selected clip, replay the final policy on the same set of body
shapes, retain tracking and termination outcomes, and record the simulator's control and contact
signals. The resulting clip-by-shape matrix supports two complementary analyses:

1. **Residual failure analysis:** determine whether failure is primarily associated with bad or
   physically inconsistent references, missing scene/contact assumptions, morphology-specific
   feasibility, actuator/controller limits, learning instability, or the evaluation definition.
2. **Physical analysis:** among rollouts that track comparably well, quantify how body mass,
   height, limb proportions, and morphology conditioning change normalized joint loading,
   mechanical work, contact forces, and balance strategy.

If the final model reaches approximately 90% success, the remaining 10% should be called
**unresolved under the evaluated controller and simulation setting**, not inherently impossible
to imitate. An impossibility claim requires independent feasibility evidence. The stronger and
more useful paper claim is that the residual failures concentrate in identifiable reference,
contact, balance, morphology, and actuation regimes.

## 1. Evidence boundary

### 1.1 What is already observed

- The older 20,946-clip, neutral-shape run plateaued around 82--85%. Across 18 late evaluation
  snapshots, 1,818 clips (8.7%) failed every time. Its worst clips were concentrated in
  single-leg balance, kicking, crawling, object-supported balance/climbing, squatting, sitting,
  backward motion, and kneeling.
- In the older 1,024-clip by 128-shape run, no clip failed for all 128 shapes at its final
  snapshot. Failure was spread across particular clip-shape pairs, with crawling, locomotion
  transitions, backward motion, sitting, and kneeling again overrepresented.
- Those historical evaluations mainly defined failure through early termination or falling. A
  stable but inaccurate imitation could therefore count as successful. They motivate hypotheses,
  but they do not establish the causes or the final refined model's failure rate.
- The selected full-scale model is direct multi-shape training with slot/type temporal attention
  on refined HUMOS references. The four-shape online evaluator was reverted to one sampled shape
  per clip because its retained GPU buffers contributed to an out-of-memory failure during a pool
  rebuild. The paper analysis must therefore be a separate, batched, offline evaluation.

### 1.2 What remains unknown

- The final static success rate and its uncertainty.
- Whether the same clips remain hard after HUMOS refinement and full-scale slot/type training.
- Whether failure is driven by motion identity, body shape, a clip-by-shape interaction, or a
  small number of bad reference frames.
- Whether observed physical differences are just expected mass/size scaling or represent a
  genuine redistribution of control strategy.
- Whether the actor causally uses its beta input.

All final numerical claims must come from the frozen final checkpoint and evaluation protocol.
Historical results should appear only as motivation or comparison.

## 2. Research questions and hypotheses

### Residual failure

**RQ-F1.** Which motion families account for the residual failure mass?  
**RQ-F2.** Are failures clip-wide, morphology-specific, or isolated to particular motion phases?  
**RQ-F3.** Which mechanism best explains each failure: reference defect, environment mismatch,
feasibility limit, controller saturation, optimization, or metric artifact?  
**RQ-F4.** How persistent are failures across late checkpoints and controlled reruns?

Expected hypotheses:

- Dynamic single-support, low-COM, backward, contact-transition, and non-foot-support motions will
  remain harder than ordinary locomotion.
- Some failures will be explained by target/contact inconsistencies rather than insufficient
  network capacity.
- Morphology will matter mainly through clip-by-shape interactions, not beta magnitude alone.
- A subset of apparent successes will fail a fidelity-aware tracking criterion even though they
  do not fall.

### Morphology-dependent physics

**RQ-P1.** For the same semantic motion content, how do joint torques, work, ground reaction
forces, and balance variables change across bodies?  
**RQ-P2.** Do differences remain after physically meaningful normalization by mass and height?  
**RQ-P3.** Are changes coherent with body properties, such as heavier bodies producing greater raw
loading and different limb ratios redistributing effort across joints?  
**RQ-P4.** Does replacing the correct beta with a neutral or mismatched beta degrade tracking or
alter the control strategy?

Expected hypotheses:

- Raw torque and contact force will increase with mass and scale.
- Non-trivial, joint-specific differences will remain after normalization.
- Tracking quality will remain comparable across the successful shape set.
- Wrong-beta inference will measurably perturb at least some tracking or physical-control metrics
  if the policy causally uses morphology conditioning.

## 3. One frozen evaluation design

### 3.1 Freeze before opening the test results

Record and hash:

- final checkpoint and resolved configuration;
- code commit and local changes;
- train/validation/test manifest versions and hashes;
- refined reference manifest and refinement version;
- morphology asset IDs and beta vectors;
- simulator, physics parameters, control frequency, PD gains, torque limits, and terrain;
- deterministic/stochastic action mode and random seeds;
- metric definitions, tracking thresholds, and aggregation rules.

The test manifest must remain unavailable for model selection. Use validation clips to debug the
instrumentation and freeze thresholds. Repeated inspection of test failures must not feed back
into architecture or reward design unless the resulting experiment is explicitly labeled a new
training round.

### 3.2 Three nested evaluation panels

1. **Broad motion census:** every test clip on four fixed, preselected shapes. Choose a median
   body, a short/light body, a tall/heavy body, and a body with a contrasting limb-to-height
   ratio. Fixed shapes make checkpoint and method comparisons reproducible; random shapes do not.
2. **Paired 128-shape benchmark:** approximately 100--200 fixed held-out clips, stratified by
   motion family, evaluated on all 128 training morphologies. This is the primary denominator for
   cross-shape success and failure claims.
3. **Instrumented physics panel:** a preselected 48--60-clip subset of the paired benchmark,
   spanning locomotion, running, turning, jumping, squatting, kneeling, and crawling. Evaluate all
   128 shapes. Start with an 8-clip by 16-shape pilot to validate signals and storage.

All panels should run offline in small clip/shape batches and release tensors between batches.
This avoids the training-time evaluation memory peak while preserving deterministic coverage.

### 3.3 Define success without hiding tracking failure

Report three outcomes separately:

- **Completion:** the rollout finishes without an early termination or fall.
- **Tracking pass:** position, articulation, root orientation, and other frozen fidelity metrics
  remain within predeclared thresholds.
- **Composite success:** completion and tracking pass are both true.

For clip `c` and shape `s`, define `S[c,s]` as composite success and

```text
clip success fraction:   p_clip[c]  = mean_s S[c,s]
shape success fraction:  p_shape[s] = mean_c S[c,s]
overall success:                     mean_c,s S[c,s]
```

Predeclare the thresholds on validation data. Always publish completion and continuous tracking
errors next to composite success so the result is not sensitive to one arbitrary cutoff.

For descriptive analysis, divide clips into:

- **robust:** `p_clip >= 0.95`;
- **morphology-sensitive boundary:** `0.05 < p_clip < 0.95`;
- **pervasive failure:** `p_clip <= 0.05`.

The physics conclusions should use the full preselected panel first, with a sensitivity analysis
conditioned on successful/comparably tracked rollouts. The boundary group is especially valuable
for explaining morphology-dependent failures.

## 4. Part I -- explaining the residual failures

### 4.1 Operational failure taxonomy

| Class | Mechanism | Diagnostic signature | Defensible conclusion |
|---|---|---|---|
| Reference defect | Discontinuity, excessive acceleration, foot sliding, penetration, self-collision, joint-limit violation, or inconsistent contact label | Reference anomaly appears before rollout divergence and is unusually large versus matched successful clips | Target quality is associated with failure; call it causal only if a corrected target rescues tracking |
| Environment/contact mismatch | Motion assumes a chair, beam, wall, hand support, or non-flat terrain absent from the simulator | Failure occurs at the missing interaction phase; non-foot support is required by the reference | Task specification is incomplete for this simulator, not necessarily beyond the policy |
| Morphology-specific feasibility | Retargeted pose or trajectory violates reach, clearance, support, joint, collision, or torque constraints for some shapes | Same clip succeeds on many bodies but fails systematically for particular physical descriptors | The clip has a morphology-dependent feasibility or control boundary |
| Actuation/control limit | Required effort approaches torque limits, PD error grows, or contact impulse cannot be produced | Sustained torque utilization/saturation precedes tracking divergence | Present actuation/control settings are insufficient for that pair; not proof of biological impossibility |
| Learning/representation | Reference is clean and feasible, no saturation is seen, but outcome varies across checkpoints/seeds or improves with sampling | Failure lacks a physical bottleneck and is optimization-sensitive | Current policy/training did not reliably learn the behavior |
| Evaluation artifact | Fall-only success, scale-confounded position error, bad termination threshold, or invalid contact interpretation | Classification changes under an audited metric while rollout behavior is unchanged | Report the metric limitation and corrected result |

Each failed clip-shape pair may receive a primary and secondary label. Preserve an `unknown` class;
forcing every example into a mechanism would overstate certainty.

### 4.2 Reference-side measurements

Compute these directly from the refined HUMOS target, before policy rollout:

- local-joint angular velocity, acceleration, and jerk;
- root linear/angular acceleration and discontinuities;
- foot velocity during semantic contact and contact-label consistency;
- ground penetration and clearance;
- joint-limit proximity and rapid limit crossings;
- self-collision or implausibly close body geometry where reliable;
- required support type: feet, knees, hands, seated/object support, or flight;
- target COM relative to the available support region;
- duration and frequency of abrupt contact transitions.

Refinement removing ground penetration does not prove dynamic feasibility. It may preserve a
kinematic trajectory that still requires unavailable contact or excessive effort.

### 4.3 Rollout-side measurements

For every failure, retain:

- termination reason, frame, normalized motion phase, and last valid contact state;
- body position, local rotation, root position/orientation, and velocity errors over time;
- actual applied DOF torque, torque utilization, PD target/error, and action;
- contact bodies, contact forces, foot slip, and unexpected collisions;
- root/COM state and support-region descriptor;
- first metric to cross its threshold.

Align failures by the first divergence frame and inspect a window before and after it. The ordering
of events is more informative than a whole-episode mean: torque saturation before pose divergence
supports a control-limit explanation, whereas a target discontinuity before saturation supports a
reference explanation.

### 4.4 Persistence is evidence, not a cause

Evaluate at least the selected final checkpoint and several nearby late checkpoints. If practical,
repeat a small diagnostic subset with independently trained seeds. Deterministic replay of the same
checkpoint is a reproducibility check, not an independent sample.

Define clip-level persistence as the fraction of evaluated checkpoints/seeds in which its failure
rate exceeds a frozen threshold. Persistent failure indicates a stable hard case, but it does not
by itself distinguish bad data from insufficient control.

### 4.5 Manual structured audit

Review a stratified sample of failures plus category- and shape-matched successes. Use a fixed form:

- motion family and assumed scene interaction;
- first visible divergence phase;
- necessary supporting body/contact;
- visible reference defect;
- plausible primary/secondary taxonomy label;
- confidence: high, medium, or low.

Where possible, have a second reviewer label a subset and report agreement. Videos should display
reference and policy side by side, include contact markers, and print shape ID, phase, tracking
error, and torque saturation.

### 4.6 Statistical analysis

Start with transparent descriptive results:

- failure rate by motion family and support mode;
- distribution of `p_clip` and `p_shape`;
- clip-by-shape success heatmap ordered by motion family and physical shape descriptors;
- failure phase histogram;
- taxonomy proportions with an explicit unknown fraction.

Then model pairwise success with a mixed-effects logistic model:

```text
logit P(S[c,s] = 1) =
    reference-quality features[c,s]
  + physical-shape features[s]
  + selected interactions
  + clip effect
  + shape effect
```

Report effect sizes or odds ratios with confidence intervals, not only p-values. Bootstrap by clip
for headline uncertainty; use false-discovery-rate correction for large joint- or feature-wise
families. A clip effect captures unmeasured motion difficulty, while selected interactions test
whether, for example, single-leg motions are especially difficult for particular limb ratios.

### 4.7 Counterfactual diagnostics and evidence levels

Use cheap, targeted counterfactuals on representative failures:

- corrected versus uncorrected reference where both exist;
- standard versus temporarily increased effort limits or adjusted PD gains;
- correct environment versus a required support object/contact, if implementable;
- correct beta versus neutral or mismatched beta;
- final versus nearby checkpoints or independent training seeds.

These tests are diagnostic and must not be folded into the main benchmark score. Apply the
following language rule:

- **Level 1 -- association:** a feature is enriched among failures;
- **Level 2 -- mechanism-consistent:** it precedes divergence and matches the expected failure
  signature;
- **Level 3 -- counterfactual support:** changing that factor rescues or reliably changes the
  outcome.

Only Level 3 supports a causal statement. Even a stronger PD controller or kinematic playback is
not an oracle proof of physical feasibility; it only narrows the explanation.

### 4.8 Failure-analysis paper outputs

**Main figure:** clip-by-shape composite-success heatmap, ordered by motion family and body
descriptors.  
**Second figure:** failure taxonomy with observed fractions and unknown cases.  
**Case figure:** time-aligned reference anomaly, contact, tracking error, and torque utilization for
three representative mechanisms.  
**Main table:** success, completion, fidelity, persistence, and predictor effect sizes by motion
family.  
**Supplement:** full clip list, labels, per-shape rates, and representative videos.

## 5. Part II -- physical analysis across body shapes

### 5.1 Terminology and scope

The primary control quantity is **actuator joint torque**, not generic joint force. Joint reaction
forces may be reported only if Isaac Gym exposes and validates them reliably. Ground contact forces
are separate measurements.

The 128 HUMOS variants share semantic/canonical motion content, but their kinematics are not
necessarily identical: local rotations, root trajectories, contacts, and timing can change with
shape. Therefore, observed variation contains four components:

1. the physical asset: mass, inertia, geometry, limits, and collision shape;
2. the shape-specific HUMOS reference;
3. the actor's morphology input;
4. controller settings such as PD gains and effort limits.

The analysis must record reference-kinematic covariates and use the wrong-beta experiment to
isolate component 3. It should not attribute all cross-shape variation to beta conditioning.

### 5.2 Physical shape descriptors

Compute descriptors from the actual simulator asset:

- total mass `M` and standing height `H`;
- leg length / height and arm length / height;
- shoulder width / height and hip width / height;
- torso-to-leg mass ratio;
- segment masses and inertias where reliable;
- beta vector and beta norm as secondary descriptors.

Prefer interpretable physical descriptors over individual beta coefficients in headline figures.
Betas remain useful for reproducibility and exploratory analysis.

### 5.3 Required per-step instrumentation

Record at simulator/control rate:

- actual applied DOF torque and effort limit;
- joint position and velocity;
- action, PD target, and PD error;
- actual and reference body/root states;
- contact body IDs and contact forces;
- root and COM position/velocity;
- tracking errors, success, termination reason, clip ID, shape ID, and phase.

Use `tools/check_replay_torque_saturation.py` as the instrumentation foundation, but first verify
that the reported tensor is the applied torque after all clipping/scaling, its units and DOF order
are correct, and multi-GPU/rank aggregation does not duplicate samples.

### 5.4 Core metrics

#### Tracking guardrails

- composite success and completion;
- DOF/local-rotation error;
- root orientation error;
- root-position error normalized by height or leg length;
- mean and peak body-position error;
- motion completion fraction and fall rate.

No physical comparison should be interpreted as adaptive control if tracking quality is materially
different between the compared shapes.

#### Joint loading

For each DOF and anatomically grouped joint, compute:

- torque RMS, 95th percentile, and peak;
- torque utilization `|tau| / torque_limit`;
- saturation-frame fraction;
- dimensionless torque `tau / (M g H)`;
- phase-normalized torque curves.

Aggregate both per-axis and as a joint norm. Group hips, knees, ankles, spine, shoulders, elbows,
and wrists for readable main-text figures while retaining DOF-level values in the supplement.

#### Mechanical power and work

Use `P = tau * qdot` and report:

- positive, negative, and absolute mechanical work;
- mean and peak positive/negative power;
- work normalized by `M g H`;
- power normalized by `M g H sqrt(g/H)`.

Call these **mechanical work** and **control effort**, not metabolic energy. Isaac Gym actuator
torque does not model human muscle efficiency or co-contraction.

#### Contact and stability

- vertical ground-reaction force normalized by `M g`;
- horizontal shear, contact impulse, and left/right load share;
- stance duration, duty factor, contact timing, foot slip, and symmetry;
- root/COM acceleration normalized by `g`;
- COM-to-support-region margin as a descriptive indicator.

Support-margin metrics need careful interpretation during running, jumping, hand/knee contact, and
flight. They are not a universal dynamic-stability proof.

### 5.5 Main paired analyses

#### A. Expected physical scaling

Estimate how raw torque, contact force, and work vary with mass and height. This validates the
instrumentation and establishes the expected baseline: larger/heavier bodies should generally
require greater raw effort.

#### B. Normalized strategy redistribution

Test whether dimensionless torque, joint-group work share, load timing, or contact timing changes
with limb and mass-distribution descriptors after accounting for clip identity. This is the main
scientific result. A finding that only raw values scale with mass is correct but comparatively
weak; coherent normalized redistribution supports shape-adaptive control.

For a metric `y` on clip `c` and shape `s`, use a paired model such as:

```text
y[c,s] = clip_effect[c]
       + beta_mass * M[s]
       + beta_height * H[s]
       + beta_leg * leg_ratio[s]
       + beta_distribution * torso_leg_mass_ratio[s]
       + error[c,s]
```

Use nonlinear terms only when plots or validation data justify them. Report standardized effects,
clip-bootstrap confidence intervals, and cross-validated fit. Correct multiple comparisons across
joint groups.

#### C. Phase-dependent adaptation

Time-normalize each semantic phase or gait cycle and compare curves rather than episode averages
alone. Peak ankle/knee/hip torque, load transfer, and contact timing may change even when total work
does not.

#### D. Morphology-dependent failure boundary

On boundary clips, compare successful and failed shapes within the same clip. Determine whether a
physical descriptor, target property, or torque/contact limit predicts the transition. Keep this
analysis separate from the common-success physics result because post-failure forces are not a fair
measure of successful strategy.

### 5.6 Wrong-beta causal experiment

For a targeted set of approximately 20--30 clips by 32 shapes, freeze the physical asset,
shape-specific reference, initial state, checkpoint, and deterministic action mode. Change only
the actor's morphology input:

1. correct beta;
2. neutral beta;
3. distant shuffled beta.

Compare tracking, torque distribution, saturation, work, and contact timing using paired
differences. The critic is irrelevant at inference. This experiment tests whether the actor uses
the beta channel causally; it does not isolate all body-shape effects.

Interpretation:

- Correct beta outperforming both controls supports causal use of morphology conditioning.
- Similar tracking but different normalized effort suggests beta changes strategy without changing
  task success.
- No measurable difference means the policy may infer morphology from state/history, ignore beta,
  or operate in a regime where morphology does not affect the chosen action. In that case, claim
  morphology-dependent **system behavior**, not causal beta-conditioned adaptation.

### 5.7 Physical-analysis paper outputs

**Main figure 1:** joint-group normalized-load heatmap across representative shape descriptors.  
**Main figure 2:** paired trends of normalized torque/work versus mass, height, and limb ratio with
clip-level confidence intervals.  
**Main figure 3:** phase-normalized torque and ground-reaction-force curves for selected motions and
shapes.  
**Causal figure:** correct-, neutral-, and shuffled-beta paired differences.  
**Main table:** tracking guardrails, dimensionless load, work, contact, and saturation statistics.  
**Video:** synchronized executions of the same motion content by contrasting bodies with contact
and torque overlays.

## 6. Data products and reproducibility

Create two linked tables plus optional frame arrays.

### Rollout table: one row per clip-shape-condition replay

```text
run_id, checkpoint_hash, split_hash, clip_id, shape_id, beta,
shape_descriptors, reference_version, condition, seed,
completion, composite_success, termination_reason, failure_frame,
tracking_aggregates, torque_aggregates, work_aggregates,
contact_aggregates, stability_aggregates, taxonomy_label, confidence
```

### Frame table/arrays: one row or array index per step

```text
clip_id, shape_id, condition, frame, phase, q, qdot, tau,
effort_limit, action, pd_target, reference_state, actual_state,
contact_body, contact_force, root_state, com_state, tracking_errors
```

Store a data dictionary with units and aggregation definitions. Save raw arrays in chunked files
and the compact rollout table as CSV/Parquet for statistical analysis. Every figure should be
rebuildable from a checked-in analysis script or notebook and the frozen aggregate table.

## 7. Validation gates

Before production evaluation, require the 8-clip by 16-shape pilot to pass:

1. one recorded torque value is independently reproduced from the controller path;
2. DOF order maps correctly to joint names and axes;
3. torque-limit utilization never exceeds the expected clipping rule;
4. power sign and units pass a simple known-motion sanity check;
5. contact forces and body IDs match visible contacts;
6. phase normalization preserves contact events;
7. deterministic reruns agree within simulator tolerance;
8. shape descriptors match asset mass/geometry, not only SMPL betas;
9. success and tracking values reproduce the evaluator on the same replay;
10. batched and unbatched evaluation produce equivalent aggregates.

Do not launch the full instrumented panel until these checks pass.

## 8. Cost-aware execution order

1. Finish training and select the checkpoint using validation data only.
2. Freeze definitions, hashes, four representative shapes, and stratified clip lists.
3. Run the 8-by-16 instrumentation pilot and validation gates.
4. Run the broad four-shape test census in offline batches.
5. Run the 100--200-clip by 128-shape paired benchmark.
6. Generate the failure matrix, boundary/pervasive groups, and structured audit sample.
7. Run the 48--60-clip instrumented physics panel.
8. Run targeted failure counterfactuals and the wrong-beta experiment.
9. Fit the predeclared models, bootstrap confidence intervals, and produce figures/tables.
10. Lock the result tables before writing final causal or limitation language.

This sequence spends little on instrumentation bugs, obtains the main success/failure result before
expensive causal diagnostics, and reuses the same rollouts for multiple paper questions.

## 9. Paper integration

### Experiments: residual failure analysis

Report the static composite success rate, its clip/shape decomposition, motion-family
concentration, persistence, and mechanism taxonomy. Historical neutral/multi-shape findings can
motivate the taxonomy, but the main numbers must come from the refined full-scale checkpoint.

Safe result template:

> The final policy successfully completed and tracked **[X]%** of fixed held-out clip-shape pairs.
> Residual failures were concentrated in **[families]** and decomposed into **[taxonomy results]**.
> These cases are not claimed to be inherently non-imitable; they identify limitations of the
> reference, task specification, actuation, and learned controller under the evaluated setting.

### Evaluation: morphology-dependent physical control

First establish comparable tracking. Then report raw physical scaling, normalized redistribution,
phase/contact adaptation, and the wrong-beta result.

Safe result template:

> Across successful paired executions, raw joint loading scaled with body size as expected, while
> **[normalized joint/contact metrics]** changed systematically with **[physical descriptors]**
> after controlling for clip identity. **[Wrong-beta result]** provides **[support/no support]** for
> causal use of the explicit morphology input.

### Discussion and limitations

State clearly that:

- HUMOS references change with shape, so "same motion" means shared semantic/canonical content,
  not identical trajectories;
- simulation torque and mechanical work are not biological muscle force or metabolic energy;
- analyses on the 128 trained bodies demonstrate multi-shape control, not unseen-shape
  generalization;
- causal attribution is limited when only observational associations are available;
- object-dependent motions cannot be judged fairly in an environment without the required object;
- physical comparisons are conditioned on adequate tracking.

### Corrections needed in the current draft

Before inserting results, update several stale statements in `paper/`:

- The selected full-scale architecture is slot/type attention, not the flat-MLP conclusion
  currently stated in the architecture subsection.
- The main pipeline is AMASS motion content to HUMOS shape-specific references to refinement to
  GlobalClipPool to one policy; the abandoned mixed-source and two-stage transfer paths are
  negative experiments.
- HUMOS supplies shape-specific rotations. Describe DOF error as scale-comparable articulation
  error; do not claim every shape has an identical joint target or that it is exactly
  shape-invariant.
- Reproduce and archive the provenance of any quoted overlap or shape-correlation number before
  retaining it as a headline conclusion.

## 10. Claim-strength decision rules

| Observed result | Claim allowed |
|---|---|
| Failure categories differ descriptively | Residual failures are concentrated in particular regimes |
| Diagnostic signal precedes failure | Result is consistent with a proposed mechanism |
| Target/controller/environment intervention rescues failure | Counterfactual evidence supports that mechanism |
| Raw torque scales with mass but normalized metrics do not change | Expected physical scaling only |
| Normalized joint/contact patterns change with physical descriptors at matched tracking | Morphology-dependent control redistribution |
| Correct beta beats neutral/shuffled beta with the physical body/reference fixed | Actor causally uses explicit morphology input |
| No wrong-beta effect | Do not claim causal beta use; retain system-level multi-shape result |
| Failure persists for one model | Not reliably tracked by this controller; not "impossible" |

## 11. Minimum publishable package

The two paper chunks are complete when the project has:

- one frozen final checkpoint and reproducible static evaluation protocol;
- composite, completion, and continuous tracking results on the paired benchmark;
- a clip-by-shape failure matrix and auditable taxonomy with unknown cases;
- at least one temporal/mechanistic failure case study;
- validated torque/contact instrumentation;
- paired normalized torque, work, and contact analysis with tracking guardrails;
- wrong-beta ablation or explicitly narrowed causal language;
- confidence intervals and clip-aware statistical treatment;
- figure source tables, scripts/notebooks, and synchronized qualitative videos;
- updated paper language that matches the actual slot/type-refined pipeline.

## Source notes

- [Neutral-run persistent failures](README.failed-motions-20946-neutral.md)
- [Multi-shape 1,024-clip failure analysis](README.failed-motions.md)
- [Detailed physical-analysis plan](README.physics-analysis-plan.md)
- [Paper completion plan](README.paper-completion-plan.md)
- [Current paper outline](README.paper-outline.md)
- [Chronological experiment record](README.note.md)

