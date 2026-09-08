# Paper Structure and Keypoints

**Date:** 2026-09-08

**Status:** Current drafting plan. Physical analysis remains a placeholder; pending experiments
and dataset checks are not presented as completed findings.

This replaces the framing in the historical [August outline](README.paper-outline.md).
It is a planning document, not a rewrite of the LaTeX manuscript or a new dataset release.

## 1. Central story

A HumanML3D-derived, multi-morphology motion corpus links source text descriptions to generated
shape-specific references and matching simulation assets. A single morphology-conditioned
physics-based policy learns to track the refined references. Controlled small-scale experiments
and held-out full-scale evaluation establish what the data and architecture contribute.

Text is dataset metadata, not an input to the current controller. The final training approach is
direct multi-shape training with slot/type attention, not neutral pretraining followed by transfer
and not the abandoned mixed canonical-AMASS/HUMOS reward experiment.

## 2. Proposed paper structure

### Abstract

Write last: problem, dataset/system, supported findings, and the scope of evaluation.

### 1. Introduction

- Motivation: diverse motions across bodies with different geometry and dynamics.
- Research scope and the dataset, system, and empirical contributions below.
- Brief results summary, using only corrected and comparable evaluations.

### 2. Related Work

- 2.1 Physics-based motion imitation.
- 2.2 Morphology-conditioned control and motion retargeting.
- 2.3 Text-motion datasets, body-shape-conditioned generation, and reference refinement.

Attribute HumanML3D, AMASS, HUMOS, SMPL/asset-generation tools, and ProtoMotions explicitly.
Do not claim that morphology-conditioned control or text-motion pairing is itself new.

### 3. Multi-Morphology Text-Associated Motion Dataset

- 3.1 Source motion and text provenance: AMASS preprocessing, HumanML3D clip identities,
  filtering, and inherited captions.
- 3.2 Shape-specific motion and asset construction: 64 ten-dimensional beta vectors sampled
  uniformly in `[-3, 3]` with seed 46, crossed with two body-model sex variants; AMASS-derived
  motion through HUMOS, and SMPL bodies through SMPLSim using the same morphology identities.
- 3.3 Conversion and reference refinement: frame-rate/coordinate conventions, contact-aware
  SO(3) smoothing, stance-root correction, collision clearance, and consistent FK/velocities.
- 3.4 Current representation and access: per-clip files, clip/asset IDs, separate text lookup,
  scale, and deterministic splits.
- 3.5 Current limitations: caption alignment is not fully audited; generated references are not
  individually human-annotated or all dynamically validated. Keep further development optional.

The pipeline figure must distinguish the source-text metadata association from the reference
motion input to the policy; it must not suggest that the controller consumes language.

### 4. Morphology-Conditioned Motion Imitation

- 4.1 Task, matching simulated bodies, PD actuation, and simulation/control timing.
- 4.2 Observation design: state variables, coordinate frames, history/lookahead schedules,
  previous actions, the 11-dimensional morphology vector, and normalization.
- 4.3 Network/data flow: token encoders, slot/type attention, pooling, morphology/action bypass,
  separate actor/critic heads, and the policy-action-to-PD-target transformation.
- 4.4 Learning procedure: tracking rewards, termination/reset rules, rollout collection, and PPO.
- 4.5 Small-scale development and direct full-scale training: morphology-consistent sampling,
  clip streaming, priority selection, uniform rehearsal, and held-out monitoring.

### 5. Experimental Setup

- 5.1 The 150-clip development subset and full-scale training protocol.
- 5.2 Architecture/refinement controls, matched budgets, parameter counts, and random seeds.
- 5.3 Morphology-matched evaluation, checkpoint selection, fixed final evaluation pairs, and
  separate tests of motion versus body-shape generalization.
- 5.4 Exact success/error/smoothness definitions, aggregation, and compute accounting.

### 6. Results and Failure Analysis

- 6.1 Full-scale held-out motion tracking, distinct from hard resident-pool monitoring.
- 6.2 Corrected architecture comparison: MLP, attention, slot/type, and actor-only AdaLN.
- 6.3 Refinement effects: reference quality/fidelity and a matched policy-level comparison.
- 6.4 Cross-shape consistency; unseen-body results only if those evaluations are completed.
- 6.5 Residual failure frequency, representative examples, and evidence-supported explanations.

Use text-labelled, multi-shape qualitative examples to connect the dataset and controller results.
Do not infer full 128-shape coverage from an evaluation using one selected shape per clip.

### 7. Physical Analysis — Placeholder

Reserved for paired comparisons of the same motion across body shapes, including joint torques,
contact forces, and related physical quantities. No results or causal claims at this stage.

### 8. Discussion and Limitations

Scope of the evidence; generated-reference and simulator limitations; body-shape generalization;
caption/data limitations; compute constraints; and optional future dataset development.

### 9. Conclusion

Summarize demonstrated contributions without extending beyond the evaluation performed.

### Appendix / supplementary material

Detailed configurations, refinement settings, split provenance, additional qualitative examples,
and reproducibility instructions. Keep neutral-transfer and mixed-source experiments historical;
pre-fix measurements cannot establish architecture rankings or an intrinsic success ceiling.

## 3. Keypoint list and contribution claims

### K1. Dataset contribution: multiple body shapes associated with source text

**Proposed claim:** We construct a multi-morphology extension of HumanML3D, associating source
motion descriptions with shape-specific generated references, matching simulation assets, and
reproducible evaluation splits.

What exists now:

- 20,951 curated clip IDs, including mirrored clips, with 128 body configurations each:
  2,681,728 generated motion variants. These are not independent motion captures or new captions.
- 128 configurations comprise 64 SMPL beta vectors crossed with two body-model sex variants,
  not 128 independently sampled beta vectors.
- HUMOS-generated references, the refinement pipeline and refined corpus, and corresponding
  morphology assets. MotionLib retains clip and asset identities.
- Source text is available separately, including
  `/home/hlz/repos/hhi/data-processing/motion_id_text.json` and the HUMOS HumanML3D annotation
  metadata. Existing analysis uses clip-ID lookup; text is not established as a packaged field
  of every training `.pt` file.
- Versioned 90/5/5 splits: 18,855 training, 1,048 validation, and 1,048 test clips. Mirrored
  partners and every shape variant of a clip stay together. The 150 development clips are
  reserved for training. This is not a demonstrated subject- or source-recording-disjoint split.

The value is associating the same described action with different body configurations, enabling
paired studies and future text-and-shape-conditioned models. The contribution is a derived
corpus and its construction, not authorship of the original text annotations.

**Still to verify:** full caption coverage, segment timing, mirrored left/right wording,
body-specific wording, and semantic fidelity after generation/refinement. Stripping an `M`
prefix for lookup alone does not establish correct mirrored captions. Do not claim a fully
audited public release, new shape-specific language annotations, or text-conditioned control.
Further packaging or annotation work can be developed later.

### K2. System contribution: one controller at combined motion and morphology scale

Explain how shape-specific references, matching simulation bodies, morphology inputs, temporal
slot/type attention, and scalable sampling support one shared policy. The full-scale training
set contains 18,855 clips across 128 body configurations; held-out evaluation must establish
performance separately. Position this as a systems contribution, not a claim to have invented
morphology conditioning or a fundamentally new transformer architecture.

### K3. Empirical contribution: supported findings about data and policy design

Use the corrected 150-motion ablations: A = temporal MLP, B = basic attention, C = slot/type,
D = slot/type plus actor-only AdaLN, all on refined references; E = C on unrefined references.
Only seed 0 is currently reported running. Architectural advantages and refinement benefits are
pending findings, not settled claims. Final C/E comparisons need common evaluation references;
their training-time metrics otherwise use different targets. Report parameter counts rather
than claiming that the architectures are capacity-matched.

### K4. Reference quality and fidelity are both necessary

Measure sliding, penetration, smoothness, and rotation/root changes. Show whether shape-specific
motion differences are preserved. Kinematic cleanup does not guarantee dynamic feasibility;
the matched learning experiment must establish any controller benefit. Do not assume the
150-clip result proves the same causal effect at full scale.

### K5. Define exactly what generalizes

Distinguish unseen motions on trained bodies, seen motions on unseen bodies, and unseen motions
on unseen bodies. A rotating one-shape-per-clip validation score does not demonstrate all-shape
success. Use identical clip/shape pairs for final model comparisons, and do not describe the
frequently monitored validation holdout as an untouched test set.

### K6. Explain residual failures without declaring them impossible

Separate reference artifacts, unavailable environmental support, morphology-specific difficulty,
controller/optimization limitations, and metric effects. Support categories with observed
evidence and retain an unresolved category. Do not assume a fixed 10% failure tail or use the
pre-fix 80% ceiling as evidence of physical infeasibility.

### K7. Describe the procedures explicitly and make them reproducible

For both dataset construction and learning, specify inputs, operations, outputs, parameter
values, and a brief rationale for each design choice. Include the AMASS/HUMOS motion path and
SMPL/SMPLSim body path, their shared beta/asset identities, the refinement order, and the full
observation-to-action data flow. Readers should be able to reconstruct the approach without
following the project's historical notes. Unverified design rationales remain hypotheses.

Report small-scale and full-scale budgets, GPU-hours, seeds, configurations, and split hashes.
The corrected reruns reassess earlier choices; do not imply they preceded the full-scale run.
Morphology-matched evaluation is required correctness, not a standalone algorithmic novelty.
Cost-conscious development is a documented practice, not proof of superior compute efficiency.

### K8. Keep physical interpretation explicitly deferred

Reserve the fourth potential contribution for paired mechanics analysis after instrumentation
and measurements are validated. Until then, simulated asset parameters and reference kinematics
are not measured human biomechanics, and this section stays a placeholder.

## 4. Required procedural detail in Dataset and Method

Keep the main section structure above, but write Sections 3 and 4 as explicit procedures, not
lists of component names. Include the essential choices in the main text; place exhaustive
parameter tables and commands in the appendix. The following records current implementation
details to cover, not additional experiments that must be developed first.

### 4.1 Dataset construction procedure

The overview figure should show three linked paths:

```text
Motion: AMASS -> preprocessing / HumanML3D clip extraction -> HUMOS(beta, body variant)
             -> MotionLib conversion -> contact-aware refinement -> per-clip references
Body:   SMPL(beta, body variant) -> SMPLSim -> MJCF assets + morphology metadata
Text:   HumanML3D descriptions -> clip-ID association with the generated variants
```

The body assets are used during conversion, morphology-specific FK, collision clearance, and
simulation. The SAME beta vector and body-model variant must identify each reference and asset.
Text is an associated metadata path, not a language input to the current policy.

1. **Source preparation.** Specify the AMASS subset used, SMPL+H source representation,
   coordinate/joint conventions, preprocessing at 20 fps, HumanML3D clip/segment extraction,
   and mirroring. Describe the treadmill/skating cleanup and later text-based exclusion of
   clips requiring external support or objects. Preserve the source IDs and segment metadata;
   do not describe the retained subset as all of AMASS.
2. **Body-shape sampling.** Sample 64 vectors with ten beta coefficients independently uniform
   in `[-3, 3]`, using `numpy.random.default_rng(46)`. Cross them with female/male SMPL body-model
   variants to obtain 128 configurations. Retain the actual beta file and stable beta keys;
   use identical values for HUMOS conditioning and SMPLSim asset generation. Do not describe
   uniform coefficient sampling as a representative distribution of human body shapes.
3. **Shape-specific reference generation.** Explain what processed source-motion features enter
   HUMOS, how beta/body-model conditioning is supplied, and what pose/root sequences are saved.
   Record the HUMOS revision and checkpoint: the documented local configuration selects
   `logs/humos/q6zbv2tu/checkpoints/latest-epoch=1599.ckpt`. Verify the feature layout and
   preprocessing against that revision; do not copy the historical notes' diffusion-model
   label or conflate HumanML3D feature vectors with HUMOS's configured training features.
4. **Simulation-asset generation.** Describe SMPL body construction from the same parameters,
   followed by SMPLSim MJCF generation. Report skeleton/joint conventions, collision proxies,
   hand/foot simplifications, mass/inertia estimation, and the asset manifest connecting
   body-model variant, beta key, and asset ID. Asset physical parameters are simulation-model
   estimates, not measurements from corresponding human participants.
5. **Conversion and refinement.** Explain the conversion to ProtoMotions coordinates and
   MotionLib tensors at 30 fps. The existing corpus had frame-0 grounding; the newer refinement
   also addresses later frames. State the actual refinement order: validate/detect and clean
   ankle/toe contacts -> smooth joint rotations with contact-transition guards -> per-shape FK
   -> conservative least-squares stance-root XY correction -> collision-geometry ground
   clearance -> final FK and per-motion velocity reconstruction. Report thresholds, windows,
   correction limits, and before/after fidelity measures. Root-only correction is not exact
   foot locking, and the procedure does not solve full-body dynamics.
6. **Packaging and splitting.** Describe one file per clip containing its 128 variants, motion
   durations/fps, clip/asset IDs, beta/body-model metadata, and kinematic/contact tensors. Keep
   captions associated by source identity, with their current validation limits explicit.
   Document the 90/5/5 grouping, development-clip reservation, split seed/version/hashes, and
   access through explicit train/validation manifests. Do not expose test motions to training.

### 4.2 Observation design and policy data flow

Use an observation table with the variable, physical meaning, coordinate frame, units,
dimension, temporal slots, normalization, and actor/critic destination. Fill dimensions from
the actual asset and resolved configuration, not the historical draft's observation counts.

| Input | Current implementation | Network path |
|---|---|---|
| Current body state | Root height above ground; root-relative, heading-aligned body positions; 6D body rotations; linear/angular velocities. Contact flags are disabled. | One current-state token. |
| State history | Same state representation at configured past steps `[1, 2, 3, 4, 8, 16, 32]`; describe reset/history initialization. | Seven tokens, with an encoder shared across history slots. |
| Future reference | Four target slots at `[1, 2, 4, 8]` control steps ahead, including root-relative targets, pose-relative differences, and velocity information. | Four tokens, with an encoder shared across future slots. |
| Previous action | One raw historical policy-action vector, before tanh/PD scaling. | Concatenated directly into each final head. |
| Morphology | `[body_model_id, beta_1, ..., beta_10]`, eleven values from the environment's assigned asset. | Concatenated directly into each final head. |

Explain the rationale for local coordinates, short/long temporal context, previous actions,
and explicit body information, while leaving their measured benefit to the ablations. Text,
AdaLN conditioning, and physics-feature vectors are not inputs/components of the selected
slot/type-only policy.

Document these network operations in order:

1. Separate current/history/future MLP encoders normalize their inputs (running mean/std,
   clamp magnitude 5) and project each slot to 256 dimensions. Each encoder has two 256-unit
   ReLU hidden layers; its weights are shared across the slots of its own type.
2. Concatenate 1 current + 7 history + 4 future tokens. Add learned embeddings for 12 temporal
   slots and three token-source types. Run two attention layers with four heads and a
   1,024-unit feed-forward block; pool from the current-state token.
3. Concatenate the 256-dimensional attention output, previous action, and morphology vector.
   The final MLP input also uses running normalization and clamp magnitude 5; beta values are
   not manually divided by three or converted to a separate conditioning token.
4. Use separate actor and critic pipelines: actor head = six 2,896-unit ReLU layers, critic
   head = four 1,024-unit ReLU layers, followed by action-mean and scalar-value outputs.
   Slot/type embeddings are enabled in both; their network parameters are not shared.
5. Describe the stochastic policy used during PPO and deterministic mean-action evaluation.
   Map raw actions through `q_target = offset + scale * tanh(action)` and built-in PD control.
   Offsets/scales come from the configured joint ranges. This experiment does NOT use the
   older reference-pose-plus-residual action mapping.

### 4.3 Training and evaluation procedure

- **Simulator/control setup:** report the SMPL morphology assets, flat-ground setup, gain and
  effort-limit tables, and enabled mass-scaled PD gains. Current IsaacGym robot defaults are
  60-Hz simulation stepping with two substeps and decimation 2 (30-Hz policy control).
- **Reference/environment assignment:** describe fixed per-environment morphology, selecting
  only matching reference variants, reference-state initialization, random clip start times,
  history initialization, and resampling on reset. The experiment sets start-at-frame-zero
  probability 0.2 and maximum training episode length 1,000 control steps.
- **Reward and termination:** write the implemented tracking reward equation and reductions
  for body position, rotation, linear/angular velocity, and root height. Configured weights
  are `0.5, 0.3, 0.1, 0.2, 0.2`, with exponential coefficients
  `-25, -5, -0.5, -0.1, -100`, respectively. Fall termination uses a 0.15-m height setting;
  tracking-error termination and action-smoothness/effort/contact-match reward terms are
  absent. Distinguish training termination from the evaluator's success criterion.
- **PPO optimization:** specify rollout collection, advantage estimation/normalization,
  minibatching, losses, clipping, optimizer, and stopping budget. Current experiment learning
  rates are actor `2e-5` and critic `1e-4`. Report all other settings and any resumed-run
  changes from saved resolved configurations and logs, rather than launch-template defaults.
- **Small-to-full-scale procedure:** explain the five matched 150-clip comparisons and the
  selected direct multi-shape full-scale run. Describe GlobalClipPool residency, rebuild
  cadence, priority selection and uniform rehearsal, weight updates, and fixed validation
  membership. Report the actual pool-size/cache/environment changes during the full run.
- **Evaluation and provenance:** explain morphology-matched rollout scheduling, shape-panel
  selection, deterministic inference, metric aggregation, checkpoint selection, and the
  separate held-out evaluation. Preserve configuration, checkpoint, and split versions;
  describe the evaluation correction without interpreting its jump as instant policy learning.

Minimum supporting material: one data/asset pipeline figure, one observation-to-action network
figure, one observation-definition table, and reproducible preprocessing/training parameter
tables. These figures explain procedures; they do not substitute for experimental evidence.

## 5. Immediate drafting priorities

1. Use K1–K3 as the provisional headline contributions, with K4–K7 specifying their evidence and
   boundaries. Keep K8 deferred.
2. Write the Dataset, Method, and Experimental Setup sections from the current implementation.
3. Prepare result-table outlines, marking corrected ablations, test evaluation, unseen-body
   evaluation, and caption-alignment checks as pending where applicable.
4. Fill conclusions only after those measurements; then finalize Introduction and Abstract.

## 6. Source context

- [Current split metadata](../data/splits/hhi_stage2_v1/split_metadata.json).
- [Chronological data pipeline](README.data-pipeline-chronological.md), especially text-based
  filtering; [historical caption lookup](README.failed-motions.md).
- [Current project findings](README.note.md), sections 75–77: evaluation correction, full-scale
  run, and newly launched corrected ablations.
- [Failure/physics evidence plan](README.failure-and-physics-analysis-report.md).
- [Full-scale slot/type experiment](../examples/experiments/mimic/mlp_wide_stage2_discover_attention_slot_type.py)
  and its [base configuration](../examples/experiments/mimic/mlp_wide_stage2_discover_attention.py).
- [Observation factories](../protomotions/envs/component_factories.py),
  [action processing](../protomotions/envs/action/action_functions.py), and
  [SMPL morphology robot configuration](../protomotions/robot_configs/smpl_mor.py).
- [Refinement implementation](../tools/refine_humos_motion.py); external local construction
  sources `/home/hlz/repos/SMPLSim/run.py`, `/home/hlz/repos/humos/humos/infer.py`, and
  `/home/hlz/repos/humos/humos/configs/cfg_template.yml`. Cite/version the actual revisions
  when writing the manuscript; historical notes can contain obsolete labels and settings.
- [HumanML3D documentation](https://github.com/EricGuo5513/HumanML3D): source annotations,
  mirrored-caption conventions, and temporal annotation fields.
- [HUMOS](https://humos.is.tue.mpg.de/): the existing body-shape-conditioned motion model.
- [Won and Lee, 2019](https://mrl.snu.ac.kr/publications/ProjectMorphCon/MorphCon.pdf): prior
  physics-based body-shape-conditioned control; conditioning itself is not a novelty claim.
