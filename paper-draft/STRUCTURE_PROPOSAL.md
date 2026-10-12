# Proposed paper structure

Prepared against the current manuscript sources on 12 October 2026. This is a structural design and content-migration plan; the manuscript and its reported measurements have not been changed.

## Reference and organizing principle

The linked paper is *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills*. ACM blocked direct PDF access, so the structural review used the [authors' arXiv version, v3](https://arxiv.org/pdf/1804.02717), which identifies the same DOI. This version has 18 pages; the [ACM record](https://doi.org/10.1145/3197517.3201311) lists 14 pages, so pagination should not be treated as identical.

Its numbered sections are:

1. Introduction
2. Related Work
3. Overview
4. Background
5. Policy Representation
6. Training
7. Multi-Skill Integration
8. Characters
9. Tasks
10. Results
11. Discussion and Limitations

Policy Representation contains States and Actions, Network, and Reward. Training separates Initial State Distribution and Early Termination. Results contains Tasks, Multi-Skill Integration, Retargeting, Ablations, and Robustness.

The transferable pattern is a short overview, distinct technical components, and capability-oriented results. For our manuscript, the equivalents are body-specific references, one shared controller, scalable training, and motion/body transfer. A separate RL tutorial and sections on goal-directed tasks or skill sequencing would not serve our current evidence.

## Recommended narrative

**One shared physics-based controller can track a large repertoire of body-specific motions and transfer to unseen body shapes, while high average success conceals severe failures on some bodies.**

The paper should answer these reader questions in order:

1. Why does body variation make shared motion imitation difficult?
2. What does the complete system take as input and produce?
3. How are a body's reference motions and simulation assets kept consistent?
4. How does the shared policy use temporal and morphology information?
5. How is it trained over the large paired corpus?
6. What can it do, how far does it transfer, and what limits the evidence?

The main structural change is to separate **Policy Representation** from **Training** and integrate the essential **Evaluation Setup** into **Results**. The number of sections is secondary to making each section perform one clear job.

## Proposed table of contents

Abstract

1. Introduction
2. Related Work
   - 2.1 Physics-Based Motion Imitation
   - 2.2 Body Variation and Morphology-Conditioned Control
   - 2.3 Shape-Specific Motion Generation
3. System Overview
4. Morphology-Specific Motion References
   - 4.1 Source Motions and Body Configurations
   - 4.2 Paired Reference and Simulation-Asset Construction
   - 4.3 Contact-Aware Reference Refinement
   - 4.4 Corpus Scale and Motion Splits
5. Policy Representation
   - 5.1 State, Reference, and Morphology Inputs
   - 5.2 Temporal Slot and Type Attention
   - 5.3 Actions and Physical Actuation
6. Training
   - 6.1 Tracking Objective
   - 6.2 Reference Initialization and Episode Termination
   - 6.3 Shared Learning Across Bodies
   - 6.4 Streamed Motion Sampling
7. Results
   - 7.1 Evaluation Setup
   - 7.2 Motion Imitation Across Trained Bodies
   - 7.3 Transfer to Unseen Body Shapes
   - 7.4 Architecture and Reference-Refinement Comparisons
   - 7.5 Failure Cases and Checkpoint Sensitivity
8. Discussion and Limitations
9. Conclusion

Statements and Declarations; References; Appendices. Their final placement should follow the target journal's template. Retain the draft's current journal target; borrowing the reference's organization does not require changing venue or document class.

## What each section should accomplish

### Abstract

Use five moves: problem, system, evaluated scale, principal results, central limitation. Report the held-out-motion result with its one-trained-body-per-clip scope, and the unseen-body results with their known-motion scope. Include the uneven per-body robustness. Avoid an inventory of implementation details.

### 1. Introduction

Use approximately five paragraphs:

1. Establish the animation need: reuse a motion repertoire across digital humans with different proportions and dynamics.
2. Explain the challenge: both reference geometry and required actuation change with body shape; motion generalization and body transfer are separate axes.
3. Present the approach: matching body-specific references and assets, one temporal controller, and streamed training.
4. State three contributions: the paired corpus/workflow; the evaluated shared controller and training configuration; and the systematic transfer evaluation with supporting design comparisons and reference audits.
5. Summarize the findings and their practical limit, then give a short roadmap.

Remove the standalone RQ1/RQ2/RQ3 enumeration and repeated contribution paragraph. Preserve those questions as the underlying logic of Results. Credit inherited components once and identify the study's additions precisely; avoid repeatedly defending the same novelty boundary.

### 2. Related Work

Keep the three focused literature themes above. End each with the relationship to this study. Move the full published-capability table into an appendix and keep a short positioning paragraph in the main text. Place the inherited-components versus study-additions statement here, with a brief corresponding contribution statement in the introduction.

Retain the substantive distinctions: body variation is established prior work; topology transfer differs from continuous body-shape variation; HUMOS supplies references; and no matched external performance comparison has been run. Move generic Transformer and adaLN descriptions to the relevant method/comparison passages.

### 3. System Overview

Give the reader the complete pipeline in three short paragraphs and one figure:

- Inputs: source motion and target body parameters.
- Preparation and learning: generate and refine a body-specific reference; construct its matching simulation asset; train one policy across these pairs.
- Execution: present state, history, future reference targets, and morphology map to PD targets and a simulated trajectory.

Keep the compact task definition and shared-policy notation from the existing problem formulation. Explain that captions serve filtering and metadata. Show the common body identity connecting reference generation, asset construction, and replay. Leave network dimensions, split algorithms, reward equations, and evaluation hashes for their owning sections.

### 4. Morphology-Specific Motion References

Make the current corpus section a technical component of the system. Explain what the controller is asked to track before explaining how it tracks it.

- **4.1:** Introduce AMASS/HumanML3D source identities, filtering, and the 64 coefficient vectors instantiated on two SMPL templates. Distinguish clips from generated variants.
- **4.2:** Explain HUMOS generation, matching SMPLSim assets, coordinate conversion, resampling, and shape-specific forward kinematics. Connect identity consistency to valid tracking supervision.
- **4.3:** Present the refinement sequence and its rationale: contact validation, bounded rotation filtering, root correction, clearance correction, and recomputed kinematics. Keep the clearance equation; move parameter-level settings to an appendix.
- **4.4:** Keep one corpus/split table, clip/mirror grouping, and the development subset's relationship to training. Explain the derived nature of the variants.

Move the long caption-alignment audit, source-collection list, tensor-field inventory, and resource-release discussion to appendices. Keep a concise sentence about captions and resource scope here. Put measured before/after refinement outcomes in Results, rather than interleaving them with the procedure.

### 5. Policy Representation

This section should explain a single control step.

- **5.1:** Describe heading-relative state, historical states, future targets, previous action, and morphology. Retain the observation table and temporal offsets.
- **5.2:** Explain the encoders, slot/type embeddings, attention, current-token pooling, and separate actor/critic pathways. Keep the network equation and diagram. Make the morphology/action bypass clear. Put a compact layer specification here or in the appendix, without repeating it in prose and the figure.
- **5.3:** Explain the 69-dimensional action, tanh/range mapping, PD actuation, control frequency, and per-body mass scaling. Keep the action equation and refer to appendix gains.

Use the name **Policy Representation** instead of the broad current title **Proposed Methodology: Shared Controller and Training**. Architecture and action choices belong here; PPO and corpus sampling belong in Training.

### 6. Training

Explain how the policy is learned, with the tracking objective first.

- **6.1:** Keep the position, rotation, velocity, angular-velocity, and height errors and reward equation. Define units and reductions once.
- **6.2:** Explain reference-state initialization and training fall termination. Distinguish training termination from the evaluation success criterion.
- **6.3:** Describe direct multi-body learning and morphology-matched reference selection. State that the selected full-scale model trains directly on refined references. Give PPO settings in a compact table or appendix reference.
- **6.4:** Explain bounded clip residency, rank partitioning, priority sampling, and uniform rehearsal. Keep enough detail to establish that the resident pool is difficulty-biased and differs from fixed validation.

Move memory-related restart history, old stage naming, cache changes, and detailed launch settings to provenance. Report actual training scale and budget compactly in Results 7.1; keep exact configurations in the appendices.

### 7. Results

Begin with the capability being evaluated. Keep protocols concise but sufficient to interpret every result. Use one short experimental-intent sentence at the start of each subsection; use an evidence scope sentence after its principal result.

**7.1 Evaluation Setup.** Consolidate the evaluation-population table, checkpoint roles, metrics, success threshold, and horizon definitions. State trained versus unseen body exposure and held-out versus known motion exposure explicitly. Distinguish the original checkpoint from the full-data candidate, including the latter's exposure to the original motion splits. Keep body-matched replay, deterministic mean actions, seed counts, unequal architecture capacities, and the meaning of uncertainty. Move the SHA-256 selection formula and detailed monitoring/resume machinery to reproducibility.

**7.2 Motion Imitation Across Trained Bodies.** Lead with the test table: 99.14% success and continuous tracking metrics on 1,048 held-out clips, one selected trained body per clip. Follow with the 150-by-128 trained-body replay and a documented visual demonstration. These are complementary coverage measurements; do not imply an all-body held-out-motion test. Move the residual failure discussion to 7.5.

**7.3 Transfer to Unseen Body Shapes.** Lead with the two original-checkpoint averages, 97.28% and 97.15%, on the same 233 known clips crossed with each 128-body panel. Immediately show the sorted per-body distribution and the worst-body outcomes, 3.0% and 13.3%. Keep a brief coefficient-tier/property summary. Explain that the trained-body control uses a different clip panel. Move detailed bins and asset listings to the complete-results appendix. This subsection should establish both the transfer capability and its unevenness before discussing checkpoint changes.

**7.4 Architecture and Reference-Refinement Comparisons.** Use two paragraph-level parts, each with its own table: three-seed architecture trade-offs, then the paired offline refinement audit. Keep the reference sliding reduction together with its increased reference jerk and small measured pose changes. Call these comparisons rather than a definitive causal ablation: architecture capacities differ, all models retain morphology inputs, and policies trained on different reference versions track different targets. Do not describe the supporting architecture reruns as prospective full-scale model selection.

**7.5 Failure Cases and Checkpoint Sensitivity.** Bring together residual held-out-motion failures and the original/candidate replay changes. Keep a compact checkpoint-delta table and representative gains/regressions; move the full asset regression table to an appendix. Explain simulator rerun variability and the confounding of training duration and motion exposure. Describe observed failure patterns without assigning unverified mechanical causes. Return to why pooled success alone is insufficient.

Keep the worst-body evidence in 7.3 even when moving deeper diagnostics to 7.5. The transfer claim should never be separated from its most important qualification.

### 8. Discussion and Limitations

Organize around implications, rather than repeating RQ summaries:

1. Reusing one controller across tested body configurations, and what this supports for animation workflows.
2. Why strong average transfer does not imply reliable tracking on every body.
3. What the design comparisons establish, including the missing matched removal of explicit morphology inputs.
4. Evidence boundaries and future work: all-body held-out-motion coverage, matched external trackers, reference feasibility and failure diagnosis, broader environments, and audited resource release.

State each limitation where it affects a result, then discuss its wider consequence here. Consolidate repeated caveats; retain their substance. Treat inferred mass/property associations as descriptive and distinguish reference contact audits from simulated-output contact quality.

### 9. Conclusion

Keep one short paragraph. Restate the paired-reference/shared-controller contribution, the average tracking and transfer finding, and the individual-body robustness limitation. Avoid another RQ-by-RQ summary or a second detailed future-work list. A separate conclusion can remain for the current journal manuscript even though the reference ends with discussion.

## Migration from the current files

Paths below are relative to `paper-draft/`.

| Current source | Proposed destination | Main action |
|---|---|---|
| `sections/01_abstract.tex` | Abstract | Condense around approach, evidence, and main limitation |
| `sections/02_introduction.tex` | 1 | Replace the RQ list and duplicated contribution statement with a continuous argument |
| `sections/03_related_work.tex` | 2 + Appendix D | Keep focused positioning; relocate the full capability table |
| `sections/00_problem_and_overview.tex` | 3 | Expand the conceptual pipeline; keep compact formulation |
| `sections/04_dataset.tex` | 4 + Appendix A | Retain the construction/refinement method; move detailed metadata audits |
| `sections/05_method.tex`: task/actuation, observation, attention | 5 | Order inputs, network, then actions |
| `sections/05_method.tex`: reward/resets/PPO, sampling | 6 + Appendix B | Separate learning from one-step policy representation |
| `sections/06_experimental_setup.tex` | 7.1 + Appendices B/D | Keep essential inference scope; move deterministic selection and monitoring mechanics |
| `sections/07_results_and_failure_analysis.tex`: RQ1 | 7.2 and 7.5 | Keep tracking/coverage together; move residual failures |
| Same file: RQ2 | 7.3, 7.5, and Appendix C | Keep averages and body distribution together; relocate deeper checkpoint diagnostics |
| Same file: RQ3 | 7.4 and 8 | Keep comparison evidence; consolidate mechanism limitations |
| `sections/09_discussion.tex` | 8 | Remove repeated RQ framing; retain substantive boundaries |
| `sections/10_conclusion.tex` | 9 | Short closing paragraph |
| `sections/11_reproducibility.tex` | Appendices A/B | Divide data preparation from training/evaluation details |
| `sections/12_experiment_provenance.tex` | Appendix B | Retain monitoring snapshots, historical comparisons, and compute records |
| `sections/13_unseen_body_tables.tex` | Appendix C | Keep complete body coverage and add detailed diagnostic tables |
| `sections/08_physical_analysis.tex` | Appendix D | Keep exploratory/descriptive status explicit |
| `sections/15_declarations.tex` | Unnumbered back matter | Retain pending author completion |

Suggested appendix grouping:

- **A. Reference Construction and Data Audits:** source collections, processing parameters, caption checks, resource scope.
- **B. Training, Evaluation, and Provenance:** optimizer/PD settings, metric details, exact body-selection rule, split/checkpoint identities, monitoring history, hardware and compute.
- **C. Complete Body-Transfer and Checkpoint Results:** all unseen bodies, detailed property bins, regressions, and diagnostic motion subsets.
- **D. Related-System Comparison and Exploratory Physical Analysis:** published capability table and explicitly descriptive physical analysis, in separate subsections.

## Figure and table plan

| Placement | Visual | Purpose and status |
|---|---|---|
| First page / Introduction | Teaser showing several body shapes tracking the same motion | Add a real documented rollout image; the current placeholder is not ready |
| Overview | Combined reference/asset/training/execution pipeline | Adapt the existing data-pipeline diagram; distinguish preparation from execution |
| Reference construction | Corpus and split table | Retain exact counts; remove the separate scale diagram if it duplicates the table |
| Policy representation | Controller diagram and input table | Reuse and simplify existing material; show the morphology/action bypass accurately |
| Results 7.1 | Evaluation populations table | Make motion exposure, body exposure, checkpoint, and horizon readable together |
| Results 7.2 | Test tracking table and trained-body visual | Pair numerical capability with documented execution |
| Results 7.3 | Unseen-body averages and sorted per-body success plot | Show aggregate transfer and the severe tail together |
| Results 7.4 | Architecture table; refinement table and distribution plot | Show the design trade-offs and distinguish reference measurements from rollout measurements |
| Results 7.5 | Compact checkpoint-change table; failure examples if available | Show recoveries/regressions; add visual diagnosis only when supported by traces |

Replace standalone prose that repeats a figure caption with a sentence stating the figure's scientific takeaway. Do not invent screenshots, timing records, contact traces, or additional experiments to fill the design.

## Editorial length target

As an editorial target, allocate roughly 35-40% of main-text space to Results, 35-40% to Overview/References/Policy/Training together, and the remainder to Introduction/Related Work/Discussion/Conclusion. This is a relative allocation, not a journal page limit. Reduce administrative detail and repeated caveats before shortening the evidence that determines the central claim.

## Acceptance criteria for the later manuscript revision

- The introduction, overview, and results describe the same shared-controller contribution.
- Control-step mechanics and training mechanics have separate homes.
- Main results use descriptive headings; the three research questions remain answerable without numbered RQ labels.
- Motion exposure, body exposure, checkpoint, and horizon accompany each headline result.
- No existing numerical result is silently changed or given broader scope.
- High average unseen-body success is presented beside poor worst-body success.
- Architecture/refinement evidence keeps its capacity, seed, target, and causal-inference limits.
- Detailed reproducibility remains accessible, with repaired cross-references after migration.
- Figures use real documented evidence; author/declaration placeholders remain visible until supplied.

This proposal does not require new experiments. Stronger mechanism, robustness, and comparative claims would require the additional controls already identified in the draft.
