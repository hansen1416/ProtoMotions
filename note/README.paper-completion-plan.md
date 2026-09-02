# Paper Completion Plan

**Date:** 2026-09-01  
**Status:** Active plan for the final experiments and paper restructuring.

## 1. Final paper story

The main pipeline should be:

> AMASS motion content → HUMOS shape-specific motions → physical refinement → GlobalClipPool → one slot/type-attention morphology-conditioned policy.

The 150-motion corpus is the compute-efficient development set used to choose preprocessing, rewards, and architecture. The selected recipe is then trained once at full scale.

The previous two-stage neutral-pretraining → shape-transfer direction is not the main method. Mention it briefly as a negative result:

> Neutral pretraining followed by shape transfer was tested, but the transfer variants plateaued below the required performance; we therefore selected direct multi-shape training.

Likewise, the unsuccessful mixed AMASS/HUMOS source-weighted pipeline belongs in the ablation/history discussion, not in the main Method pipeline.

## 2. Refinement evidence

Before committing to another full training control:

1. Select a fixed sample of approximately 200–500 clips.
2. Compare the original and refined HUMOS references using:
   - Foot skating
   - Ground penetration
   - Normalized jerk
   - Contact consistency
   - Rotation fidelity
3. Produce paired before/after distributions and several qualitative videos.
4. Run a matched 150-motion slot/type experiment on unrefined HUMOS if this exact control does not already exist.

The small-scale comparison establishes whether refinement improves the training signal under the paper's compute-aware protocol. Keep the full unrefined slot/type run in reserve until the refined full-scale result is known.

## 3. Static final evaluation set

GlobalClipPool metrics are useful for training monitoring, but the paper needs one reproducible evaluation set:

- Approximately 100–200 fixed held-out clips
- All 128 training shapes per clip
- Identical clip/shape pairs for every evaluated checkpoint
- Deterministic inference

Report:

- Success rate
- DOF rotation error
- Root-position error normalized by body height
- Root-orientation error
- Normalized jerk
- Foot skating and contact quality

Describe DOF error as **scale-comparable articulation error**. HUMOS provides shape-specific rotation targets, so it is incorrect to claim that all shapes have an identical joint-angle target.

## 4. Cross-shape consistency

For each evaluation clip, summarize performance over its 128 body shapes:

- Median error
- IQR or standard deviation
- Worst-shape error
- Success fraction
- Relationships with mass, height, limb proportions, and beta extremity

This is the minimum evidence needed to support the claim that one policy controls a broad morphology distribution.

## 5. Held-out body shapes

Use a staged, practical evaluation:

1. Generate 16 unseen interpolation shapes within the training beta range.
2. Create their SMPL assets and HUMOS references for approximately 50–100 fixed evaluation clips.
3. Apply the same HUMOS refinement process.
4. Evaluate with the same metrics used for the 128 training shapes.
5. Add extrapolation shapes only if interpolation works reliably.

If the held-out-shape pipeline remains unreliable, narrow the paper's claim to **multi-shape control across 128 trained morphologies** rather than morphology generalization.

## 6. Physical analysis

After selecting the final checkpoint:

1. Validate instrumentation on 8 clips × 16 shapes.
2. Run the paired torque, work, contact, and stability analysis on the fixed evaluation subset.
3. Run the correct-beta versus neutral/shuffled-beta causal experiment.
4. Produce the joint-load heatmap, physical-feature trends, result table, and qualitative video.

The complete protocol is documented in `note/README.physics-analysis-plan.md`. These experiments require inference rather than additional policy training and should be much cheaper than a new full-scale run.

## 7. Full unrefined control decision

Decide whether to run the full unrefined slot/type control only after inspecting the refined full-scale result:

- **Clearly strong refined result:** run the full unrefined control to support a strong policy-level refinement claim.
- **Modest result:** rely on the matched 150-motion training comparison and the full-dataset reference-quality analysis.
- **No training benefit:** present refinement only as data cleaning and avoid claiming that it improves policy learning.

## 8. Recommended execution order

1. Restructure the paper around the final refined-HUMOS/slot-type pipeline.
2. Aggregate the reference-refinement evidence.
3. Complete the refined full-scale training.
4. Freeze and run the static 128-shape evaluation set.
5. Run the held-out interpolation-shape evaluation.
6. Run the paired physical and wrong-beta analyses.
7. Decide whether the full unrefined training control is justified.
8. Finalize tables, figures, qualitative video, abstract, and conclusion.

This order places a decision point before every expensive step and keeps the paper focused on the final method rather than the complete research history.
