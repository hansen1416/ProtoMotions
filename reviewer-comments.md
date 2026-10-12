Nick:

I recommend changing the title to (or anything similar): Morphology-Conditioned Physics-Based Motion Imitation for Diverse Human Body Shapes

The paragraph at the end of the introduction should be generic. The remaining of the paper is structured as follows: Section 2 contains the related work, section 3 delivers the methodology … and so on. (not specific mentions to subsections or results)

Figures 1 and 2 are next to each other. Can you move fig 2 and put some text between?
Is section 4 the proposed methodology? I think it should be renamed to make this clear.
In section 4 for both equations 5 and 6: there are two equations in each. Should these be separate? If not it should be clear what we see here.
In section 5. We need to compare the proposed methodology with 2-3 other state of the art in a dedicated subsection. Even if these aren’t well suited for this problem we need to make this clear and easy to find perhaps using a subsection to elaborate on this. If these can be executed for this problem then we need a subsection with the results.
What is physical analysis and why is it a separate section? Can it be a subsection of 6?
Regarding all figure captions: Please reduce the caption to one line max for each and make sure all figures are mentioned in the text before each figure and it is well explained (in the same text) what the figure shows.


Almas:

Concerns and Required Revisions

- [Major] The architecture selection and headline numbers are single-seed, single-panel, single-checkpoint, and the selection is retrospective. All eight grid runs use seed 0; the full-scale slot/type run was launched on earlier (pre-correction) small-scale runs, with the corrected grid launched later to reassess that choice; the grid's discriminating axis is jerk, while success (96–98%) is explicitly non-discriminating. The 98.47/99.14% figures come from one panel index, one checkpoint, and a 0.5 m threshold on per-frame mean body error, with no confidence intervals; the 0.67-point validation–test gap is within the ±0.7-point 95% Wilson interval of a single 1,048-sample measurement. Consequence: the claim that slot/type attention is the best configuration, and the precision of the headline success, rest on single-run evidence. Revision: report Wilson intervals on the headline success rates (cheap); add at least 2–3 seed repeats for the top two low-jerk candidates (slot/type vs temporal MLP) to separate the jerk ordering from seed luck; the deferred frozen all-panel evaluation on a clip subset would bound the single-panel effect.
- [Major] The development-set size is internally inconsistent. Section 5.1 states the development corpus is "150 clips … 19,200 variants," and the refinement audit and Section 7 are framed on 150 clips, but Appendix A.4 states the 150 requested development identities "expand to 286 retained identities when existing mirrored partners are included," and Section 3.4 says the 150 development clips are reserved "together with their mirror partners." Consequence: if the actual manifest holds 286 identities, the denominators of the 19,200-reference audit (Table 10, Figure 7) and of the interaction analysis (class counts in Section 7) are misstated by ~1.9×; if 150 is correct, A.4 must say what the 286 refers to. Revision: state the authoritative development-manifest identity count once and align Sections 3.4, 5.1, 6.4, 7, Table 10, and A.4.
- [Major] The 4/32 unseen-shape failures are reported under a "gender-specific asset artifact" hypothesis that the paper concedes was not confirmed by direct asset inspection. Consequence: 12.5% of held-out shapes fail severely (one at 0.1% success), and the 28/32 generalization claim inherits an unverified cause; if the artifact is a collision-proxy or mass-estimation quirk, similar latent issues could exist in other (gender, beta) combinations, including among the 28 clean shapes. Revision: a short appendix table with total mass, collision-proxy count/volume, and joint-limit ranges for the 8 involved shapes (4 failing + 4 counterparts) would confirm or refute the artifact at low cost.
- [Major] The split is not subject- or source-recording-disjoint, so "untouched test" means untouched by clip identity only. The paper discloses this, but 99.14% is the main held-out-motion evidence, and different segments of the same recording in train and test share subject-level kinematics; the unseen-shapes test does not compensate, since 932 of its 1,024 clips come from training. Consequence: the test number may overestimate motion generalization. Revision: a metadata audit of the fraction of test/validation clips sharing an AMASS subject or source recording with training clips, reported next to Table 8.
- [Major] Section 7's primary inference is still row-level, and the semantic labels are unvalidated. The reported Levene p-values are computed on pooled per-clip-demeaned residuals (all clip × shape rows) — the same pseudo-replication structure the authors correctly identified in the CV analysis; the power discrepancy (p = 1.1 × 10⁻⁷ with overlapping clip-level CIs) is explained, but the primary test itself is not clip-level. Keyword-matched captions may misclassify clips, and the analysis covers only the 62 of 150 development clips in the compared classes, with zero failed rollouts. Consequence: the properly negative result is appropriately scoped, but its power and label fidelity are weaker than presented. Revision: report a clip-level permutation p-value for the variance comparison and a small manually checked sample of semantic labels with agreement.
- [Minor] Unit inconsistency in Section 6.5.1: the 28 clean shapes "succeed at 99.75% on the 932 train-split pairs" but "99.61% on the 2,576 validation/test-split pairs"; since 92 clips × 28 shapes = 2,576 pairs, the corresponding train figure should be 932 × 28 = 26,096 pairs. As written, the train figure counts clips where the validation/test figure counts pairs. Revision: state the pair (or rollout) count explicitly for both populations.
- [Minor] Protocol comparability and provenance gaps: all 1,048 test clips are 6.6 s and reach no horizon cap, but the fraction of validation clips that do reach the cap is not stated; the full-scale training seed is "recorded in saved run configuration" rather than given; the failed initial launch cost is "not separately quantifiable." Consequence: the validation–test comparison is not exactly like-for-like on horizon, and two provenance facts a reader would want are absent from the text. Revision: report the validation cap fraction, the training seed, and, if recoverable, the failed-launch estimate.
- [Minor] Corpus-level caveats that bound the "text-associated" contribution: 40% of mirrored captions contain left/right or clockwise terms that mirroring inverts; caption semantic fidelity is not audited; the 1,508-clip text filter is a heuristic (one retained clip, the sparring stick, clearly required missing support); generated references are never quantitatively compared against the source mocap, so HUMOS generation error sits upstream of every tracking number; and licensing of the 18 AMASS collections is deferred. Each is disclosed or scoped as future work, so the consequence is limited to the corpus contribution rather than the controller. Revision: a small quantitative fidelity check of generated versus source motion on a clip sample, and completion of the licensing audit before release.


Questions for the Authors

1. What is the exact number of clip identities in the development manifest — 150 or 286 (Section 5.1 vs Appendix A.4) — and which count do the 19,200-variant audit (Table 10, Figure 7) and the Section 7 analysis actually use?
2. What fraction of the 1,048 test (and validation) clips share an AMASS subject or source recording with a training clip?
3. For the four flagged unseen shapes, can you report asset-level diagnostics (total mass, collision-proxy count/volume, joint-limit ranges) for each failing (gender, beta) and its other-gender counterpart, to confirm or refute the artifact hypothesis?
4. In Section 7, what is a clip-level (e.g., permutation) p-value for the mass-normalized power variance comparison, and how were the keyword-matched semantic labels validated (sample size, agreement)?
5. What fraction of the 1,048 validation clips reach the 600-step horizon cap, and what is the full-scale training seed?

Beck:

Abstract

Remove “Source captions remain associated metadata rather than controller inputs”

Perhaps rephrase to: “We construct a multi-morphology motion corpus from AMASS-derived HumanML3D clips using HUMOS, pairing shape-specific reference motions and associated captions with matching SMPL-based simulation assets.” Or similar

 

Consider removing “these are trained-body, one-shape-per-clip results, not all-shape or unseen-body tests.”

 

1. Introduction

Split up the second sentence of the first paragraph:

Changes in limb lengths, mass distribution, and collision geometry alter both the reference trajectory and the dynamics required to follow it. Learning across body shapes requires a consistent relationship between source motion, shape-specific reference, and simulated body.

 

Move the list of contributions to below the second paragraph

“…not a claim that morphology conditioning or attention is new. \n This paper makes four contributions: …”

 

Replace the last paragraph “The selected checkpoint…” with an overview of sections.

 

2. Related Work

Consider removing or reducing size of Table 1.

 

3. Multi-Morphology Dataset

Move Figure 1 to below the first paragraph

 

3.2. Body Sampling

“we sample x using NumPy’s `default_rng(46)`”  à  “We sample x using NumPy’s random number generator with a seed of 46.” Alternatively, move this specific detail to the appendix with reproduction details. How relevant is the seed number?

 

This line may not be required or could be better placed in the introduction or discussion “…HUMOS is an existing identity-conditioned model; neither it’s training nor a text-to-motion generator is a contribution of this work”

 

3.4. Packaging, scale…

“We use the versioned hhi_stage2_v1 split…”  - specific detail may be more suited to Appendix, unsure

 

Table 2 Caption: “Current corpus and split sizes.”

 

4.1. Task, Simulation, and actuation

Ensure “PD” is defined before abbreviation is used

 

4.2. Observation Design

Move Table 3 to the end of 4.2

Table 3 caption: “Observation inputs supplied to both actor and critic, where B = 24 denotes the number of simulated bodies.”

If using this caption, remove “Let B = 24 denote the number of simulated bodies” from main body

 

4.3. Temporal Slot/Type Annotation

Define PPO as Proximal Policy Optimisation at it’s mention in the final paragraph before using the abbreviation in the next subtitle.

 

5.5. Compute Accounting

Consider removing or rephrasing and shortening the discussion of GPU resource. It may be worth a mention but is not a primary contribution of the paper

 

Table 7 Caption: “Recorded full-scale monitoring snapshot. No confidence intervals or all-shape coverage are implied”.

Table 8 Caption: “Offline evaluation of the epoch-28170 checkpoint on the 1048-clip validation and test splits.”

Figure 5: Combine the two captions: “Failure analysis taxonomy for the held-out evaluation, using per-motion records and rendered rollouts” or similar

 

7. Physical Analysis

This section should be shortened and merged as a subsection of 6

E.g. “This section examines how body morphology changes the physical control demand of a motion, questioning whether the effect depends on the motion being performed”…

 

Consider mentioning how clips are grouped and sorted into classes via their text descriptions earlier in the paper. I believe this is the first mention of how the captions are used!