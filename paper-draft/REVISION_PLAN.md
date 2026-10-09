# Structural revision and claim–evidence map

Working revision: 6 October 2026. Changes are confined to `paper-draft/`.
GPU evaluation is deferred. No pending result is assumed.

## Central argument

A shared controller can track a large paired corpus of morphology-specific
motions across diverse simulated bodies. The paper establishes the system's
scale, measured tracking, and substantial but non-uniform unseen-body transfer.
It does not yet establish uniform held-out all-body coverage or the causal
benefit of explicit morphology inputs.

## Claim–evidence map

| Claim | Available evidence | Defensible wording / remaining gap |
|---|---|---|
| Large paired motion/body construction | 20,951 clip identities × 128 configurations; HUMOS references, matching SMPLSim assets, deterministic splits | Derived paired corpus; variants are not independent captures. Resource release requires provenance/licensing audits. |
| One shared policy tracks held-out motions | Original epoch 28,170: validation 98.47%, test 99.14%; 1,048 pairs per split | Strong tracking on selected trained-body pairs under the specified threshold. No all-body or source-recording-disjoint claim. |
| Broad trained-body coverage | 150 training clips × 128 bodies, physical replay with zero recorded failed episodes | Coverage of the development subset. Held-out all-body consistency remains pending. |
| Unseen-body transfer | Saved 32,768 replay rows: 30,027 successes, 91.6351% overall; 28-body subgroup 99.7419%; four-body subgroup 34.8877% | Substantial, non-uniform transfer in the sampled coefficient range. Include all 32 in the headline. |
| Four bodies fail because of bad assets | Opposite-template contrasts; static mass differences in three cases; fourth lacks that pattern | Cause unresolved. Template changes affect geometry/dynamics; symmetry in beta coefficients does not prove a defect. |
| Slot/type attention is a practical choice | Three refined-reference seeds per architecture; best mean success for C, best errors for D, lowest mean jerk for A | Retain C as a compromise. No universal dominance or consistent factor-of-two jerk advantage. Reruns support an existing choice. |
| Refinement improves reference contact quality | 19,200 paired references: foot speed 0.055→0.038 m/s; below-floor frames 39.6→3.3%; small pose/root changes | Artifact reduction with a jerk cost (45.2→60.7 m/s³). Not a causal policy-learning benefit or proof of no cost. |
| Dynamic motions are more/less shape-sensitive | Recorded residual spreads and clip bootstrap intervals; historical pooled-row tests | Descriptive only pending a direct clip-level contrast. Interval overlap is not a test of a difference. |
| Failure families explain mechanical causes | Caption joins of 25 failed trained-body pairs | Descriptive categories; class denominators and dynamic interventions needed for enrichment/causal claims. |
| Explicit morphology conditioning causes transfer | Model includes morphology inputs; unseen-body success | Mechanism not isolated: state/reference also encode shape. Add interventions or a matched training control for that claim. |
| Full-data polishing improves transfer | Candidate checkpoint exists; additional training includes all original motion splits | Pending paired replay. It will test unseen bodies on known motions, not untouched motion generalization. |

## Revised outline implemented

1. Introduction: problem, shared-control claim, three contributions, principal limits.
2. Related work: large-scale tracking versus morphology-conditioned control.
3. Problem formulation and system overview: shared policy and paired body/reference identity.
4. Paired motion and body corpus: construction, refinement, scale, splits.
5. Shared controller and training: observations, attention, PPO, streaming.
6. Evaluation protocol: exposure populations, seed aggregation, asset matching, metrics.
7. Results, ordered by question:
   - held-out motions on trained bodies;
   - trained-body consistency and missing held-out panel;
   - unseen-body transfer including all four severe cases;
   - pending full-data comparison;
   - architecture trade-offs over three seeds;
   - reference-quality trade-offs;
   - descriptive failure patterns.
8. Exploratory physical analysis: secondary question with inference limitations.
9. Discussion and limitations.
10. Conclusion.
11. Appendices: reproducibility, monitoring snapshot, original seed-0 grid, compute history.

## Minimum remaining evidence, ordered by purpose

1. **Unseen-body checkpoint comparison (GPU pending).** Freeze epoch 34,000;
   reuse original 1,024 × 32 references/assets and horizons. Report overall
   and per-body scores, recoveries/regressions; add matched trained-body control.
   See `../note/README.unseen-morphology-fulldata-evaluation.md`.
2. **Held-out all-body panel (GPU pending).** Choose a frozen, affordable subset
   of held-out clips, cover all 128 trained bodies, retain paired records,
   and report per-clip worst-body errors and all-body success fractions.
3. **Inference repair for physical analysis.** Use a direct contrast resampled
   at the clip level; check class construction, normalization, and units.
   Does not require new policy training, but source instrumentation must be audited.
4. **Qualitative and failure verification.** Same-clip multi-body reference
   overlays; recovered/persistent unseen failures; first-crossing traces;
   caption-family denominators. Do not replace these with unrelated parallel scenes.

Optional stronger claims require additional controls: correct/neutral/shuffled
morphology-input replay plus a matched policy trained without explicit inputs;
or common-reference replay for refined/unrefined checkpoints. These are not
prerequisites for the narrower system claims in this revision.

## Evidence provenance and validation limits

- Unseen population numbers recalculated directly from
  `data_cache/unseen_morphology_generalization.joined.csv`, including failures.
- Repeated-seed summaries use the verified values from the prior local log
  review; seed-0 values are inherited from the historical paper table. That
  review could not independently recover seed-0 logs. The source grid and
  repeated-seed uncertainty conventions are distinguished in the manuscript.
- Monitoring epoch 28,159 is moved to an appendix; original final results use
  epoch 28,170. The full-data candidate is a separate checkpoint.
- Existing `main.pdf` predates this source revision unless a new build is
  explicitly reported. Pending evidence markers are for the working draft.
