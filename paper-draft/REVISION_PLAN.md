# Connection Science revision — 10 October 2026

The editable manuscript is main.tex and sections/. review.tex is generated.
The current revision implements the agreed RQ1–RQ3 argument in paper-draft/.

## Claim–evidence map

| Question | Evidence | Claim boundary |
|---|---|---|
| RQ1: shared tracking | Epoch 28,170: 99.14% test success, 1,048 clips, one trained body per clip; development replay across 128 bodies | Strong selected-pair motion transfer and trained-body execution, not every held-out clip on every body |
| RQ2: unseen transfer | New 9 October report: 233 known clips × 128 inside and 128 outside bodies; 97.28% / 97.15% refined success | High average transfer, with worst-body success 3.0% / 13.3%; sampled parameter extrapolation, not arbitrary body robustness |
| RQ2: checkpoint sensitivity | Paired full-data deltas −0.37 / +0.42 pp; control −0.17 pp; 3 inside and 6 outside substantial regressions | Measured change between two checkpoints; duration and coverage confounded; rerun measures simulator variability only |
| RQ3: architecture | Three refined seeds per A–D; best average success C, lowest pose errors D, lowest mean jerk A | Practical trade-offs, no dominance; capacities differ; seed-0 original logs not independently rechecked |
| RQ3: reference refinement | 19,200 paired references: lower sliding/penetration, small pose changes, higher target jerk | Reference artifact reduction, not a causal policy-learning improvement |
| RQ3: conditioning | All compared models retain explicit shape inputs | Causal contribution remains untested |
| External positioning | Primary sources in CAPABILITY_SOURCES.md and cited headline table | Published capability comparison, no matched performance superiority |

## Implemented structure

Introduction → Related Work (capability table) → Problem → Paired Corpus →
Shared Controller → Evaluation Protocol → Results (RQ1, RQ2, RQ3) → Discussion → Conclusion.
Appendices: reproducibility, monitoring/evaluator history, exploratory physical
analysis, and all 256 new per-body rows. Old unseen-body results, figures,
interpretations and pending old-panel comparison are removed entirely from
this revised manuscript, including its generated review copy.

## Missing experiments, prioritised by claim

1. Fixed held-out-motion subset crossed with all 128 trained bodies: per-clip
   coverage, worst-body errors, and every-body success.
2. Matched explicit-conditioning removal: same body/reference panel, budget,
   and seeds. Mask/shuffle interventions supplement rather than replace it.
3. Common-reference policy comparison to separate refinement from target changes.
4. Dynamic first-crossing diagnostics and per-asset feasibility audit for severe
   unseen failures; successful counterparts and contact/root/actuation traces.
5. Matched external tracker replay (e.g. PHC) if performance superiority is sought.
6. Additional training seeds / checkpoint ladder for transfer uncertainty.

Physical-property groups and caption families remain descriptive; direct
clip-level physical contrasts and class denominators remain unresolved.
Release checksums, licensing and caption provenance need completion.

No additional GPU experiments were run in this revision. The new evaluation
is read from its source report, not inferred from the planning chat.
