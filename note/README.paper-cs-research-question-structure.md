| Research question | Evidence and role |
|---|---|
| **RQ1. Can one shared controller learn to track diverse motions across different bodies?** | Held-out-motion tracking and performance across the 128 trained bodies establish the shared-learning capability. |
| **RQ2. How well does that capability generalise to unseen body shapes, and where does it fail?** | Inside/outside evaluations, per-shape results, physical-property groups, and checkpoint comparisons establish the extent and limits of transfer. |
| **RQ3. Which design choices support that capability, and what trade-offs do they introduce?** | Architecture comparisons, reference refinement, and morphology-conditioning controls explain the system’s behaviour. Claims about conditioning depend on the available ablation evidence. |

The central claim becomes:

> **A shared controller can learn morphology-specific motion tracking across diverse bodies, with strong average transfer to unseen shapes but uneven robustness across individual bodies.**

The structure would then be:

1. **Introduction:** introduce the learning problem, central claim, and three questions.
2. **Related Work:** position shared motion learning, morphology conditioning, and temporal architectures.
3. **Problem Formulation and System Overview:** define motion/body variation, matching references and assets, and the generalisation settings.
4. **Paired Corpus and Reference Construction:** explain how the data supports learning across bodies.
5. **Shared Controller and Training:** describe the policy and training choices.
6. **Evaluation Protocol:** map each experiment to a research question.
7. **Results:** organise explicitly as **RQ1 → RQ2 → RQ3**, with failures included in RQ2.
8. **Discussion and Limitations:** explain what we learned about morphology-aware learning and what remains unresolved.
9. **Conclusion:** answer the three questions concisely.

The other agreed changes still apply: move monitoring history and evaluator fixes to appendices; present architecture results as trade-offs; report every unseen shape; and keep physical analysis secondary unless it directly explains a research question.
