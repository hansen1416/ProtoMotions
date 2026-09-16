# Reference Expansion Plan

Checked `/home/hlz/repos/LLMs-papers-survey` (an unrelated LLM/multimodal-generation survey, but
one of its categorized bib files — `notes/bib/t2hm.bib`, text-to-human-motion, 73 entries — does
overlap this paper's domain) against `paper/references.tex`'s current 19 hand-written `\bibitem`
entries, to find real gaps rather than pad the count.

**Scope check first**: grepped all six of the survey's bib files (`t2m`, `t2hm`, `t2i`, `t2v`,
`t2-3d`, `t2t`) for `physics|reinforcement learning|humanoid|morpholog|body shape|shape-condition`.
Only `t2hm.bib` has genuine hits — the "reinforcement learning" matches in `t2i.bib`/`t2v.bib`/
`t2t.bib` are all RLHF-for-generative-models (diffusion/LLM preference tuning), unrelated to
physics-based character control. So **`t2hm.bib` is the only useful source here**; no need to mine
the other five.

## Where the paper is actually thin

`paper/sections/03_related_work.tex` has three subsections:
- §3.1 Physics-based motion imitation — 7 refs (deepmimic, phc, pulse, omnih2o, exbody2, gmt,
  protomotions/isaacgym/ppo). Reasonably covered.
- §3.2 Morphology-conditioned control and retargeting — **only 2 refs** (won2019, smp2020). This
  is the paper's actual core topic and its thinnest section. **The survey repo does not help
  here** — it's a T2M/multimodal-generation corpus, not an RL/robotics one. This gap needs a
  separate literature search outside today's scope; flagging it so it isn't forgotten.
- §3.3 Text-motion data, shape-conditioned generation, and refinement — 3 refs (amass, humanml3d,
  humos), all already cited elsewhere too. This is where the survey repo actually helps: the
  section currently presents HumanML3D/HUMOS in isolation, with no sense of the surrounding
  landscape of text-motion datasets or generative approaches, which is exactly what a "too thin"
  reviewer complaint usually targets.

`paper/sections/04_dataset.tex` §4.1 ("Source motion and text provenance") also discusses caption
coverage/quality as an open limitation without citing any comparison point for what a text-motion
dataset's caption model normally looks like (BABEL's action labels vs. HumanML3D's free-text vs.
KIT's structured annotations are genuinely different designs, worth a sentence + citations here).

## Candidate references, by fit

Cross-checked each against `references.tex` for duplicates first — `mahmood_amass_2019` and
`guo_generating_2022` in `t2hm.bib` are the same papers as the existing `amass`/`humanml3d`
bibitems, so excluded below.

**Tier 1 — recommended, direct fit for §3.3 / §4.1 (dataset-landscape breadth):**

| Candidate | Why it fits | Target location |
|---|---|---|
| `lin_motion-x_2023` — Motion-X (NeurIPS 2023) | Large-scale expressive whole-body motion dataset; natural contrast point to HumanML3D's scale/expressiveness when discussing §4's "not a fully audited text-and-motion benchmark release" limitation | §3.3 and/or §4.1 |
| `punnakkal_babel_2021` — BABEL (CVPR 2021) | Action-label motion dataset; a genuinely different captioning design (discrete action labels vs. free text) — directly useful contrast for §4.1's caption-quality discussion | §4.1 |
| `plappert_kit_2016` — KIT Motion-Language Dataset (Big Data 2016) | Earliest text-motion dataset; establishes the field's starting point before HumanML3D | §3.3 |
| `petrovich_temos_2022` — TEMOS (ECCV 2022) | Generates SMPL body motions from text (not skeleton-only) — same body representation as this paper's pipeline, a closer generative precedent than most T2M work | §3.3 |
| `guo_tm2t_2022` — TM2T (ECCV 2022) | Same authors as HumanML3D; natural companion citation when introducing that dataset's lineage | §3.3 |
| `tevet_human_2023` — Human Motion Diffusion Model / MDM (ICLR 2023) | The foundational text-to-motion diffusion model; useful as one sentence contextualizing HUMOS among generative motion models generally, before the paper's existing "HUMOS is not a text-conditioned policy" disclaimer | §3.3 |

**Tier 2 — supports §3.2's kinematic-vs-physics distinction (still thin, but at least strengthens
the one sentence that's already there: "Kinematic retargeting and dynamic tracking address
different requirements"):**

| Candidate | Why it fits |
|---|---|
| `starke_deepphase_2022` — DeepPhase (SIGGRAPH 2022) | High-quality data-driven *kinematic* character control — good contrast example: solves motion quality, not shape/physics feasibility |
| `zhang_mode-adaptive_2018` — Mode-Adaptive Neural Networks (SIGGRAPH 2018) | Same category — data-driven kinematic control, no shape or physics dimension |

These don't fix §3.2's real thinness (still no additional RL/physics morphology-control papers),
but they do make the existing kinematic-vs-dynamic contrast sentence concrete instead of assertion-only.

**Tier 3 — optional, weaker fit, use only if a specific sentence needs it:**

| Candidate | Note |
|---|---|
| `zhang_primal_2025` — PRIMAL (arXiv 2025) | Title sounds physics-relevant ("Physically Reactive and Interactive Motor Model") but it's a kinematic autoregressive diffusion model in Unreal Engine — "physics effects emerge from training" is a soft claim, not an actual simulated controller. Could work as one Discussion-section nod to concurrent kinematic approaches claiming physical plausibility, but don't oversell the fit. |
| `zhang_motiondiffuse_2024` — MotionDiffuse (TPAMI 2024) | Another foundational T2M diffusion baseline, redundant with MDM/TEMOS if only one generative-model citation is wanted — include only if the sentence needs 2-3 examples rather than 1. |

**Not recommended**: the ~15 MotionGPT/LLM-as-motion-generator entries, retrieval (TMR/ReMoDiffuse),
editing (MotionFix/SimMotionEdit), and interaction-control (OmniControl/InterControl) papers in
`t2hm.bib`. These are real T2M subfields but this paper explicitly disclaims being a T2M
contribution ("not a text-to-motion generator... not a text-conditioned policy," §3.3) — citing a
dozen generation-method papers would read as padding, not landscape context, and risks inviting
"why didn't you compare against X" reviewer questions this paper isn't set up to answer.

## Mechanical execution plan (once you confirm which tier(s) to add)

1. `references.tex` is hand-maintained (`\begin{thebibliography}`, no `.bib` file — see the file's
   own header comment: "Primary papers and official software repositories; checked during
   drafting"). Keep that convention: write each new `\bibitem` by hand in the same style as the
   existing 19 (author list, title, venue/year, one link), don't introduce a `.bib`/BibTeX
   pipeline for just 6-9 new entries.
2. Full author lists and exact venue/page info for every Tier 1/2 candidate above are already
   pulled from `t2hm.bib` in this session (see the entries printed while researching this plan) —
   reuse those rather than re-deriving from scratch.
3. Pick citation keys following the existing convention (short, lowercase, no year suffix needed
   since none currently collide): `motionx`, `babel`, `kit`, `temos`, `tm2t`, `mdm`, `deepphase`,
   `modeadaptive`.
4. Insert `\cite{}` calls at specific sentences, not just append refs to the bibliography with no
   in-text anchor:
   - §3.3 opening ("AMASS consolidates... HumanML3D associates...") — add MDM/TEMOS/TM2T/KIT as a
     one-sentence landscape gesture before pivoting to "We inherit those source identities..."
   - §4.1's caption-coverage-limitations paragraph — cite BABEL (and optionally Motion-X) when
     noting that caption completeness/alignment is a known open problem across the field, not
     unique to this pipeline.
   - §3.2's "Kinematic retargeting and dynamic tracking address different requirements" sentence —
     cite DeepPhase/Mode-Adaptive as the kinematic side of that contrast.
5. Recompile (`pdflatex` from `paper/`) and check for undefined-reference warnings the same way
   the earlier figure-citation pass in this repo was verified.
6. §3.2's thinness (previously an open item, since the survey repo checked above doesn't cover
   RL/physics-control) — **resolved below via targeted web search**, see Part 2.

---

## Part 2 — RL / physics-based-control literature (web search, not the survey repo)

User-requested follow-up: find more RL papers beyond the already-cited PHC/AMP/ASE/ProtoMotions
lineage. Searched directly (not via the survey repo, which has nothing in this space — confirmed
in Part 1's scope check). All bibliographic details below were pulled from arXiv/ACM listings
during this session, not from memory — verify before writing final `\bibitem`s regardless.

**Important scope check done first**: is AMP/ASE actually used by this paper's trained models, or
just available in the framework? Checked `examples/experiments/mimic/mlp.py` (the experiment file
behind every checkpoint this paper reports) — its agent config targets plain `PPO`
(`protomotions/agents/ppo/agent.py`), **not** `AMP`/`ASE` (`protomotions/agents/amp/agent.py`,
`ase/agent.py` — confirmed those classes exist in the codebase but aren't in this paper's actual
training path). So AMP/ASE/CALM/MaskedMimic below are **landscape/lineage citations** (the
algorithmic family this codebase's agent hierarchy is structured after, per `CLAUDE.md`'s own
"PPO → AMP → ASE" diagram), not "this paper uses AMP" claims — phrase any new §3.1 text
accordingly, the same careful non-overclaiming style the rest of `references.tex`'s surrounding
prose already uses.

### §3.2 — morphology-conditioned RL control (the real fix for the previously-unresolved gap)

| Candidate | Details | Why it fits |
|---|---|---|
| **AdaptNet** | Pei Xu, Kaixiang Xie, Sheldon Andrews, Paul G. Kry, Michael Neff, Morgan McGuire, Ioannis Karamouzas, Victor Zordan. *AdaptNet: Policy Adaptation for Physics-Based Character Control.* SIGGRAPH Asia 2023 / ACM TOG. arXiv:2310.00239 | Physics-based character control policy adaptation across body-shape changes (and joint locks) — arguably a closer fit than `smp2020` for *continuous* shape variation specifically, since smp2020 is about topology/limb-count, not anthropometric variation. |
| **ModuMorph / Universal Morphology Control via Contextual Modulation** | Zheng Xiong, Jacob Beck, Shimon Whiteson. ICML 2023. arXiv:2302.11070 | Direct RL morphology-conditioning architecture (hypernetwork-generated per-morphology parameters + morphology-dependent attention) — the modern continuation of `smp2020`'s "one policy, many morphologies" framing. Strong §3.2 fit. |
| Distilling Morphology-Conditioned Hypernetworks for Efficient Universal Morphology Control | arXiv:2402.06570 | Follow-up to ModuMorph, same lineage. Optional — only add if the paragraph wants 2-3 examples of this sub-literature rather than 1. |
| Shared Modular Recurrence in Contextual MDPs for Universal Morphology Control | arXiv:2506.08630 | Most recent in the same lineage. Optional, same reasoning as above. |

Recommend: add AdaptNet + ModuMorph as the two new §3.2 citations (both direct, both well-attested,
both fill a real gap without padding); treat the other two as optional extras only if the
paragraph is rewritten to survey the sub-field rather than name 1-2 examples.

### §3.1 — physics-based motion imitation lineage (currently absent despite being the codebase's own algorithmic namesake)

| Candidate | Details |
|---|---|
| **AMP** | Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa. *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.* SIGGRAPH 2021 / ACM TOG. arXiv:2104.02180 |
| **ASE** | Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, Sanja Fidler. *ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters.* SIGGRAPH 2022 / ACM TOG. arXiv:2205.01906 |
| **CALM** | Chen Tessler, Yoni Kasten, Yunrong Guo, Shie Mannor, Gal Chechik, Xue Bin Peng. *CALM: Conditional Adversarial Latent Models for Directable Virtual Characters.* SIGGRAPH 2023. arXiv:2305.02195 |
| **MaskedMimic** | Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, Xue Bin Peng. *MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting.* SIGGRAPH Asia 2024 / ACM TOG. arXiv:2409.14393 |

All four correspond 1:1 to agent classes already present in this repo
(`protomotions/agents/{amp,ase,masked_mimic}/agent.py`) — citing them properly credits the
framework's own algorithmic lineage, which is currently uncited even though the class hierarchy is
directly named after these papers. Natural insertion point: §3.1, right after the DeepMimic/PHC
sentence, as "subsequent work added adversarial style priors (AMP), reusable skill embeddings
(ASE), directable latent conditioning (CALM), and unified inpainting-based control (MaskedMimic)"
— landscape sentence, not a claim that this paper's experiments use them.

### Discussion/§9 or §5 refinement context — physics-consistent reference-generation lineage

These matter less for the Related Work section and more because `note/README.physical-analysis-plan.md`'s
sibling doc, the offline-refinement plan (`/home/hlz/.claude/plans/shiny-mapping-thimble.md`),
already *names* PARC/ASAP/InterMimic/OmniTrack inline as the ruled-out "train-a-tracker-to-clean-the-
reference" pattern, with no citation at all. If that plan is ever executed and written up in the
paper (even just in Discussion, as "we considered and rejected the X pattern"), these need real
citations:

| Candidate | Details |
|---|---|
| **PARC** | *PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers.* arXiv:2505.04002. Iteratively trains a kinematic motion generator + RL tracker, uses the tracker to physically correct the generator's output — closest match to the ruled-out pattern's description. |
| **OmniTrack** | *OmniTrack: General Motion Tracking via Physics-Consistent Reference.* arXiv:2602.23832 (2026). Explicitly decouples physical feasibility from tracking via a privileged generalist stage — directly addresses the same floating/penetration artifact problem this paper's own HUMOS-refinement work is solving, just with a different (rejected-here) architecture. Worth a sentence in Discussion contrasting approaches even without adopting the method. |
| ASAP | He et al. (CMU/NVIDIA), RSS 2025, arXiv:2502.01143. Sim-to-real dynamics alignment via delta-action residual model. Weaker fit — real-robot deployment focus, not shape-conditioning — include only if the sim-to-real angle becomes relevant. |
| InterMimic | Sirui Xu, Hung Yu Ling, Yu-Xiong Wang, Liangyan Gui. CVPR 2025 Highlight. Teacher-student distillation for human-object interaction tracking. Weaker fit for this paper (object interaction, not morphology) — cite only if the teacher/student curriculum pattern itself gets discussed. |

### Updated mechanical execution note

Same convention as Part 1 (hand-written `\bibitem`, no `.bib` pipeline). Suggested new keys:
`amp`, `ase`, `calm`, `maskedmimic`, `adaptnet`, `modumorph`, `parc`, `omnitrack`. Priority order if
only doing a subset: **AMP, ASE, MaskedMimic, AdaptNet, ModuMorph** first (five citations, closes
both the §3.1 lineage gap and the §3.2 gap cleanly) — CALM/PARC/OmniTrack/ASAP/InterMimic are all
legitimate but lower-urgency additions.
