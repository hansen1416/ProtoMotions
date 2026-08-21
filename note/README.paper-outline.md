# Paper Outline (current, 2026-08-20)

Supersedes `README.paper.md` (Kimodo-era, archived — that pipeline was dropped 2026-06-22) and
`README.prelim-report.md` / `README.eval-plan.md` (1024-clip pilot, June 27 — kept as historical
reference, not this outline). This file tracks section structure and framing decisions only;
results/figures get filled in as runs finish.

## 1. Abstract

## 2. Introduction

Motivate a single physics policy that imitates arbitrary human motions across a wide range of
body shapes (128 SMPL betas, 26-144 kg, 1.13-1.67 m), rather than the fixed-morphology setting
of prior physics-based trackers. State the contribution: a two-stage curriculum
(neutral pretrain -> shape transfer) that decouples motion-OOD from shape-OOD, run at full
HumanML3D scale (20,946 clips).

## 3. Related Work

- Physics-based motion tracking (PHC, PULSE, ExBody2, GMT, H2O-style trackers) — all
  fixed-morphology.
- Morphology-conditioned RL (HUMOS, MorFiC).
- MoE in actor-critic RL.

## 4. Method

### 4.1 Problem Formulation
Task definition, observation/action space, morphology conditioning (`morphology_obs`:
gender + betas, or physics-derived features — see 4.6).

### 4.2 Two-Stage Curriculum
Stage 1 (neutral pretrain, β=0, full 20,946-clip library) -> Stage 2 (shape transfer, 128
shapes). Why decoupled: separates motion-OOD (does the policy know the *motion*) from
shape-OOD (does it know the *body*).

### 4.3 Architecture
Presents the **final** architecture used for the full-scale run: wide-MLP trunk (actor
2896×6, critic 1024×4), morphology-conditioned. Note up front that this was reached via
iterative trial against two alternatives (mixture-of-experts, self-attention temporal encoder)
that are described and ablated in **5.3**, not here — Method states the answer, Experiments
narrates how we got there.

### 4.4 Reward and Termination Design
The "discover" relaxation: drop tracking-error termination (fall-only) and the
effort/smoothness/contact-timing reward terms, keep only tracking-shaping rewards.
CMU getting-up-policies-style two-phase curriculum motivation — relax every constraint that
isn't "did you get anywhere near the reference" so the policy's only job is finding *any*
successful trajectory through the hardest clips.

### 4.5 Training Infrastructure at Scale
`GlobalClipPool`: streams per-clip motion files from R2 instead of loading one static file,
resident training pool + fixed eval holdout (immune to resident-set churn) + weight floor
(solved clips keep nonzero rehearsal priority) + random rehearsal fraction.

### 4.6 Morphology Conditioning Representation
Final representation used at full scale: raw SMPL betas + gender, concatenated as
`morphology_obs`. Note, as in 4.3, that this was chosen after comparing against a
physics-derived feature representation (mass, height, limb-length-style scalars, z-scored,
~15-dim) — comparison and result in **5.4**.

## 5. Experiments

### 5.1 Setup
Dataset (20,946 clips × 128 shapes), metrics (success rate, gt/gr error, holdout eval),
baselines.

### 5.2 Main Results
Stage 1 neutral pretrain + Stage 2 shape-transfer success rates.

### 5.3 Architecture Evolution — Flat MLP → MoE → Self-Attention
**Framing decision (2026-08-20, user-confirmed):** presented as a narrative of iterative design,
not a flat ablation table — this is literally the order the architectures were tried, each one
motivated by the failure mode of the last, all validated at reduced scale (150-clip / 1024-clip
corpora) rather than the full 20,946×128 dataset, for cost reasons (see below).

1. **Flat-MLP baseline.** Wide MLP over flat-concatenated observations (including history/lookahead
   frames concatenated, not tokenized). Establishes the ~75-80% success-rate ceiling on the
   hardest-clip subset that motivates everything that follows.
2. **Mixture-of-Experts (MoE).** Hypothesis: different motion classes (crawl/kneel/squat/turning)
   need different specialist sub-policies, so a routed MoE actor/critic should beat a flat trunk of
   equal total capacity. Result: **null** — a capacity-matched wide-MLP control caught up to the
   MoE-stable variant on all metrics; "does routing matter" reopened, not resolved.
3. **Self-attention temporal encoder.** Hypothesis: the flat-concat trunk can't easily learn
   structure across the dilated history/lookahead frame sequence, so tokenizing per-frame and
   attending (reusing MaskedMimic's existing `Transformer` module, zero new infrastructure) should
   help. Result: **null** — no separation from flat-concat at matched steps.

**Why reduced scale only:** full-scale training runs are expensive enough that exhaustively
re-testing every architecture lever at full scale isn't viable, so architecture and reward/
termination choices were validated at reduced scale first and then carried forward as a bundled
recipe into the one full-scale run. This is presented as the actual decision-making process, not
reconstructed after the fact — the same logic used in practice to choose the Stage 2 full-scale
config (§ discussed in `note/README.note.md`, `mlp_wide_stage2_discover_attention.py`'s
docstring). Both MoE and attention were null/inconclusive results at the scale tested — reported
honestly as "we checked, it didn't clearly help, so we kept the simpler/already-validated flat-MLP
choice under the assumption it doesn't get worse at scale."

### 5.4 Morphology Conditioning Ablation — Raw Betas vs. Physics-Derived Features
Pilot-scale comparison (1024-clip × 128-shape corpus): `hhi_1024_transfer` (raw SMPL betas,
21,400 epochs) vs. `hhi_phy_1024_transfer` (15-dim z-scored physics-derived features — mass,
height, limb-length-style scalars — 17,200 epochs). T1 persistent-failure analysis: 177 vs. 167
failing clips, **not statistically significant**; qualitative visual review favored physics
features (5/8 vs. 0/8 clips judged visually improved). Same cost-sensitive framing as 5.3: run
once at reduced scale, result inconclusive-but-suggestive, raw betas kept as the simpler default
for the full-scale run since the physics-feature signal wasn't strong enough to justify the extra
feature-engineering surface at 20,946×128 scale.

### 5.5 Reward/Termination Ablation
Discover-relaxed vs. stricter (`mlp_wide.py`-style) config — also validated at reduced scale
for the same cost reason as 5.3/5.4, then carried forward as-is to full scale by explicit decision
rather than reverting.

### 5.6 Failure Analysis
Persistent-failure motion clusters; overlap across single-shape and multi-shape settings
(97.1% overlap between full-scale-scratch hardest-clip scoreboard and single-shape neutral
persistent-failure set — the plateau is an intrinsic motion-difficulty ceiling, not shape- or
reward-specific).

### 5.7 Cross-Shape Generalization
Held-out beta interpolation/extrapolation (E7). **Blocked** on the `infer.py` held-out-beta
pipeline bug in the humos repo — deferred in favor of reusing already-existing held-out-shape
data in R2 once a good Stage 2 checkpoint exists.

## 6. Discussion / Limitations

Capability ceiling on the hard-motion cluster (persists across shape, reward-relaxation,
architecture, and morphology-representation interventions). What didn't work and why: FiLM
conditioning, residual adapters (v1-v6, plateaued 78-82%), MoE routing (no separation from
capacity-matched control), self-attention temporal encoding (no separation from flat-concat, plus
no positional encoding in the shared `Transformer` class — flagged, deliberately not fixed, judged
unlikely to move the ceiling), physics-derived morphology features (no statistically significant
gain over raw betas, though qualitatively suggestive). Broader lesson from the design-evolution
narrative (5.3/5.4): the hard-motion ceiling looks like it's about *motion difficulty*, not about
which architecture or which conditioning representation carries the signal — every lever tried
moved metrics by less than noise. Resource constraints on ablation breadth addressed directly in
5.3/5.4's framing rather than hidden.

## 7. Conclusion
