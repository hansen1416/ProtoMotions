# Paper Outline (current, 2026-08-24)

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

### 4.1 Data & Asset Pipeline
Three-repo pipeline: ProtoMotions (RL framework), a HUMOS fork (`sha2nkt/humos_website_backend`,
shape-conditioned motion diffusion, ECCV 2024) for shape-adapted motion targets, and SMPLSim for
per-shape body assets. 128 body shapes = 64 SMPL betas sampled uniformly on [-3,3] (seed 46) x 2
genders, each keyed by a deterministic hash. SMPLSim's `run.py` generates one MJCF per shape (real
inertial weight, simplified feet, frozen hands, mass-scaled PD gains); multi-shape support
(per-env asset assignment, per-shape motion indexing) built into `MotionLib`/
`isaacgym/simulator.py`/`env.py`. Frame-0 grounding computes each shape's lowest collision point
once (one IsaacGym env per shape, not per motion — ~64x cheaper) and shifts every motion's root
height so it starts just above the ground.

HUMOS solves a specific problem: naive AMASS retargeting keeps identical joint angles across every
body shape, so the reference motion is exactly shape-invariant and carries no shape-adaptation
signal. HUMOS resamples each clip via shape-conditioned diffusion instead. Pipeline: AMASS
(18 of 23 sub-datasets used) -> SMPL+H forward kinematics at 20fps -> mocap-artifact cleaning
(treadmill/skating clip removal) -> HumanML3D 263-dim feature extraction -> HUMOS diffusion
inference per (clip, shape) pair -> packaged into MotionLib `.pt` files (full pipeline documented
in `README.data-pipeline-chronological.md`). Known caveat, confirmed later (`README.note.md`
§65-67): HUMOS output is noisier/jitterier than ground-truth AMASS mocap — this becomes the
central constraint on the reward-design work in **5.6**.

### 4.2 Problem Formulation
Task definition, observation/action space, morphology conditioning (`morphology_obs`: gender +
betas, or physics-derived features — see 4.7). Observation vector evolution: the original design
(~1014-dim) concatenated full-body kinematic state, future reference poses, previous action, and
an 11-dim morphology vector (gender id + raw betas), all under one shared normalizer. A phase
variable (motion time / clip length, in [0,1]) was added to resolve temporal aliasing in periodic
motions (a squat and a kneel look identical at two points in their own cycle without it). Contact
bodies were extended from feet-only to feet+knees+wrists so `contact_match_rew` gives signal on
crawl/kneel clips. See **4.7** for the morphology-representation ablation itself.

### 4.3 Two-Stage Curriculum
Stage 1 (neutral pretrain, β=0, full 20,946-clip library) -> Stage 2 (shape transfer, 128
shapes). Conceived after finding the original held-out eval set accidentally conflated motion-OOD
and shape-OOD (an artifact of how the eval cache was built, not a deliberate split) — decoupling
lets each stage isolate one axis of generalization. Two structural fixes were needed to make the
transfer work at all: (1) residual PD control (`q_target = q_ref(t) + 0.3*tanh(action)` instead of
`q_neutral + scale*tanh(action)`), since without it Stage 2 fine-tuning produced 3-4x higher action
jerk than Stage 1; (2) a normalizer reset for the morphology observation, since Stage 1's
constant-β training drives that dimension's running variance toward zero, which then clamps
Stage 2's real per-shape betas to a near-binary ±5 signal — fixed with a one-off script that resets
mean/variance for just that slice. Stage 1 alone reached 84.9% success at epoch 20,200, with an
8.7% persistent-failure tail; at full 20,946-clip scale a **single-leg dynamic-balance** failure
category dominates that tail for the first time (not visible in the 1,024-clip pilot) — the direct
transfer attempt from this checkpoint failed outright, later diagnosed as targeting the wrong
failure mode (the residual-PD fix addresses floor-contact jerk, not single-leg balance).

### 4.4 Architecture
Presents the **final** architecture used for the full-scale run: wide-MLP trunk (actor
2896×6, critic 1024×4), morphology-conditioned. Note up front that this was reached via
iterative trial against two alternatives (mixture-of-experts, self-attention temporal encoder)
that are described and ablated in **5.4**, not here — Method states the answer, Experiments
narrates how we got there. The MoE alternative was itself motivated by a specific diagnosis: going
from the 1,024-clip pilot (128 shapes) to the full 20,946-clip Stage 1 (1 shape) *decreased* shape
diversity yet success still dropped (0.84 -> 0.82-0.85 with a new persistent-failure tail),
pointing at gradient interference between dissimilar motion types rather than a capacity limit —
`mlp_wide.py`, the flat-MLP control, was built specifically parameter-matched (~95%) to the MoE
expert stack so the ablation could isolate "does routing help" from "does more capacity help."

### 4.5 Reward and Termination Design
The "discover" relaxation: drop tracking-error termination (fall-only) and the
effort/smoothness/contact-timing reward terms, keep only tracking-shaping rewards.
CMU getting-up-policies-style two-phase curriculum motivation — relax every constraint that
isn't "did you get anywhere near the reference" so the policy's only job is finding *any*
successful trajectory through the hardest clips. This followed four earlier, independent attempts
at the same hard-clip plateau (mass-scaled PD gains, per-body-segment mass-scaled gains,
soft-tracking termination/reward relaxation, an AMP-style discriminator reward) that all converged
to the same 40-65% success band at matched early epochs with no lever beating the others — notably,
AMP cut action jerk 5-7x but left success rate flat-to-worse, which ruled out control-jerk/
instability as the actual bottleneck and motivated a literature search that surfaced the
discover-then-refine framing. The resulting "discover" config reached 75-78% success, the largest
single gain in the 150-clip-lineage, with near-zero fall-terminations (failures are now almost
entirely `gt_error`-threshold breaches, not falls) — it is the direct ancestor of the
attention/DOF-reward/source-switching reward-design lineage in **5.6**.

### 4.6 Training Infrastructure at Scale
`GlobalClipPool`: streams per-clip motion files from R2 instead of loading one static file,
resident training pool (per-rank deterministic clip vocabulary, UCB-style priority refresh) +
fixed eval holdout (immune to resident-set churn) + weight floor (solved clips keep nonzero
rehearsal priority) + random rehearsal fraction. First full-scale launch hit an `EMFILE`
(too-many-open-files) crash from ~256 concurrent `rclone` fetches per rank — fixed before the run
that reached the results reported in **5.3**.

### 4.7 Morphology Conditioning Representation
Final representation used at full scale: raw SMPL betas + gender, concatenated as
`morphology_obs`. Note, as in 4.4, that this was chosen after comparing against three
alternatives (comparison and result in **5.5**): FiLM-style multiplicative conditioning (**failed
outright**, ~0.40-0.45 vs. ~0.84 baseline — traced to a 12,288-output conditioner bottleneck and
multiplicative-gradient instability), a learned shape-embedding concatenated in place of raw betas
(neutral, no measurable gain), and a 15-dim z-scored physics-derived feature vector (mass, height,
limb-length-style scalars).

## 5. Experiments

### 5.1 Setup
Dataset scale evolved through the project rather than being fixed upfront: 1,024-clip x 128-shape
pilot -> Stage 1 full 20,946-clip library at a single (neutral) shape -> a 20,946x2-shape
intermediate set (population-median male/female bodies, used to stress-test MoE without waiting on
full 128-shape data) -> a 150-clip x 128-shape reduced-scale testbed adopted for cost reasons as
the standard architecture/reward ablation corpus (5.4-5.6) -> full 20,946x128-shape training via
`GlobalClipPool` (4.6) for the final run. Metrics: `dp_error` (DOF-angle, shape-invariant — the
primary metric, see **5.2**), `gt_error`/`gr_error` (world-space position/rotation, scale-confounded
across shapes, kept for legacy/reward-design tracking), success rate, holdout eval.

### 5.2 Core Evaluation: Shape-Invariant Skill Consistency
**Framing decision (2026-08-24, user-confirmed):** success rate against a single-body external
baseline (PHC/PULSE/ExBody2/GMT) cannot demonstrate this paper's actual claim — that one policy
learns the *same motion* across a wide body-shape distribution — since those methods only ever
evaluate one body. World-space error (`gt_error`, meters) is not a fair axis even for *internal*
cross-shape comparison: a taller body produces larger absolute position deviations for the same
relative motor error, so raw meters bake in a body-scale confound before measuring skill at all.
DOF-angle-space error (`dp_error`, radians) is exactly shape-invariant by construction (same joint
target regardless of body) and is therefore the primary lens for the paper's core claim, not a
secondary metric introduced for the reward-design work in 5.6.

**Scope note: `dp_error` measures joint rotation, not root trajectory.** `dof_pos` is the actuated
joint-angle vector (radians) local to each body's own kinematic tree — hinge/spherical joint values
relative to their parent link. It excludes the root free-joint (pelvis translation and orientation
in the world), which is tracked separately via `gt_error`/`gr_error` and is exactly the
shape-dependent signal we're deliberately excluding from the headline metric. So the shape-invariance
claim built on `dp_error` is scoped to **limb articulation** — "the same joint rotates the same way
regardless of body" — not to whole-body world-space behavior. A tall and a short body executing the
same clip will show identical `dp_error` trajectories even though their root paths differ in absolute
distance (shorter legs, shorter stride, same joint angles) — that is expected and outside what this
metric measures. If the paper also wants to claim root-level shape-adaptation (e.g., stride length
scaling correctly with leg length), that needs a separate, explicitly shape-normalized root metric
(e.g., root displacement normalized by height/leg length) — raw `gt_error` cannot support that claim
either, for the same body-scale-confound reason given above.

Three components:
1. **Cross-shape consistency per clip.** For each clip, `dp_error` spread (std/IQR) across the 128
   training shapes. A tight spread demonstrates shape-independent skill execution; a wide spread
   would undercut the claim.
2. **Shape-failure correlation.** Already computed across multiple runs (see 5.8): Pearson
   r ≈ -0.09 to 0.003 between shape extremity (mass/height) and failure — effectively zero
   relationship. Promoted here from a footnote in failure analysis to the headline evidence that
   body shape does not predict success — i.e. the same motion is being learned regardless of body.
3. **Held-out shape generalization (5.9, E7).** Extends the consistency claim past the 128 training
   shapes to interpolated/extrapolated unseen shapes — the strongest version of this evaluation,
   currently blocked on the `infer.py` pipeline bug.

External comparison (PHC-style neutral-shape success rate, β=0) is retained only as a minor
sanity-check footnote — confirming neutral-shape quality wasn't sacrificed for shape generalization
— not as a headline result, since a single-body number structurally cannot speak to the
shape-generalization claim this section exists to make.

### 5.3 Main Results
Stage 1 neutral pretrain: 84.9% success at epoch 20,200 (~174h, 6 GPUs), 8.7% persistent-failure
tail (4.3). Stage 2 shape-transfer: the residual-adapter lineage (frozen backbone + adapter, v1-v6)
plateaued at 78-82% success across every variant tried and was abandoned in favor of a from-scratch
Stage 2 trunk, still in progress — full-scale Stage 2 success numbers are not yet final. Report
alongside the 5.2 shape-consistency metrics, not success rate alone.

### 5.4 Architecture Evolution — Flat MLP → MoE → Self-Attention
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

### 5.5 Morphology Conditioning Ablation — Raw Betas vs. Physics-Derived Features
Pilot-scale comparison (1024-clip × 128-shape corpus): `hhi_1024_transfer` (raw SMPL betas,
21,400 epochs) vs. `hhi_phy_1024_transfer` (15-dim z-scored physics-derived features — mass,
height, limb-length-style scalars — 17,200 epochs). T1 persistent-failure analysis: 177 vs. 167
failing clips, **not statistically significant**; qualitative visual review favored physics
features (5/8 vs. 0/8 clips judged visually improved). Same cost-sensitive framing as 5.4: run
once at reduced scale, result inconclusive-but-suggestive, raw betas kept as the simpler default
for the full-scale run since the physics-feature signal wasn't strong enough to justify the extra
feature-engineering surface at 20,946×128 scale.

### 5.6 AMASS/HUMOS Reward Source Design — **in progress, not yet resolved**
Motivation: AMASS-canonical DOF-space targets are exactly shape-invariant, so training on them
alone gives zero shape-conditioning gradient; HUMOS is the only channel whose target differs by
shape. Iterative design, same framing as 5.4/5.5: (1) DOF-space-only bundle on canonical corpus
(`README.note.md` §65-66) — clean `dp_error` convergence, world-space metrics flat; (2)
episode-level source-switching (§67, mask reward by AMASS-vs-HUMOS per episode) — world-space
improves over (1) but underperforms the unmasked world-space baseline at matched steps (63% vs.
83% success), likely because masking halves how often world-space reward fires; (3) combined
unmasked full-factor reward on the mixed corpus, retuned weights — outcome pending.

### 5.7 Reward/Termination Ablation
Discover-relaxed vs. stricter (`mlp_wide.py`-style) config — also validated at reduced scale
for the same cost reason as 5.4/5.5, then carried forward as-is to full scale by explicit decision
rather than reverting.

### 5.8 Failure Analysis
Persistent-failure motion clusters; overlap across single-shape and multi-shape settings
(97.1% overlap between full-scale-scratch hardest-clip scoreboard and single-shape neutral
persistent-failure set — the plateau is an intrinsic motion-difficulty ceiling, not shape- or
reward-specific). Shape-failure correlation (Pearson r ≈ -0.09 to 0.003) reported here in full
detail; headline framing of the same number lives in **5.2**.

### 5.9 Cross-Shape Generalization
Held-out beta interpolation/extrapolation (E7) — the strongest form of **5.2**'s shape-consistency
claim, extended past the 128 training shapes to unseen interpolated/extrapolated ones. **Blocked**
on the `infer.py` held-out-beta pipeline bug in the humos repo — deferred in favor of reusing
already-existing held-out-shape data in R2 once a good Stage 2 checkpoint exists.

## 6. Discussion / Limitations

Capability ceiling on the hard-motion cluster (persists across shape, reward-relaxation,
architecture, and morphology-representation interventions). What didn't work and why: FiLM
conditioning, residual adapters (v1-v6, plateaued 78-82%), MoE routing (no separation from
capacity-matched control), self-attention temporal encoding (no separation from flat-concat, plus
no positional encoding in the shared `Transformer` class — flagged, deliberately not fixed, judged
unlikely to move the ceiling), physics-derived morphology features (no statistically significant
gain over raw betas, though qualitatively suggestive). Broader lesson from the design-evolution
narrative (5.4/5.5): the hard-motion ceiling looks like it's about *motion difficulty*, not about
which architecture or which conditioning representation carries the signal — every lever tried
moved metrics by less than noise. Resource constraints on ablation breadth addressed directly in
5.4/5.5's framing rather than hidden. Shape-invariance evaluation (5.2) is a separate axis from this
ceiling discussion — the ceiling is about which *motions* are hard, not about *which bodies*
execute them, consistent with the near-zero shape-failure correlation.

## 7. Conclusion
