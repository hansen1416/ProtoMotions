# Paper Plan: Morphology-Generalized Physics-Based Motion Generation

**Target venue:** ICRA 2027
**Deadline:** ~September 2026
**Time remaining:** ~3 months

---

## Core Contribution

> "We present the first system for text-conditioned, morphology-aware physical motion generation: we extend a kinematic diffusion model (Kimodo) with body shape conditioning, then ground its outputs in physics via a morphology-conditioned imitation policy — enabling physically plausible motion generation across diverse human body shapes from text prompts."

The end-to-end capability: given a text description and a SMPL body shape (beta), the system produces a physically simulated motion tailored to that morphology. Neither component exists in isolation — Kimodo is kinematic and shape-agnostic; existing physics controllers assume a fixed body.

**Why this matters for robotics:**
- Deploy a single controller across humanoid robots with manufacturing variations
- Sim-to-real transfer where simulation morphology ≠ real robot morphology
- Digital humans and embodied AI with on-demand, shape-consistent motion

**What is novel (the combination, not the parts):**
1. **Kimodo + beta conditioning** — fine-tune the kinematic diffusion model to accept SMPL beta parameters as an additional conditioning signal, generating shape-appropriate kinematics from text
2. **Kinematic → physical grounding** — a morphology-conditioned imitation policy tracks Kimodo outputs in physics simulation, closing the gap Kimodo does not address
3. **Cross-shape generalization evaluation** — per-body-shape tracking error on held-out betas (interpolation and extrapolation), a metric not previously used for physics-based controllers

---

## Paper Structure

### 1. Introduction
- Gap 1: text-to-motion models (Kimodo, MDM) are kinematic and shape-agnostic — they cannot produce body-proportionate motion for diverse morphologies
- Gap 2: physics-based humanoid controllers (AMP, ASE, ProtoMotions) assume fixed morphology and require pre-existing motion clips
- Motivation: real applications (digital humans, humanoid robots, sim-to-real) need on-demand, text-driven, physically plausible motion across body shapes
- Contribution summary: beta-conditioned Kimodo + physics grounding + evaluation protocol

### 2. Related Work
- Physics-based motion imitation: PFNN, AMP, ASE, ProtoMotions
- Text-to-motion (kinematic): MDM, MotionDiffuse, Kimodo — none condition on body shape
- Morphology-robust control: evolution strategies, modular robots
- SMPL-based animation and HUMOS
- Morphology-conditioned policies: shape-conditioned MDP formulations

### 3. Method

#### 3.1 Problem Formulation
- Morphology-conditioned MDP — state includes body shape parameters (SMPL beta)
- Two-stage pipeline: kinematic generation → physical grounding

#### 3.2 Stage 1: Body-Shape-Conditioned Kimodo
- Baseline: Kimodo generates motions conditioned on text + keyframes for fixed skeletons (SOMA, G1, SMPLX) — no beta variation
- Extension: fine-tune Kimodo with a beta embedding injected as an additional conditioning signal (cross-attention or concatenation into the diffusion denoiser)
- Training data: 1024 text-annotated motion clips × 128 body shape variants (64 betas × 2 genders), produced by retargeting reference motions to each SMPL beta via HUMOS or shape-aware retargeting
- Output: kinematic motion sequences (joint positions, rotations, root trajectory) conditioned on (text, beta)

#### 3.3 Stage 2: Physics Grounding via Morphology-Conditioned Policy
- Kimodo output serves as the reference motion for a physics-based imitation policy
- Policy: MLP with morphology_obs (beta) concatenated to state input
- Morphology-matched simulation: each env instantiates the SMPL asset for its assigned beta, tracks the Kimodo-generated reference for that same beta
- At inference: text + beta → Kimodo → kinematic reference → physics policy → physically simulated motion

### 4. Experiments

The central claim: **text-conditioned physical motion generation that adapts to body morphology**. Experiments demonstrate both generation quality and physical meaningfulness.

#### 4.1 Kimodo beta-conditioning quality
- Given held-in betas, measure how well the extended Kimodo generates kinematically plausible shape-appropriate motion vs the baseline (no beta conditioning)
- Metric: joint position error against HUMOS-retargeted reference, foot-skating, naturalness

#### 4.2 Per-shape physics tracking performance (128 training betas)
- Tool: `evaluate_hhi_faults.py` + `HHIFaultEvaluator`
- Metric: per-(gender, beta_key) mean/max body distance
- Analysis: distribution of per-shape performance, correlation with beta L2 norm (shape extremity)
- Expected result: uniform performance across shapes, mild degradation at extremes

#### 4.3 Generalization to held-out shapes
- **Interpolation:** ~16–32 new betas sampled from [-3, 3] (different seed from training 128)
- **Extrapolation:** betas scaled to [-5, 5] range
- Metric: body distance degradation pattern across unseen shapes
- Note: SMPL motions at ±5 may be noisier — account for this in analysis

#### 4.4 Shape-adaptive physical analysis (key differentiator)

These experiments show the policy has learned physically meaningful shape-conditioned behaviour, not just blind reference tracking.

**4.4a Shape-conditioned torque and energy analysis**
- For the same text prompt, record joint torques and total energy expenditure across different body shapes
- If the policy is genuinely shape-aware: heavier bodies → higher torques, shorter limbs → different timing

**4.4b Stability analysis across shapes**
- Track COM trajectory and support polygon coverage during locomotion across shapes
- Extreme body proportions (very tall, very heavy) should show measurable but bounded stability degradation

**4.4c Contact quality across shapes**
- Foot contact patterns (timing, forces, duration) for the same motion across different shapes
- Does the policy adapt contact timing to body proportions?

**4.4d Shape extremity vs physical failure correlation**
- Plot body tracking distance against beta L2 norm AND against specific physical features (estimated limb length ratio, mass)
- Explains *why* extreme shapes are harder in physical terms

#### 4.5 Qualitative visualization
- Same text prompt executed simultaneously by multiple body shapes side-by-side
- Demonstrates the end-to-end pipeline: text → diverse physically simulated morphologies
- Visually compelling for ICRA reviewers and video submission

### 5. Conclusion
- First system combining text-conditioned kinematic generation with physics grounding across diverse body shapes
- Policy learns shape-adaptive physical strategies — not just reference tracking, but physically consistent control
- 1024-clip pilot validates the approach; scaling to full motion library and more body shapes is straightforward (future work)
- Potential extensions: sim-to-real transfer with morphology mismatch, SMPLX faces/hands, non-humanoid morphologies

---

## Experiment Status

| Experiment | Status |
|---|---|
| Architecture search (MLP, FiLM, ShapeEmbed, physics features — 4 runs) | **Done** |
| Baseline pilot training — 1024×128, `hhi_1024_motion` | **Done** — reward 0.84, 12,021 epochs |
| Hard-clip fine-tune — `hhi_1024_motion_tune` | **Done** — abandoned (jerk + catastrophic forgetting) |
| Transfer: raw betas — `hhi_1024_transfer` | **Done** — 21,400 epochs |
| Transfer: physics features — `hhi_phy_1024_transfer` | **Done** — 17,200 epochs |
| T1 clip overlap analysis | **Done** — 10-clip net difference, p=0.55 (not significant by binary metric) |
| Visual inference smoke test (8 envs) | **Done** — 5/8 vs 0/8 (physics features vs raw betas) |
| **Stage 1: 20,946 neutral clips — `hhi_20946_neutral`** | **Running** |
| Stage 2 data generation (20,946 clips × 128 shapes) | **In progress** — see `README.stage2-data-pipeline.md` |
| **Stage 2 transfer — `hhi_stage2_transfer`** | **Blocked on data** |
| E1 full CSV evaluation (Stage 2 checkpoint) | Not yet — needs Stage 2 checkpoint |
| E3 smoothness evaluator augmentation | Not yet |
| E7 held-out beta generalization | Partial — interp HUMOS inference done (717 files); pipeline incomplete |
| Kimodo beta-conditioning fine-tune (§3.2) | Not started |
| Kimodo generation quality eval (§4.1) | Not started — needs fine-tuned Kimodo |
| Per-shape physics eval (§4.2) | Not yet — needs Stage 2 checkpoint |
| Held-out beta eval (§4.3) | Not yet — needs pipeline + Stage 2 checkpoint |
| Torque / energy / contact / COM analysis (§4.4) | Not yet |
| Qualitative visualization (§4.5) | Not yet |

---

## Timeline (3 months to Sep 2026)

| Period | Tasks |
|---|---|
| Weeks 1–3 | Fine-tune Kimodo with beta conditioning; MLP physics training finishes |
| Weeks 2–3 | Run per-shape physics eval on 128 betas; generate held-out betas |
| Weeks 3–4 | Kimodo generation quality eval; held-out generalization eval |
| Weeks 4–5 | Extract torque, energy, contact, COM data; shape extremity correlation |
| Weeks 5–7 | Write paper, qualitative visualization, video |
| Weeks 8–12 | Buffer for revisions, polish, submission |

---

## Open Questions

- How difficult is beta conditioning in Kimodo? → Cross-attention injection is standard in diffusion models; risk is training instability on a small dataset (1024 clips × 128 shapes)
- Does the fine-tuned Kimodo generate noticeably different kinematics per beta, or does it collapse to the mean? → Key validation before investing in physics training
- Do we need more than 1024 motion clips? → Probably not for ICRA; full library is "future work"
- Is HUMOS still needed? → Yes, as the source of shape-specific reference motions for Kimodo fine-tuning and as a baseline comparison

---

## Training Details (for reference)

| Run | Config | Step time | Status |
|---|---|---|---|
| `hhi_1024_motion` | MLP, 4096 envs, 4× A40 | ~22s/step | ~8k steps |

Hardware note: 4× A40 ≈ 4× A100 for this workload (IsaacGym physics-sim bound, 30-50% GPU util).

---

## Strengthening Experiments (additional)

These build on the existing plan and address likely reviewer challenges. All are low effort — no new training required, just rollout data analysis on existing checkpoints.

### S1. Motion Type x Shape Extremity Interaction (2D analysis)

**Gap addressed:** current results average across motion types, hiding the structure of failures.

- Categorise the 1024 motion clips into types (locomotion, dynamic/jumping, manipulation, static/pose) using `data-processing/motion_id_text.json` text descriptions — can be done with simple keyword clustering
- For each (motion category, body shape extremity bucket) cell, compute mean body tracking distance
- Output: 2D heatmap — one axis motion category, other axis shape extremity (beta L2 norm buckets), color = tracking error
- Expected finding: locomotion robust across shapes, dynamic motions degrade sharply for extreme shapes
- **High impact, low effort** — motion categorisation is the only new work

### S2. Failure Mode Taxonomy

**Gap addressed:** current metrics measure *how much* the policy fails, not *why* physically.

- From rollout data, categorise failures into:
  - **Fall** — root height drops below threshold during episode
  - **COM drift** — horizontal displacement from reference exceeds bound
  - **Joint limit violation** — DOF hits limit repeatedly
  - **Contact failure** — foot sliding or floating (detected from contact forces)
- Plot failure type distribution per shape extremity bucket and per motion category
- Reveals which physical failure modes are shape-sensitive vs motion-sensitive
- Reusable diagnostic tool for the community — adds a methodological contribution

### S3. Motion Retargeting Behaviour

**Gap addressed:** shows the policy does implicit retargeting, not blind reference tracking.

- For a walking/locomotion clip, measure stride length and stride frequency across all 128 shapes
- Plot stride length vs body height — if positive correlation exists, the policy is scaling motion to body proportions
- Also check: does a shorter body take more steps to cover the same distance?
- **Very visual, good for video** — intuitive result that ICRA reviewers will appreciate
- Extractable from root position trajectory in existing rollouts

### S4. Fine-tuning Efficiency (optional, low priority)

**Gap addressed:** shows the single policy is also a strong initialisation for shape-specific specialisation.

- Take converged single policy, fine-tune for 200-500 steps on 2-3 specific shapes
- Compare convergence speed vs training from scratch on those same shapes
- If fine-tuning reaches same quality in ~10% of the steps: the single policy has practical value beyond deployment
- Cost: ~1 week of additional training on a subset — do only if timeline allows

---

## Strengthening Priority

| Experiment | Narrative value | Effort | Priority |
|---|---|---|---|
| S1. Motion type x shape heatmap | High — visual, reveals failure structure | Low | Do it |
| S2. Failure mode taxonomy | Medium — grounded, community value | Low | Do it |
| S3. Retargeting behaviour | Medium — intuitive, great for video | Low | Do it |
| S4. Fine-tuning efficiency | Medium — practical value | Medium | If time allows |
