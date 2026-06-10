# Paper Plan: Morphology-Generalized Physics-Based Motion Imitation

**Target venue:** ICRA 2027
**Deadline:** ~September 2026
**Time remaining:** ~3 months

---

## Core Contribution

> "We present the first physics-based motion imitation system that generalizes a single policy across a distribution of human body shapes, and introduce an evaluation protocol for measuring per-shape tracking quality."

The central capability: a single physics-based controller that imitates human motion across 128 SMPL body shape variants (64 shapes × 2 genders), trained via morphology-conditioned imitation learning.

**Why this matters for robotics:**
- Deploy a single controller across humanoid robots with manufacturing variations
- Sim-to-real transfer where simulation morphology ≠ real robot morphology
- Digital humans and embodied AI with diverse body proportions

**What is novel (the combination, not the parts):**
1. Shape-conditioned data pipeline — HUMOS generates per-beta motion predictions, creating paired (motion, body shape) training data at scale
2. FiLM-conditioned policy — shape parameters modulate feature processing rather than naive concatenation
3. Cross-shape generalization evaluation — per-body-shape tracking error on held-out betas (interpolation and extrapolation), a metric not previously used for physics-based controllers

**What is NOT the headline:** FiLM vs MLP architecture comparison — this becomes an ablation, not the main claim.

---

## Paper Structure

### 1. Introduction
- Gap: physics-based humanoid controllers (AMP, ASE, ProtoMotions) assume fixed morphology
- Motivation: real applications need shape generalization (digital humans, humanoid robots, sim-to-real)
- Contribution summary: single policy across SMPL shape distribution, evaluation protocol, FiLM conditioning

### 2. Related Work
- Physics-based motion imitation: PFNN, AMP, ASE, ProtoMotions
- Morphology-robust control: evolution strategies, modular robots
- SMPL-based animation and HUMOS
- Conditional/adaptive policies: FiLM, hypernetworks

### 3. Method
- **Problem formulation:** morphology-conditioned MDP — state includes body shape parameters
- **Data pipeline:** HUMOS → shape-paired motion library (1024 clips × 128 variants = 131k motions)
- **Policy architecture:** FiLM conditioning — morphology_obs as conditioner, not trunk input
- **Morphology-matched motion sampling:** each env assigned a fixed asset, samples only matching-shape motions

### 4. Experiments

The central claim is not just "our policy tracks motion across shapes" — it is that **the policy learns shape-adaptive physical control strategies**. Different body shapes have fundamentally different dynamics (inertia tensors, COM height, limb lengths, natural frequencies). The experiments below demonstrate both tracking quality and physical meaningfulness.

#### 4.1 Per-shape tracking performance (128 training betas)
- Tool: `evaluate_hhi_faults.py` + `HHIFaultEvaluator`
- Metric: per-(gender, beta_key) mean/max body distance
- Analysis: distribution of per-shape performance, correlation with beta L2 norm (shape extremity)
- Expected result: uniform performance across shapes, mild degradation at extremes

#### 4.2 Generalization to held-out shapes
- **Interpolation:** ~16–32 new betas sampled from [-3, 3] (different seed from training 128)
- **Extrapolation:** betas scaled to [-5, 5] range
- Metric: body distance degradation pattern, MLP vs FiLM comparison
- Note: SMPL motions at ±5 may be noisier — account for this in analysis
- FiLM should degrade more gracefully on extreme/unseen shapes

#### 4.3 Shape-adaptive physical analysis (key differentiator)

These experiments show the policy has learned physically meaningful shape-conditioned behaviour, not just blind reference tracking.

**4.3a Shape-conditioned torque and energy analysis**
- For the same motion clip, record joint torques and total energy expenditure across different body shapes
- If the policy is genuinely shape-aware: heavier bodies → higher torques, shorter limbs → different timing
- Directly shows FiLM conditioning is doing something physically meaningful

**4.3b Stability analysis across shapes**
- Track COM trajectory and support polygon coverage during locomotion across shapes
- Extreme body proportions (very tall, very heavy) should show measurable but bounded stability degradation
- Connects shape extremity to physical difficulty in a principled, interpretable way

**4.3c Contact quality across shapes**
- Foot contact patterns (timing, forces, duration) for the same motion across different shapes
- Does the policy adapt contact timing to body proportions?
- Physically interpretable and visually compelling for ICRA

**4.3d Shape extremity vs physical failure correlation**
- Plot body tracking distance against beta L2 norm AND against specific physical features (estimated limb length ratio, mass)
- More insightful than "extreme betas are harder" — explains *why* in physical terms

**4.3e FiLM activation analysis**
- Visualise how FiLM gamma/beta modulation parameters vary across body shapes for the same motion
- If FiLM learns physically meaningful modulations: smooth variation correlated with physical body properties (height, mass)
- The "AI connects to physics" story in a single figure

#### 4.4 MLP vs FiLM ablation
- Already running: `hhi_1024_motion` (MLP) vs `hhi_film_1024_motion` (FiLM)
- Compare: reward convergence, per-shape tracking variance, held-out generalization, physical metrics (4.3a–d)
- FiLM's advantage should be most visible on unseen/extreme shapes and in physical consistency

#### 4.5 Qualitative visualization
- Same motion clip performed simultaneously by multiple body shapes side-by-side
- Visually compelling for ICRA reviewers and video submission

### 5. Conclusion
- Demonstrated a single physics-based policy generalizing across 128 SMPL body shape variants
- Policy learns shape-adaptive physical strategies — not just reference tracking, but physically consistent control
- 1024-clip pilot validates the approach; scaling to full 20,951 clips is straightforward (future work)
- Potential extensions: sim-to-real transfer with morphology mismatch, SMPLX, non-humanoid morphologies

---

## Experiment Status

| Experiment | Status | ETA |
|---|---|---|
| MLP training (`hhi_1024_motion`) | Running, ~8k steps, reward 0.84 | ~+2 days |
| FiLM training (`hhi_film_1024_motion`) | Running, ~2.3k steps, reward 0.72 | ~+7 days |
| Per-shape eval on 128 training betas (4.1) | Not started — needs checkpoint | After MLP converges |
| Held-out beta generation via HUMOS (4.2) | Not started — can run now | This week |
| Held-out generalization eval (4.2) | Not started | After held-out data ready |
| Torque / energy analysis (4.3a) | Not started — needs checkpoint | Weeks 2–3 |
| Stability / COM analysis (4.3b) | Not started — needs checkpoint | Weeks 2–3 |
| Contact quality analysis (4.3c) | Not started — needs checkpoint | Weeks 2–3 |
| Shape extremity correlation (4.3d) | Not started | Weeks 3–4 |
| FiLM activation analysis (4.3e) | Not started — needs FiLM checkpoint | After FiLM converges |
| Qualitative visualization (4.5) | Not started | Weeks 4–5 |

---

## Timeline (3 months to Sep 2026)

| Period | Tasks |
|---|---|
| Weeks 1–2 | Both trainings finish; run per-shape eval on 128 betas; generate held-out betas via HUMOS |
| Weeks 2–3 | Run held-out generalization eval; extract torque, energy, contact, COM data from rollouts |
| Weeks 3–4 | FiLM activation analysis; shape extremity correlation; all physical analysis figures |
| Weeks 4–7 | Write paper, qualitative visualization, video |
| Weeks 8–12 | Buffer for revisions, polish, submission |

---

## Open Questions

- Do we need more than 1024 motion clips? → Probably not for ICRA; full 20,951 is "future work"
- Transfer learning from pilot to full dataset → Reduces steps 30–50% but wall-clock bottleneck is physics sim, not NN; revisit if needed
- Does FiLM clearly outperform MLP on held-out shapes? → The entire architecture justification depends on this; if not, reframe as ablation showing comparable performance with better generalization potential

---

## Training Details (for reference)

| Run | Config | Step time | Status |
|---|---|---|---|
| `hhi_1024_motion` | MLP, 4096 envs, 4× A40 | ~22s/step | ~8k steps |
| `hhi_film_1024_motion` | FiLM, 8192 envs, 4× A40 | ~34s/step | ~2.3k steps |

Hardware note: 4× A40 ≈ 4× A100 for this workload (IsaacGym physics-sim bound, 30–50% GPU util).
