# Research Results: Morphology-Generalized Physics-Based Motion Imitation

*Deep research on the five questions in README.research-prompt.md. Focus: actionable experiments.*

---

## Literature Gap Confirmed

**PULSE (ICLR 2024) uses mean SMPL shape only.** The paper explicitly states:
> "Our humanoid follows the kinematic structure of SMPL using the **mean shape**."

Proprioception in PULSE is `(joint_rotations, joint_positions, angular_velocities, linear_velocities)` — no beta parameters. PHC is the same. Neither XHugWBC nor H-Zero work with continuous SMPL beta variation. The whole "multiple SMPL body shapes" problem is ours to solve. This is a genuine gap.

---

## Q1: Why are concat and shape-embed identical?

### Literature

The answer is most likely **value miscalibration in the shared critic**, not actor encoding capacity.

The MorFiC paper (arXiv 2603.14554, 2026) identifies this exactly for multi-morphology training:
> "A shared critic tends to average incompatible value targets across embodiments, yielding miscalibrated advantages."

When 128 body shapes are trained together with a shared critic, the critic must predict a single value for states that have very different expected returns depending on body shape. A tall heavy body and a short light body doing the same squat motion have different physical difficulties, different natural frequencies, different torque requirements — yet the shared critic averages over all of them. This produces biased advantage estimates, which corrupts the actor update **regardless of how the actor encodes morphology**.

This explains the empirical finding directly: no matter how well the actor encodes the 11-dim betas (raw concat or nonlinear projector), the actor update signal is corrupted at the source (the advantage). A better actor encoding can't fix a broken advantage estimator.

The other two hypotheses are also partially true:
- **Shape variance << motion variance in gradients.** With 1024 diverse motion clips, the gradient signal is dominated by the wide variation in motion content. The 128-shape signal is a much smaller component of the loss landscape. The network learns motion-invariant features first; shape-invariant features emerge later if at all.
- **Redundancy via proprio.** Some shape information (body height, segment length ratios) is implicitly available from proprioceptive states. The 11-dim betas offer an explicit shortcut, but the network may not need it when the implicit signal is sufficient for seen shapes.

### Recommended experiment: MorFiC-style asymmetric critic conditioning

Condition **only the critic** on morphology. Keep the actor architecture unchanged (raw concat as currently). The morphology vector modulates the critic's value prediction per-shape, producing shape-specific advantage estimates.

Two sub-variants to compare:
- **Variant A (simple):** Critic receives a larger morphology input — instead of concatenating `morphology_obs` into the shared obs vector, give the critic a separate FiLM conditioning (since the critic is a single MLP, the fanout bottleneck is much smaller: 2 × 2 × 256 = 1024 outputs vs 12,288 for the actor).
- **Variant B (cleaner):** Critic obs = `[standard_obs]`; morphology enters only via a separate small network that produces a per-shape value offset/scale. This is the cleanest separation.

MorFiC results on quadrupeds: +16% on A1, ~2× on Cheetah, ~5× on B1 — purely from fixing value miscalibration.

**Risk/reward: high reward, 2–3 days. This is the highest-priority experiment.**

---

## Q2: What alternative conditioning architectures should be explored?

### Finding 1: Physics-derived features instead of raw betas

The multi-morphology robot control literature (XHugWBC, MetaMorph, ManyQuadrupeds, Body Transformer) consistently uses physics-derived features rather than raw model parameters. Typical features per limb/link:
- Link length (relative to parent)
- Link mass and mass ratio to total
- Joint limits (range, axis)
- Link geometry (radius/height for capsules)

For our SMPL case, compute from the loaded MJCF at asset-assignment time:
```
total_mass, com_height_standing,
upper_arm_len, forearm_len, thigh_len, shin_len,
torso_height, shoulder_width, hip_width,
arm_mass_ratio, leg_mass_ratio, torso_mass_ratio
```
~15 dims. These are nonlinear functions of the 10 betas and are directly in the space of the control problem (forces scale with mass, dynamics depend on limb lengths). They should generalize better to **unseen shapes** because the network learns to respond to physical quantities rather than PCA components.

**Risk/reward: medium reward (especially on held-out generalization), 3–5 days.** Requires extracting physics features from the MJCF at asset load time and caching per asset_id.

### Finding 2: Per-shape observation normalization

Standard running-mean/std normalization averages over all 128 shapes. A tall heavy body and a short light body have different proprioceptive distributions (different root height ranges, different joint velocity magnitudes). The shared normalizer produces incorrectly normalized inputs for each shape.

Fix: maintain separate `RunningMeanStd` buffers per `asset_id` (128 buffers). Each shape's obs is normalized to its own distribution. This is distinct from the morphology conditioning question — it is about providing the trunk with properly conditioned inputs regardless of shape.

SimBa (2024) showed that running normalization is the most important observation-processing choice for RL. Getting it wrong for a multi-shape setup is plausible.

**Risk/reward: low-medium, 1 day. Safe to try alongside other experiments.**

### Finding 3: PopArt / per-shape return normalization in the critic

A direct complement to MorFiC: use PopArt normalization (per-shape running return statistics) in the critic so that value predictions across shapes are on the same scale. Different shapes have genuinely different expected return magnitudes (easy upright walking vs hard squat), and the critic's output scale affects advantage quality. PopArt tracks a running mean/std of returns per shape bucket and normalizes the value head output accordingly. Used in multi-task RL (DeepMind) to handle reward scale mismatches.

**Risk/reward: medium, 2 days. Pairs well with asymmetric critic conditioning.**

### Finding 4: Hypernetwork weight generation

HyperDistill (arXiv 2402.06570) showed a hypernetwork generating full robot-specific MLP policies from morphology parameters achieves universal-controller-level performance on UNIMAL benchmark while being 6–14× smaller than a transformer. For our case, the hypernetwork could generate only the **bias terms** (or a low-rank weight adaptation, LoRA-style) of the actor given the 11-dim beta vector. This gives the actor shape-adaptive computation without the FiLM instability (no multiplicative coupling of gradients).

**Risk/reward: medium-high, 1 week. More engineering, but well-motivated by HyperDistill results.**

---

## Q3: State of the art for morphology-generalized control

### What exists (and why it doesn't directly apply)

| Paper | Venue | Morphology variation | Conditioning | Gap vs our work |
|---|---|---|---|---|
| PHC (Luo et al.) | NeurIPS 2023 | Single SMPL mean shape | None | No shape variation |
| PULSE (Luo et al.) | ICLR 2024 | Single SMPL mean shape | None | No shape variation |
| XHugWBC | 2026 | 12 different humanoid robots | Semantically aligned obs | Different robots, not SMPL betas |
| H-Zero | 2025 | Multiple humanoid designs | Pretraining + fine-tune | Different robots, not SMPL betas |
| MetaMorph | ICLR 2022 | 100 diverse robot designs | Morphology as positional embedding to transformer | Non-humanoid, much larger morphology variation |
| ManyQuadrupeds | ICRA 2024 | 3 quadruped designs | CPG phase scaling | Very different morphologies, not continuous |
| MorFiC | 2026 | Multiple quadruped designs | Asymmetric critic FiLM | Locomotion only, not motion imitation |

**The combination of (1) continuous SMPL body shape variation, (2) physics-based motion imitation across diverse motions, and (3) a single shared policy has not been done.** This is the novel contribution.

### What FiLM literature says about RL instability

FiLM was designed for visual question answering (supervised learning) where the conditioning signal is a rich language embedding and the modulated signal is a convolutional feature map. In RL, the conditioning signal is a low-dimensional physical parameter and the modulated signal is a learned feature representation with gradient flow. The FiLM-Ensemble paper shows FiLM's instability in probabilistic settings. AlphaStar uses FiLM successfully but with a large, well-trained language conditioning signal. For RL with small conditioning vectors, FiLM consistently underperforms simpler approaches — no paper was found showing FiLM improving on raw concat in this specific setting.

**Known fix for FiLM instability:** initialize gamma weights near zero (not standard init), so the modulation starts as near-identity. This avoids early-training gamma drift. If FiLM is retried, use zero-init for the gamma output layer.

---

## Q4: Training-side interventions to push past 0.85

### Finding 1: Contact reward for squat/crawl (MOST ACTIONABLE)

The 0.85 ceiling is substantially caused by ground-contact motions failing. The standard contact reward in PHC/SkillMimic is:
```
r_contact = (1/N) * sum_i [ 1(contact_sim_i == contact_ref_i) ]
```
where `i` indexes body parts that have contact in the reference (feet, knees, hands, hips). `contact_ref_i` is derived from the reference motion: a body part is in contact if its global position is within threshold of the floor height for the given body shape.

This provides explicit binary supervision on where the body should be touching the ground, which is otherwise invisible to the pose-tracking reward for non-standing motions. Without contact reward, the policy can achieve acceptable joint angle targets for a squat without the knees/hips actually touching the ground.

**Risk/reward: medium-high, 1–2 days. Directly addresses the known failure mode.**

### Finding 2: Torque Variation Score (TVS) as difficulty metric

The "Benchmarking Humanoid Imitation Learning with Motion Difficulty" paper (arXiv 2512.07248) introduces TVS — a physics-grounded difficulty metric that measures the magnitude of torque variation required to correct small pose perturbations around a reference motion. Key insight:
> "High-TV motions induce flat reward landscapes and vanishing policy gradients."

This directly explains why squat/crawl motions fail: they require large, precise torques to maintain non-standing configurations against gravity. The policy gradient signal near these reference states is very small (flat reward landscape), making them very slow to converge.

Practical implication: **the current difficulty score** (based on root velocity, flight ratio, DOF velocity, kinetic variance) **misclassifies squats/crawls as easy** (low root velocity, no flight). TVS would correctly rate them as hard. Consider re-computing difficulty scores using TVS rather than kinematic features, and weighting them more heavily in curriculum.

**Risk/reward: medium, 3–5 days for TVS computation. High value for curriculum design.**

### Finding 3: Motion phase variable in observation

Several recent papers (PULSE, Bi-Level Motion Imitation) include a phase variable φ ∈ [0, 1] that tracks progress through a motion clip. For episodic motion tracking:
```
φ_t = current_frame_idx / total_frames
```
This provides the policy with temporal context: "you are 60% through this motion clip." Without it, the policy cannot distinguish between the going-down phase and the coming-up phase of a squat (same joint angles, different velocities, but the reference pose is identical), which creates an aliasing problem.

**Risk/reward: medium, 1 day. Try it — it's a one-line observation addition.**

### Finding 4: Residual PD control

Used in PHC (validated on 11,313 AMASS clips). Changes action semantics from:
```
q_target = q_neutral + scale * action      # must learn full pose
```
to:
```
q_target = q_ref + scale * action          # only learns correction around reference
```
For squats and crawls, `q_neutral` is standing upright, far from the reference. The policy must output large actions just to reach the reference region, then fine-tune. With residual PD, `action=0` already tracks the reference, and the policy only learns balance corrections. This dramatically reduces the learning problem for non-standing motions.

**Risk/reward: high reward for hard motions, 2–3 days.** Requires passing `q_ref` (current reference DOF positions) from the motion manager to the PD controller at each step. Already partially implemented in the codebase (the mimic control component has access to the reference state).

### Finding 5: Early termination threshold relaxation for ground-contact motions

Standard fall detection uses root height threshold. For squat/crawl, the humanoid legitimately drops below this threshold. Either (a) classify motions by contact pattern and use motion-type-specific termination thresholds, or (b) use reference root height as the termination baseline (terminate if root is far below *reference* root height, not absolute floor height).

**Risk/reward: low-medium, 1 day. Required for squat/crawl to converge at all.**

---

## Q5: Novel research directions

### Direction 1: Asymmetric critic conditioning as the paper's method (strong reframe)

The finding that concat ≈ shape-embed is actually the most interesting empirical result in the paper, because it leads directly to a clean research question:
> *Is the bottleneck in morphology generalization the actor encoding or the critic's value estimation?*

The MorFiC result on locomotion (+16–500% from critic conditioning alone) suggests the answer is the critic. Demonstrating the same effect for **physics-based motion imitation across SMPL body shapes** would be a novel, clean, and well-motivated contribution. The narrative becomes:
- Naive approach: concatenate betas into obs → no benefit
- Wrong fix: better actor encoding (FiLM, shape embed) → still no benefit
- Right fix: morphology-conditioned critic → value miscalibration resolved → improved training

This would make the paper's "method" the asymmetric critic conditioning, not the FiLM comparison. The FiLM failure becomes an important negative result that motivates the approach.

### Direction 2: Physics-features as morphology descriptor (generalization claim)

Replace raw betas with physics-derived features (mass, COM, limb lengths) in the morphology obs. The paper claims generalization to unseen body shapes — but if the actor is conditioned on raw PCA betas, it has never seen the beta values of held-out shapes. If instead it's conditioned on physics features (mass, limb length), held-out shapes with similar masses/lengths to training shapes will map to nearby points in feature space. This makes the generalization claim physically principled:
> *A policy conditioned on body physics (not beta coordinates) generalizes to any body with physically similar properties.*

This is measurable: compare body distance on held-out shapes with betas vs physics features as input.

### Direction 3: Zero-shot generalization as the headline metric

The held-out evaluation (interpolation + extrapolation betas) is the core paper result. Structure the paper around a clear generalization curve:
- x-axis: beta distance from training distribution (interpolation → extrapolation)
- y-axis: body tracking error (MPJPE or custom distance metric)
- Compare: no conditioning vs betas concat vs physics features vs morphology-conditioned critic

The FiLM result fits here as a negative data point. The paper's conclusion is that physics features + critic conditioning gives the flattest generalization curve (least degradation at unseen shapes).

### Direction 4: Motion retargeting behavior analysis (visual contribution)

For a walking clip tracked across 128 body shapes: measure stride length and stride frequency per shape. If the policy is doing implicit retargeting (taller bodies take longer strides at similar frequency), this is a physically meaningful and visually compelling result — distinct from "the policy tracks the reference motion" (which says nothing about adaptation).

This analysis requires no new training, just rollout analysis on the existing checkpoint.

### Direction 5: Embodiment encoding probe

Linear probing on policy hidden activations is a standard analysis technique (confirmed by search: used in RL agents to measure state information captured by encoders). Application to morphology is less studied. Procedure:
1. At inference, record hidden activations of the actor/critic at the last hidden layer for each of 128 shapes
2. Fit linear regression: `activation_vector → [total_mass, com_height, limb_lengths]`
3. Report R² per physical property

This is "the AI learns physics" result. If R² > 0.8 for mass and limb length from activations, the policy has built an internal representation of body physics purely from imitation learning. Strong finding. No new training needed — only works on the already-converged checkpoint.

---

## Ranked Experiment List

Ordered by expected impact × implementation speed, focusing on what to try next:

| Priority | Experiment | Expected gain | Implementation cost | Status |
|---|---|---|---|---|
| 1 | **Asymmetric critic conditioning** (morphology → critic FiLM/concat, actor unchanged) | High — fixes value miscalibration | 2–3 days | Not started |
| 2 | **Residual PD control** (`q_target = q_ref + scale*action`) | High — directly helps squat/crawl | 2–3 days | Not started |
| 3 | **Contact reward** for ground-contact motions | Medium-high — fixes squat/crawl termination and reward signal | 1–2 days | Not started |
| 4 | **Early termination fix for squat/crawl** (reference-relative height threshold) | Medium — necessary prerequisite for 3 | 1 day | Not started |
| 5 | **Motion phase variable φ** in observation | Medium — resolves temporal aliasing in symmetric motions | 1 day | Not started |
| 6 | **Per-shape running normalization** (separate RunningMeanStd per asset_id) | Low-medium — correctness improvement | 1 day | Not started |
| 7 | **Physics-derived morphology features** (mass, COM, limb lengths from MJCF) | Medium — better generalization to held-out shapes | 3–5 days | Not started |
| 8 | **Held-out evaluation** (generate interpolation + extrapolation betas via HUMOS) | Critical for paper | Ongoing — generate now | Not started |
| 9 | **TVS difficulty re-scoring** (recompute difficulty with physics-grounded metric) | Medium — better curriculum | 3–5 days | Not started |
| 10 | **Embodiment encoding probe** (linear regression on activations) | High narrative value | 1–2 days analysis only | Needs checkpoint |

### Natural grouping for two parallel tracks:

**Track A — Fix performance (push past 0.85):**
Experiments 1 + 2 + 3 + 4 + 5. These address the value miscalibration and ground-contact failure modes. Run as a single combined experiment after the current run converges — no need to ablate individually first.

**Track B — Fix generalization and analysis:**
Experiments 6 + 7 + 8 + 10. These address the paper's main claim (cross-shape generalization). Start experiment 8 now (no training needed, just HUMOS inference). Run 6 + 7 once Track A converges.

---

## Summary of Key Literature Findings

- **PULSE and PHC use mean SMPL shape only** — our multi-shape work fills a genuine gap
- **MorFiC (2026)**: conditioning only the critic on morphology fixes value miscalibration; +16–500% on quadrupeds
- **FiLM in RL**: no paper shows FiLM improving on raw concat for low-dimensional physics conditioning vectors; the known fix if retried is zero-init on the gamma output layer
- **Torque Variation Score (TVS)**: physics-grounded difficulty metric that correctly rates squats/crawls as hard (flat reward gradient), unlike kinematic metrics
- **Phase variable φ**: standard in character animation, resolves temporal aliasing in bi-directional motions
- **Contact reward**: standard in PHC/SkillMimic for body-object or body-floor contact matching
- **Residual PD**: used in PHC, validated on 11,313 AMASS clips, most impactful for non-standing motions
- **Linear probing on activations**: known technique in RL, not yet applied to morphology encoding — strong analysis opportunity

---

## Sources

- [PULSE: Universal Humanoid Motion Representations for Physics-Based Control](https://arxiv.org/abs/2310.04582)
- [MorFiC: Fixing Value Miscalibration for Zero-Shot Quadruped Transfer](https://arxiv.org/abs/2603.14554)
- [Benchmarking Humanoid Imitation Learning with Motion Difficulty](https://arxiv.org/abs/2512.07248)
- [Scalable and General Whole-Body Control for Cross-Humanoid Locomotion (XHugWBC)](https://arxiv.org/abs/2602.05791)
- [H-Zero: Cross-Humanoid Locomotion Pretraining](https://arxiv.org/abs/2512.00971)
- [MetaMorph: Learning Universal Controllers with Transformers](https://arxiv.org/abs/2203.11931)
- [Universal Morphology Control via Contextual Modulation](https://proceedings.mlr.press/v202/xiong23a/xiong23a.pdf)
- [Distilling Morphology-Conditioned Hypernetworks (HyperDistill)](https://arxiv.org/abs/2402.06570)
- [ManyQuadrupeds: Learning a Single Locomotion Policy for Diverse Quadruped Robots](https://arxiv.org/abs/2310.10486)
- [ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation](https://arxiv.org/abs/2510.05070)
- [Bi-Level Motion Imitation for Humanoid Robots](https://arxiv.org/abs/2410.01968)
- [Body Transformer: Leveraging Robot Embodiment for Policy Learning](https://arxiv.org/abs/2408.06316)
- [SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning](https://arxiv.org/abs/2410.09754)
- [Hypernetworks for Zero-shot Transfer in Reinforcement Learning](https://arxiv.org/abs/2211.15457)
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://ojs.aaai.org/index.php/AAAI/article/view/11671)
