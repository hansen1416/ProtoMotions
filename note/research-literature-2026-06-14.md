# Literature Review: Morphology-Generalized Humanoid Motion Imitation (2020–2026)

**Date**: 2026-06-14  
**Context**: ProtoMotions project — single policy across 128 SMPL body shapes, 69-DOF humanoid in IsaacGym, HUMOS/AMASS motion data.

**Current state**: All 4 conditioning approaches (raw beta concat, FiLM, shape embed+concat, physics features) plateau at reward ≈ 0.84. Hard failure modes: 65 floor-contact clips (crawl/kneel/squat/backward-walk) persistent across all runs. Physics features experiment (hhi_physics_feat_1024) shows early reward similar to baseline — likely not a breakthrough.

---

## Section 1: Physics-Based Motion Imitation — Core Foundations

### PHC: Perpetual Humanoid Control for Real-Time Simulated Avatars
- **Venue/Year**: ICCV 2023 | Zhengyi Luo, Jinkun Cao, et al.
- **Method**: Progressive multiplicative control policy (PMCP) that dynamically allocates new network capacity to increasingly difficult motion sequences via a mixture-of-experts gating mechanism, without catastrophic forgetting. Trains on ~15k AMASS clips; uses residual force control during early training then phases it out. **Critically: PHC explicitly supports multi-shape SMPL humanoids — 16 humanoids with different gender and body proportions are co-trained under one policy, with shape beta parameters fed as input.**
- **Relevance**: Directly addresses this problem. PHC's multi-shape training is the closest prior work. The PMCP/MoE structure means different network sub-trees can specialize to different body morphologies rather than a single flat MLP — avoids the "fanout bottleneck" hit with FiLM. Achieves 98.9% on AMASS-Train.
- **Key insight**: MoE over body shape clusters may let different expert heads specialize without conflicting gradients across diverse shapes.
- **Links**: [GitHub](https://github.com/ZhengyiLuo/PHC), [Project page](https://www.zhengyiluo.com/PHC-Site/)

### PULSE: Universal Humanoid Motion Representations for Physics-Based Control
- **Venue/Year**: ICLR 2024 (Spotlight) | Zhengyi Luo et al.
- **Method**: Builds on PHC+ (100% AMASS success). Learns a 32-dimensional latent motion space via a VAE with variational information bottleneck trained jointly with a motion imitator. A proprioception-conditioned prior allows random sampling from the latent space for generative tasks. Hierarchical RL then operates in this latent space.
- **Relevance**: If near-100% tracking exists for average body shape, PULSE's latent space provides a foundation for hierarchical control across body shapes. The latent space separates "what motion" from "how to execute it given this body" — the high-level latent can be shape-agnostic while the low-level tracker is shape-conditioned. Could address floor-contact failures by routing through a more expressive motion prior.
- **Links**: [arXiv 2310.04582](https://arxiv.org/abs/2310.04582), [GitHub](https://github.com/ZhengyiLuo/PULSE)

### MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting
- **Venue/Year**: SIGGRAPH Asia 2024 | Nv-TLabs (Peng et al.)
- **Method**: Frames motion tracking as a masked inpainting problem — randomly masks subsets of body part targets during training, forcing the policy to infer full-body motion from partial specifications. Handles contact-rich and variable behaviors better than DeepMimic/AMP.
- **Relevance**: The masking approach makes the policy robust to imperfect or partial reference motion, directly relevant to floor-contact clips where MoCap contacts may be noisy or inconsistent across body shapes.

### InterMimic: Universal Whole-Body Control for Physics-Based Human-Object Interactions
- **Venue/Year**: arXiv 2502.20390, February 2025 | UIUC + EA
- **Method**: Curriculum teacher-student distillation for HOI. Trains subject-specific oracle teacher policies (with privileged info), then distills into a single student. Handles **diverse human shapes** and performs motion retargeting across different human models.
- **Relevance**: The teacher-per-shape → single student distillation strategy is highly relevant. Train 10–20 shape-cluster teachers → distill into one student conditioned on beta embedding. Decomposes the hard joint optimization into easier per-cluster subproblems.
- **Links**: [arXiv 2502.20390](https://arxiv.org/pdf/2502.20390)

### SMPLOlympics: Sports Environments for Physically Simulated Humanoids
- **Venue/Year**: NeurIPS 2024 Workshop | arXiv 2407.00187
- **Method**: Benchmark of Olympic sports (golf, tennis, basketball) using SMPL/SMPL-X compatible humanoids in physics simulation. Shows AMP-style motion priors + simple task rewards → human-like athletic behavior across body-shape-diverse SMPL humanoids.
- **Relevance**: Confirms SMPL body shape diversity in physics-based RL is tractable with AMP/motion priors. The conditioning strategy used there is worth examining.
- **Links**: [arXiv 2407.00187](https://arxiv.org/abs/2407.00187), [GitHub](https://github.com/SMPLOlympics/SMPLOlympics)

---

## Section 2: SMPL / Human Body Shape in Physics-Based RL

### HUMOS: Human Motion Model Conditioned on Body Shape
- **Venue/Year**: ECCV 2024 | Shashank Tripathi, Omid Taheri, Michael J. Black et al.
- **Method**: Data-driven motion generation model conditioned on SMPL shape beta parameters. Uses identity-preserving cycle consistency loss and **differentiable physics terms**: foot-slide penalty, ground penetration, dynamic stability via CoM/CoP/ZMP. First model generating body-shape-aware motions that are dynamically stable.
- **Relevance**: This is the source motion dataset. HUMOS shows that conditioning on beta is necessary but not sufficient — derived physical quantities (limb lengths, CoM height, inertia) are what matter for dynamic stability. Their ZMP/CoP losses directly predict the floor-contact failure modes. **Using HUMOS-style physics supervision on the policy (as reward terms, not just input features) might directly help the 65 failing clips.**
- **Key data point**: Their physics terms improve foot-slide and ground penetration — exactly the floor-contact failure class.
- **Links**: [ECCV 2024](https://dl.acm.org/doi/10.1007/978-3-031-72640-8_8), [arXiv 2409.03944](https://arxiv.org/html/2409.03944v1)

---

## Section 3: Morphology-Generalized Policies — Universal Controllers

### MetaMorph: Learning Universal Controllers with Transformers
- **Venue/Year**: ICLR 2022 | Agrim Gupta, Li Fei-Fei et al.
- **Method**: Encodes robot morphology as positional embeddings in a Transformer encoder. Proprioceptive observations and morphology context are node inputs; shared Transformer produces per-joint actions. Pre-trained on diverse modular robot morphologies.
- **Relevance**: Morphology as a modality (structural positional encoding) rather than a global conditioning vector. For SMPL, encode each of the 69 DOFs with its kinematic context (parent body, limb length, mass contribution) as per-joint tokens, then use attention to let joints coordinate given their body-shape-adjusted kinematics.
- **Links**: [arXiv 2203.11931](https://arxiv.org/abs/2203.11931), [GitHub](https://github.com/agrimgupta92/metamorph)

### Body Transformer (BoT): Leveraging Robot Embodiment for Policy Learning
- **Venue/Year**: arXiv 2408.06316, August 2024
- **Method**: Models robot body as a graph where nodes are sensors and actuators. Applies **highly sparse masked attention** — each node attends only to its direct kinematic neighbors. Outperforms vanilla Transformer and MLP on both IL and RL tasks with better scaling and efficiency.
- **Relevance**: **Strong next-architecture candidate.** For the 69-DOF SMPL humanoid, the kinematic adjacency mask is fixed (SMPL skeleton topology), but attention weights can be dynamically influenced by body shape features injected into node embeddings. Shape info is local (each joint attends to shape of adjacent body segment) — sidesteps the FiLM fanout bottleneck entirely.
- **Links**: [arXiv 2408.06316](https://arxiv.org/abs/2408.06316)

### Universal Morphology Control via Contextual Modulation (Xiong et al.)
- **Venue/Year**: ICML 2023
- **Method**: Learns an embodiment context vector from consecutive proprioceptive observation history, then uses this context to modulate the policy via multiplicative and additive conditioning at each layer. Conditioning on **inferred** latent morphology (not raw beta).
- **Relevance**: The FiLM experiment here failed because it conditioned on raw beta (11 PCA components). This work suggests learning a latent morphology embedding from **observation history** (which implicitly encodes body dynamics and inertia) and using that as the conditioning signal. The context is inferred, not given — analogous to online system identification.
- **Reported metrics**: Improved zero-shot generalization on unseen morphologies vs. MetaMorph and NerveNet baselines.
- **Links**: [arXiv 2302.11070](https://arxiv.org/abs/2302.11070), [ICML proceedings](https://proceedings.mlr.press/v202/xiong23a/xiong23a.pdf)

### DMAP: Distributed Morphological Attention Policy
- **Venue/Year**: NeurIPS 2022 | amathislab
- **Method**: Biologically-inspired: independent proprioceptive processing per body part + distributed per-joint controllers + attention gating sensory information from different body parts. **Trained without explicit morphology parameters** — infers morphology implicitly from proprioceptive signals.
- **Relevance**: DMAP shows explicit morphology access is not necessary for good generalization — but attention-based gating of proprioceptive streams per joint is highly effective. Key result: **matches oracle agent (with privileged morphology access) despite not seeing morphology parameters.** Consider combining DMAP-style distributed attention with explicit beta as an optional auxiliary.
- **Links**: [arXiv 2209.14218](https://arxiv.org/abs/2209.14218), [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f0fae49cdfab57c41c30c9b0244093cb-Abstract-Conference.html)

### URMA: One Policy to Run Them All — Multi-Embodiment Locomotion
- **Venue/Year**: CoRL 2024 | Bohlinger et al.
- **Method**: Unified Robot Morphology Architecture: attention encoder for joint/feet info → core gait network (shared) → universal decoder generating per-joint actions. Morphology-agnostic encoders and decoders; the core network learns an abstract "gait primitive" shared across embodiments.
- **Relevance**: The separation of **(1) morphology-aware encoding → (2) shared dynamics → (3) morphology-aware decoding** is architecturally elegant. For 128 SMPL shapes: shape-specific encoder extracts normalized proprioceptive representation, shared trunk learns motion, decoder projects to shape-specific joint targets. Zero-shot transfer demonstrated on real robots.
- **Links**: [arXiv 2409.06366](https://arxiv.org/pdf/2409.06366)

### HyperDistill: Distilling Morphology-Conditioned Hypernetworks
- **Venue/Year**: arXiv 2402.06570, February 2024
- **Method**: Morphology-conditioned hypernetwork generates robot-wise MLP policies (full weight matrices, not just scale/shift). Distills into a single efficient policy. Achieves transformer-level performance at 6–14× smaller model size on the UNIMAL benchmark.
- **Relevance**: Hypernetworks generate specialized weights per body shape — directly addresses the fanout bottleneck that FiLM hit. FiLM only modulates via scale/shift; hypernetworks generate entire weight matrices. For 128 fixed shapes, hypernetwork outputs can be **precomputed per shape** (amortized at test time). The distillation step collapses into a single model.
- **Links**: [arXiv 2402.06570](https://arxiv.org/abs/2402.06570)

### Learning to Get Up Across Morphologies: Zero-Shot Recovery
- **Venue/Year**: arXiv 2512.12230, December 2024
- **Method**: Single deep RL policy trained with CrossQ that recovers from falls across 7 humanoid robots with diverse heights (0.48–0.81m) and weights (2.8–7.9 kg). Leave-one-out experiments show targeted morphological coverage is essential for zero-shot generalization.
- **Relevance**: **Diversity of training morphologies improves zero-shot transfer.** The "targeted coverage" finding is critical — representative body shapes in training matter more than uniform random sampling. For 128 shapes spanning height 1.13–1.67m and mass 26–144 kg, stratified coverage (extremes of the distribution) matters.
- **Links**: [arXiv 2512.12230](https://arxiv.org/abs/2512.12230)

### Embedding Morphology into Transformers for Cross-Robot Policy Learning
- **Venue/Year**: arXiv 2603.00182, March 2026 | Suzuki, Liu, Wang et al.
- **Method**: Three mechanisms for injecting morphology into a VLA transformer: (1) kinematic tokens factorizing actions across joints with per-joint temporal chunking, (2) topology-aware attention bias encoding kinematic tree topology in self-attention, (3) joint-attribute conditioning augmenting topology with per-joint descriptors (link length, DOF axis, etc.).
- **Relevance**: Directly actionable: for SMPL humanoid, "joint attributes" = limb lengths, body segment masses, inertias derived from beta. Encoding these as per-joint attributes in a topology-aware attention layer incorporates shape directly into the computation graph — each joint's policy output shaped by its own physical properties. More expressive than global conditioning.
- **Links**: [arXiv 2603.00182](https://arxiv.org/pdf/2603.00182)

---

## Section 4: Policy Architecture — Nuanced Conditioning Mechanisms

### Policy-Space Diffusion for Physics-Based Character Animation
- **Venue/Year**: ACM Transactions on Graphics 2025 | Sheldon Andrews et al.
- **Method**: Uses policy networks as representations of motion. Common Neighbor Policy regularization constrains policy similarity during training. Then trains a diffusion model over policy **parameter space** to sample policies for novel conditions (morphologies, motions). New morphologies get policies generated in seconds without retraining.
- **Relevance**: Radically different approach: instead of conditioning one policy on morphology, train a diffusion model over policy parameter space and sample a policy conditioned on morphology at inference. For 128 fixed body shapes, precompute 128 policies via sampling conditioned on beta embeddings. Especially useful if per-shape specialization is truly necessary and a single shared policy fundamentally cannot achieve it.
- **Links**: [ACM ToG 2025](https://dl.acm.org/doi/full/10.1145/3732285)

### NerveNet: Learning Structured Policy with Graph Neural Networks
- **Venue/Year**: ICLR 2018 | Tingwu Wang et al. (foundational)
- **Method**: Models agent as a graph; propagates information via GNN message passing over kinematic structure. Policies are significantly more transferable and generalizable than MLP baselines, with zero-shot transfer to unseen morphologies.
- **Relevance**: For SMPL, the kinematic graph has fixed topology but **variable edge weights** (limb lengths from beta). GNN message passing with beta-conditioned edge features encodes "this leg is longer than average by X cm" as an edge feature flowing from hip to knee to ankle — principled propagation of shape through kinematic hierarchy.

---

## Section 5: Curriculum Learning for Hard Motion Classes

### Benchmarking Humanoid Imitation Learning with Motion Difficulty
- **Venue/Year**: arXiv 2512.07248, December 2024
- **Method**: Introduces Motion Difficulty Score (MDS) — measures torque variation induced by small pose perturbations. Larger torque-to-pose variation = flatter reward landscape = harder to learn. Demonstrates 3-stage curriculum: (1) easy only → (2) medium → (3) all motions. **Key finding: curriculum ordering from easy to hard significantly outperforms uniform random sampling.**
- **Relevance**: Directly applicable to the floor-contact failure problem. Compute MDS for 1024 clips — crawl/kneel/squat/backward-walk are almost certainly high-MDS due to non-upright CoM and high ground-contact torques. MDS-based curriculum is also compatible with morphology curriculum: start easy motions on average shapes, expand to extreme shapes on easy motions, then add hard motions.
- **Links**: [arXiv 2512.07248](https://arxiv.org/abs/2512.07248)

### Learning Motion Skills with Adaptive Assistive Curriculum Force
- **Venue/Year**: arXiv 2506.23125, June 2025
- **Method**: Uses adaptive assistive forces (external forces applied to the humanoid) as a curriculum signal. Force magnitude reduces as policy improves, providing smooth curriculum from "physically assisted" to "unassisted" for dynamically challenging motions.
- **Relevance**: Floor-contact motions (crawl, kneel, squat) are hard because getting into and out of ground contact requires managing contact forces precisely. Assistive forces at pelvis/torso during floor-contact phase, phased out as reward improves, could break the cold-start problem for the 65 hard clips.
- **Links**: [arXiv 2506.23125](https://arxiv.org/html/2506.23125)

### HiFAR: Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery
- **Venue/Year**: arXiv 2502.20061, February 2025
- **Method**: Multi-stage curriculum for fall recovery — starts with simplified dynamics (reduced physics fidelity, assistive forces), progressively moves to full physics. Applied to getting-up motions which are structurally similar to floor-contact failures.
- **Relevance**: Getting off the floor (squat→stand, kneel→stand, crawl→stand) is mechanistically related to fall recovery. HiFAR's multi-stage curriculum for ground-contact motions is directly applicable to the 65 hard clips.
- **Links**: [arXiv 2502.20061](https://arxiv.org/pdf/2502.20061)

---

## Section 6: Teacher-Student / Distillation for Multi-Shape Policies

### UniLegs: Universal Multi-Legged Robot Control through Morphology-Agnostic Policy Distillation
- **Venue/Year**: arXiv 2507.22653, July 2025
- **Method**: Per-morphology teacher policies trained with PPO using privileged information (full state), then distilled into a single morphology-agnostic student using **asymmetric actor-critic** (actor: proprioception only; critic: full state + morphology). Student achieves 99.36% of teacher performance on trained morphologies, 72.64% zero-shot on unseen morphologies.
- **Relevance**: The asymmetric actor-critic during distillation is key: the critic sees body shape explicitly and guides value estimation, while the actor learns to infer shape implicitly from proprioception. For this problem: train 10–20 shape-cluster teacher policies → distill into one student. Decomposes the hard joint problem into easier subproblems.
- **Links**: [arXiv 2507.22653](https://arxiv.org/pdf/2507.22653)

### Concurrent Teacher-Student with Asymmetric Actor-Critic
- **Venue/Year**: arXiv 2405.10830, May 2024
- **Method**: Teacher and student train concurrently using PPO within asymmetric actor-critic. Agents share the same policy network but receive different observations (teacher: privileged; student: proprioceptive only). Critic always sees privileged information even during student training.
- **Relevance**: Can be adapted so teacher sees body shape beta + full state, student sees proprioception only. Concurrent training avoids two-stage pipeline overhead. Addresses the scenario where body shape is known during training but ideally should not be required at inference.
- **Links**: [arXiv 2405.10830](https://arxiv.org/html/2405.10830v1)

---

## Section 7: Domain Adaptation / Zero-Shot Morphology Transfer at Inference

### Fast Adaptation with Behavioral Foundation Models
- **Venue/Year**: arXiv 2504.07896, April 2025
- **Method**: Meta-learning approach enabling 10–40% improvement over zero-shot behavioral foundation model performance in only a few episodes of online interaction. Adapts a pretrained policy to new morphologies/dynamics rapidly.
- **Relevance**: If a good base policy exists for average shapes, rapid few-episode adaptation to each of the 128 shapes at inference time could be more practical than single-policy generalization. Especially useful for extreme shapes (1.13m/26kg and 1.67m/144kg).
- **Links**: [arXiv 2504.07896](https://arxiv.org/pdf/2504.07896)

### Knowledge Diversion for Efficient Morphology Control and Policy Transfer
- **Venue/Year**: arXiv 2512.09796, December 2024
- **Method**: Uses knowledge distillation with "knowledge diversion" — explicitly routing different aspects of the policy (style, dynamics adaptation) to different components. Achieves efficient transfer across morphologies.
- **Relevance**: Disentangling what the policy needs to know about motion style from what it needs to know about morphology dynamics may help the case where both vary simultaneously.
- **Links**: [arXiv 2512.09796](https://arxiv.org/pdf/2512.09796)

---

## Section 8: Data Augmentation and Motion Retargeting

### Implicit Kinodynamic Motion Retargeting for Human-to-Humanoid Imitation Learning
- **Venue/Year**: arXiv 2509.15443, September 2025
- **Method**: Neural motion retargeting that learns a direct mapping from SMPL motion sequences to physically feasible humanoid robot motion, bypassing local optima in optimization-based retargeting. Uses kinematics + dynamics awareness to avoid morphology gap issues.
- **Relevance**: Floor-contact failures may partly stem from retargeting artifacts — SMPL MoCap clips of crawling/kneeling may have contact positions not physically valid for all 128 body shapes. Neural retargeting that is body-shape-aware could generate per-shape reference motions, giving the policy a physically consistent target for each shape.
- **Links**: [arXiv 2509.15443](https://arxiv.org/pdf/2509.15443)

### MotionAug: Augmentation with Physical Correction for Human Motion Prediction
- **Venue/Year**: arXiv 2203.09116, 2022
- **Method**: Physics-in-the-loop data augmentation: generate motion variants, apply physical correction (physics simulation stepping) to make them physically plausible.
- **Relevance**: For the 65 failing floor-contact clips, physics-corrected augmentation could generate per-shape-corrected variants. This expands hard-motion training data from 65 clips to 65 × K clips where K body shapes each contribute a physically valid version.
- **Links**: [arXiv 2203.09116](https://arxiv.org/pdf/2203.09116)

---

## Section 9: Alternative Observation Features

### ZMP / Support Polygon as Policy Conditioning

HUMOS uses ZMP-based stability margin as a training signal. For a physics policy, these can be runtime **observation features** (not just training losses):

- **Dynamic Stability**: ratio of CoM horizontal velocity to CoP position — high values indicate instability during floor-contact transitions (kneeling, squatting).
- **ZMP-based stability margin**: directly measures whether Zero Moment Point is within the support polygon. For floor-contact motions, the support polygon changes dramatically (foot-only → full-body contact area). Conditioning the policy on current support polygon size/shape (derived from body shape + current contacts) is novel and potentially impactful.

### Latent Morphology Inference from Observation History (Contextual Modulation, Xiong et al.)

Rather than feeding raw 11-dim beta, infer a 16–32 dim latent body context from the last N proprioceptive observations. This approach:
1. Works even if beta parameters are not available at inference
2. Captures dynamic properties (not just kinematic) — a heavy wide body moves differently even at the same beta
3. Adapts online to body changes without retraining

**This is fundamentally different from all current input-feature engineering approaches (beta concat, embedding, physics features)** and may explain why all input-feature approaches plateau at ≈0.84 — the bottleneck is not what features are fed in, but whether the policy can differentiate dynamics from them.

---

## Section 10: Synthesis — Concrete Next Steps Ranked by Likely Impact

### High-Priority Experiments

**1. Body-Shape-Aware Curriculum (MDS-based)** — *Low implementation cost*

Compute Motion Difficulty Score (arXiv 2512.07248) for all 1024 clips. Start training on easy motions × average shapes, progressively expand to hard motions × all shapes. Expected to directly address the 65 floor-contact clip failures without architectural changes.

Implementation: add difficulty schedule to the motion sampler in `components/motion_lib.py`.

**2. Per-Shape Teacher Distillation** — *Medium implementation cost*

Train 10–20 teacher policies on shape clusters (k-means in beta space, or stratified by height/mass), then distill into one student (à la InterMimic, UniLegs). Each teacher specializes without competing with other shapes. Distillation then achieves the "single policy" goal.

Expected to overcome the 0.84 ceiling since each teacher can specialize. Use asymmetric actor-critic during distillation (critic sees beta; actor sees proprioception only).

**3. Body Transformer (BoT) Architecture** — *Medium implementation cost*

Replace the flat MLP with a BoT (arXiv 2408.06316) using SMPL kinematic adjacency as the attention mask. Inject per-joint physical attributes (limb length, body segment mass, cross-sectional width from beta) as node features alongside proprioceptive observations. Shape information flows locally through the kinematic graph — avoids FiLM fanout bottleneck.

**4. Latent Morphology Inference (Contextual Modulation)** — *Medium implementation cost*

Replace raw beta input with a latent morphology context vector inferred from proprioceptive history (Xiong et al. ICML 2023, DMAP NeurIPS 2022). A small encoder (GRU or 1D conv) over the last 10–20 steps of (q, dq, contact forces) → 32-dim morphology latent. The latent conditions the policy. Unlike the previous FiLM experiment, the conditioning signal is richer (dynamics-aware) and appropriately dimensioned.

**5. Assistive Curriculum Force for Floor-Contact Motions** — *Low implementation cost*

Apply assistive external forces to specific body segments during training on the 65 hard clips (arXiv 2506.23125). For crawl/kneel/squat: apply upward assistive forces at pelvis/torso during floor-contact phase, reduce magnitude as reward improves. Targeted curriculum for the exact failure modes identified.

Implementation: Can be done via IsaacGym's `apply_rigid_body_force_tensors` on a per-env basis conditioned on motion type.

### Medium-Priority Experiments

**6. Hypernetwork Conditioning (HyperDistill)** — *Higher implementation cost*

Generate per-shape MLP weight perturbations from a hypernetwork conditioned on beta. More expressive than FiLM (generates full weight matrices). For 128 fixed shapes, hypernetwork outputs can be precomputed and cached. Addresses FiLM's fanout bottleneck.

**7. ZMP/Support-Polygon Observation Features** — *Low implementation cost*

Augment the 15 physics features with dynamic stability features: current ZMP position relative to support polygon, support polygon area. Relevant to the physics of kneeling/squatting and may give the policy explicit information about stability margin during ground contact transitions.

**8. Policy-Space Diffusion for per-Shape Specialization** — *Research-level effort*

For cases where per-shape policies are needed (extreme body shapes), train a diffusion model over policy parameter space conditioned on beta embeddings. Use to initialize per-shape fine-tuning from a good shared baseline, then distill back. Useful only if the 0.84 ceiling is due to fundamental conflicting gradients between shapes.

---

## Key Papers Summary Table

| Paper | Venue/Year | Core Relevance | Priority |
|---|---|---|---|
| PHC (Luo et al.) | ICCV 2023 | Multi-shape SMPL training, PMCP/MoE architecture | **High** |
| PULSE (Luo et al.) | ICLR 2024 | Latent motion space for hierarchical control across shapes | **High** |
| HUMOS (Tripathi et al.) | ECCV 2024 | Body-shape-conditioned motion model, ZMP physics losses | **High** |
| Body Transformer (BoT) | arXiv 2408.06316 | Local kinematic graph attention with per-joint shape features | **High** |
| MDS Curriculum | arXiv 2512.07248 | Motion difficulty scoring and curriculum scheduling | **High** |
| InterMimic | arXiv 2502.20390 | Per-shape teacher → single student distillation | **High** |
| Contextual Modulation (Xiong et al.) | ICML 2023 | Latent morphology inference from proprioception history | **High** |
| DMAP (amathislab) | NeurIPS 2022 | Distributed attention for implicit morphology inference | Medium |
| HyperDistill | arXiv 2402.06570 | Hypernetwork for per-shape weight generation | Medium |
| Assistive Curriculum Force | arXiv 2506.23125 | Curriculum assistive forces for hard floor-contact motions | Medium |
| URMA / One Policy | CoRL 2024 | Shape-aware encoder-core-decoder separation | Medium |
| UniLegs | arXiv 2507.22653 | Asymmetric actor-critic morphology-agnostic distillation | Medium |
| HiFAR curriculum | arXiv 2502.20061 | Multi-stage curriculum for ground-contact motions | Medium |
| Policy-Space Diffusion | ACM ToG 2025 | Diffusion over policy params for shape adaptation | Medium |
| MetaMorph | ICLR 2022 | Morphology as positional embedding in Transformer | Low |
| MaskedMimic | SIGGRAPH Asia 2024 | Masked inpainting for robust contact-rich motion | Low |
| SMPLOlympics | NeurIPS 2024 | SMPL body shape RL benchmark | Low |
| Learning to Get Up (morphologies) | arXiv 2512.12230 | Diverse morphology coverage for zero-shot transfer | Low |
| Embedding Morphology into Transformers | arXiv 2603.00182 | Per-joint attribute conditioning in VLA transformers | Low |
| Fast Adaptation (BFM) | arXiv 2504.07896 | Few-episode online adaptation to new morphologies | Low |
| Neural Motion Retargeting | arXiv 2509.15443 | Body-shape-aware retargeting for consistent reference motions | Low |

---

## Key Takeaway: Why All Input-Feature Approaches Plateau at 0.84

The literature provides a plausible explanation for the consistent 0.84 ceiling across raw beta, shape embedding, and physics features:

**The bottleneck is not what features are given, but whether the policy can act on them.** All three approaches give the trunk a global conditioning vector. For a motion imitation task where most motions are upright locomotion (where body shape has minimal impact), a flat MLP can always find an average strategy that works across all shapes — this is the 0.84 baseline. The conditioning vector only helps if the policy can differentiate *behavior* based on shape, but the gradient signal for this differentiation is weak when 90%+ of training motions (locomotion) are solved similarly across shapes.

The 65 floor-contact clips require meaningfully different behavior per shape, but they are too few (6.3% of clips) to drive a 11/15-dim global conditioning vector to carry the right information. This is why curriculum (concentrate learning signal on hard clips) and teacher distillation (isolate per-shape learning) are likely to actually break the ceiling, while input feature engineering cannot.

---

## References

- PHC: https://github.com/ZhengyiLuo/PHC
- PULSE: https://arxiv.org/abs/2310.04582
- HUMOS: https://arxiv.org/html/2409.03944v1
- Body Transformer: https://arxiv.org/abs/2408.06316
- MDS Curriculum: https://arxiv.org/abs/2512.07248
- InterMimic: https://arxiv.org/pdf/2502.20390
- Contextual Modulation: https://arxiv.org/abs/2302.11070
- DMAP: https://arxiv.org/abs/2209.14218
- HyperDistill: https://arxiv.org/abs/2402.06570
- Assistive Curriculum Force: https://arxiv.org/html/2506.23125
- URMA: https://arxiv.org/pdf/2409.06366
- UniLegs: https://arxiv.org/pdf/2507.22653
- HiFAR: https://arxiv.org/pdf/2502.20061
- Policy-Space Diffusion: https://dl.acm.org/doi/full/10.1145/3732285
- MetaMorph: https://arxiv.org/abs/2203.11931
- SMPLOlympics: https://arxiv.org/abs/2407.00187
- Learning to Get Up (morphologies): https://arxiv.org/abs/2512.12230
- Embedding Morphology in Transformers: https://arxiv.org/pdf/2603.00182
- Fast Adaptation (BFM): https://arxiv.org/pdf/2504.07896
- Neural Motion Retargeting: https://arxiv.org/pdf/2509.15443
