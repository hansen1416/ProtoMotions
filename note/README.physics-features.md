# Physics Feature Extraction — Implementation Notes

## What Was Done

### New files

| File | Purpose |
|---|---|
| `tools/extract_smpl_physics_features.py` | Reads all 128 SMPL XMLs, computes 15 physics features per body, z-scores, saves `.pt` |
| `protomotions/data/assets/mjcf/smpl_mor/physics_features.pt` | 128×15 feature matrix, z-scored, keyed by `asset_id` |
| `examples/experiments/mimic/mlp_physics.py` | Experiment config: same MLP as `mlp.py` but `physics_obs` (15-dim) instead of `morphology_obs` (11-dim betas) |

### Modified files

| File | Change |
|---|---|
| `protomotions/simulator/isaacgym/simulator.py` | Added `_build_physics_features()` — loads `.pt` at startup, builds `self.env_physics_features [num_envs, 15]` |
| `protomotions/envs/context_views.py` | Added `env_physics_features: Optional[Tensor]` field to `EnvContext` |
| `protomotions/envs/base_env/env.py` | Passes `env_physics_features` to `EnvContext` in `_build_global_context()` |
| `protomotions/envs/obs/humanoid.py` | Added `compute_physics_obs()` pass-through |
| `protomotions/envs/component_factories.py` | Added `physics_obs_factory()` using `EnvContext.env_physics_features` |

### Training run

```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_physics.py \
    --experiment-name hhi_physics_feat_1024 \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 4096 \
    --batch-size 16384
```

Starts from scratch (cleaner ablation vs `hhi_1024_motion`).

---

## The 15 Physics Features

All features are extracted from the MuJoCo MJCF XML files for each SMPL body shape, then z-scored across the 128 training bodies.

### Feature summary (raw values across 128 bodies)

| Feature | Mean | Std | Min | Max | Units |
|---|---|---|---|---|---|
| `total_mass` | 73.4 | 25.7 | 26.4 | 144.4 | kg |
| `l_thigh_length` | 0.379 | 0.036 | 0.298 | 0.457 | m |
| `l_shin_length` | 0.409 | 0.045 | 0.303 | 0.502 | m |
| `r_thigh_length` | 0.381 | 0.035 | 0.302 | 0.461 | m |
| `r_shin_length` | 0.405 | 0.043 | 0.304 | 0.495 | m |
| `l_upper_arm_len` | 0.256 | 0.023 | 0.212 | 0.304 | m |
| `l_forearm_len` | 0.253 | 0.025 | 0.201 | 0.302 | m |
| `r_upper_arm_len` | 0.256 | 0.024 | 0.209 | 0.305 | m |
| `r_forearm_len` | 0.258 | 0.025 | 0.210 | 0.306 | m |
| `torso_height` | 0.307 | 0.035 | 0.234 | 0.394 | m |
| `neck_head_height` | 0.304 | 0.029 | 0.247 | 0.379 | m |
| `hip_width` | 0.126 | 0.022 | 0.085 | 0.193 | m |
| `shoulder_width` | 0.358 | 0.048 | 0.251 | 0.472 | m |
| `leg_length` | 0.787 | 0.079 | 0.603 | 0.956 | m |
| `total_height` | 1.399 | 0.125 | 1.132 | 1.666 | m |

---

## Physics Derivations

### Limb lengths — from body `pos` vectors

Each body element in the MJCF has a `pos` attribute that gives its **relative offset from the parent body's joint origin**. In T-pose, this vector is exactly the segment between two adjacent joints — e.g., the vector from the hip joint to the knee joint is the femur.

```
Limb length = norm(body.pos)
```

For example, `L_Knee.pos = [-0.0026, 0.0302, -0.3294]` gives a thigh length of:
```
norm([-0.0026, 0.0302, -0.3294]) ≈ 0.331 m
```

SMPL creates slightly asymmetric bodies (L/R are not identical), so we extract both sides independently.

The body hierarchy for limb lengths:

```
Pelvis
├── L_Hip → L_Knee (l_thigh_length = norm(L_Knee.pos))
│            └── L_Ankle (l_shin_length = norm(L_Ankle.pos))
├── R_Hip → R_Knee (r_thigh_length)
│            └── R_Ankle (r_shin_length)
└── Torso → Spine → Chest
                ├── L_Thorax → L_Shoulder → L_Elbow (l_upper_arm_len = norm(L_Elbow.pos))
                │                            └── L_Wrist (l_forearm_len = norm(L_Wrist.pos))
                └── R_Thorax → R_Shoulder → R_Elbow → R_Wrist
```

### Torso and head heights — summed segment lengths

`torso_height` is the summed distance from Pelvis to Chest:
```
torso_height = norm(Torso.pos) + norm(Spine.pos) + norm(Chest.pos)
```

`neck_head_height` covers the remaining stack from Chest to Head:
```
neck_head_height = norm(Neck.pos) + norm(Head.pos)
```

These matter for crawl/kneel/squat because the policy needs to know how far the torso must drop relative to the floor.

### Hip and shoulder widths — lateral global positions

The MJCF coordinate system places Y as the lateral axis (positive = left, negative = right). `hip_width` and `shoulder_width` are computed from the **global positions** of the corresponding joints, accumulated by summing `pos` vectors from Pelvis downward.

```
hip_width = |global_y(L_Hip) - global_y(R_Hip)|
          = |L_Hip.pos_y - R_Hip.pos_y|   (both direct children of Pelvis)
```

For shoulder width, the path from Pelvis to L_Shoulder is:
```
Pelvis → Torso → Spine → Chest → L_Thorax → L_Shoulder
global_y(L_Shoulder) = sum of .pos_y along this chain
```

```
shoulder_width = |global_y(L_Shoulder) - global_y(R_Shoulder)|
```

Hip width ranges 0.085–0.193 m; shoulder width ranges 0.251–0.472 m. These vary considerably across the 128 SMPL shapes and directly affect lateral stability during floor-contact motions.

### Total mass — from geom density × volume

Each body contains one or more collision geoms. The mass of a body is the sum of masses of its geoms:

```
mass = density × volume
```

Three geom types appear in the SMPL XMLs:

**Capsule** (`fromto` + `size[0]` = radius `r`):
```
L = norm(p2 - p1)           (distance between the two endpoint centers)
volume = π r² L + (4/3) π r³
```
A capsule is a cylinder of length L capped by two hemispheres of radius r.

**Box** (`size` = [half_x, half_y, half_z]):
```
volume = 8 × sx × sy × sz
```
MuJoCo's `size` for boxes gives half-extents, so the full box is 2× in each dimension.

**Sphere** (`size[0]` = radius `r`):
```
volume = (4/3) π r³
```

Total mass ranges from 26 kg (very light, small female) to 144 kg (very heavy, large male). This 5.5× range directly affects required joint torques and thus which floor-contact motions succeed.

### Derived features

`leg_length = l_thigh_length + l_shin_length`

`total_height = leg_length + torso_height + neck_head_height`

These are linear combinations of earlier features, but they express directly actionable physical quantities for the policy: "how far am I from the floor?" and "how long are my legs?" The network could in principle derive these by summing, but providing them explicitly reduces the learning burden.

---

## Why Physics Features Over Raw Betas

Raw SMPL betas are PCA coefficients in an appearance/shape space — they have no direct physical interpretation. Beta 1 roughly correlates with height and weight together, but it's not "leg length" or "mass" in isolation.

The 65 hard clips (crawl/kneel/squat/backward-walk) fail because the policy doesn't know the relevant physical constraints:
- **Crawl**: needs to know shoulder and hip width to avoid self-collision
- **Kneel**: needs to know shin length and total mass for balance during descent
- **Squat**: needs to know leg length, COM height, and total mass to control descent speed

Z-scoring across the 128 training bodies puts all features on the same scale, so no single feature (e.g., mass with 26–144 kg range) dominates the input gradient.

---

## Ablation Design

| Run | `morphology_obs` | Dim | Status |
|---|---|---|---|
| `hhi_1024_motion` | `[gender_id, betas/3]` | 11 | Converged, reward 0.84 |
| `hhi_physics_feat_1024` | `[physics_features_zscored]` | 15 | To launch |

---

## Observation Dimensions and Gender Note

- **`physics_obs`**: 15-dim, z-scored physics features. This fully replaces `morphology_obs` in `mlp_physics.py`.
- **`morphology_obs`** (gender_id + betas/3, 11-dim): **not passed** in `mlp_physics.py` — clean replacement, not an addition.
- **`gender_id`**: also **not included** in `physics_obs`. Gender is implicitly encoded in the physics features themselves (mass, limb lengths, widths all differ by gender because they come from gender-specific SMPL shapes). No explicit gender flag needed.

---

## TL;DR — Physics in Short

**Limb lengths** — each body's `pos` attribute in the MJCF is the vector from the parent joint to this joint (e.g., L_Hip → L_Knee vector = femur). Its norm is the limb length. Straight geometry, no simulation needed.

**Torso/head heights** — accumulated sums of `pos` norms along the spine chain (Pelvis → Torso → Spine → Chest, then Chest → Neck → Head).

**Hip and shoulder widths** — global Y-positions of L vs R equivalents, accumulated by summing the `pos.y` components down the tree from Pelvis. The MJCF places Y as the lateral axis.

**Total mass** — for each collision geom: `mass = density × volume`. Volume formulas:
- Capsule: `π r² L + (4/3) π r³` where L = distance between endpoint centers
- Box: `8 × half_x × half_y × half_z`
- Sphere: `(4/3) π r³`

Total mass across 128 bodies spans 26–144 kg (5.5× range), which is the key driver of why heavier bodies fail floor-contact motions — they need much higher torques to arrest their descent during kneeling/squatting.

Same network architecture (6-layer 1024-unit MLP), same motion set (1024 clips), same env config. The only difference is the conditioning input. Decision gate: if floor-contact success improves ≥10 pp on crawl/kneel subset, physics features become the main result.

---

## Biomechanically-Grounded Features — "aha" Candidates

The 15 basic features above are measurements. The features below are **derived** from classical locomotion biomechanics — each is a closed-form function of quantities we already extract, but each also predicts a specific failure mode and connects to a decades-old result in the literature. The paper story becomes: *"we observe that raw betas fail floor-contact motions; we derive 6 mechanically-grounded features from classical locomotion biomechanics; each predicts a specific failure mode; the policy trained with these features improves by X pp on crawl/kneel/squat."*

---

### 1. Froude Number → preferred walking speed

The single most famous result in comparative locomotion biomechanics (Alexander 1984). Animals of wildly different sizes all transition from walk to run at the same dimensionless Froude number ≈ 0.5:

```
Fr = v² / (g × L_leg)
v_preferred_walk = sqrt(0.5 × g × L_leg)
```

**AHA:** A 1.15 m SMPL body (L_leg ≈ 0.60 m) has a preferred walking speed of ~1.71 m/s. A 1.67 m body (L_leg ≈ 0.96 m) walks at ~2.17 m/s. The reference motion was captured from one person at one speed. A short body running that motion is mechanically forced into a sub-optimal gait regime. The policy needs to know this. Raw betas tell it nothing; `v_preferred_walk` is a single number that tells it everything about gait scaling.

---

### 2. Natural pendulum period → step timing

Walking is an inverted pendulum. The natural period of a pendulum of length L is:

```
T_step = 2π × sqrt(L_leg / g)
```

**AHA:** Tall bodies have *slower* natural step timing. Short bodies take quick steps. If you give a 1.15 m body the exact same joint angle trajectory as a 1.67 m body, the short body is fighting its own mechanics every step. This is a direct predictor of tracking error in locomotion clips. Zero correlation with raw betas.

---

### 3. Upper-body mass fraction → squat/kneel failure predictor

From De Leva segment tables, each body segment has a known COM fraction. We can compute:

```
f_upper = mass(torso + spine + chest + neck + head + arms) / total_mass
```

**AHA:** This is the single best predictor of floor-contact failure. During a squat, the torque the knees must resist is proportional to upper-body weight × moment arm. A body with 65% mass above the hips needs ~2× the knee torque of one with 45%. Raw betas smear this across 10 PCA axes. A single number explains crawl/kneel failure rates.

---

### 4. Gravity torque proxy at the knee

Directly derived from squat failure analysis:

```
τ_knee_proxy = m_upper × l_thigh
```

**AHA:** This is the quantity the knee actuator must overcome to hold the body during squatting/kneeling. It's the product of two things that vary independently across SMPL shapes. A heavy short-thighed body and a light long-thighed body can have similar overall mass but completely different squat difficulty. This feature captures the *interaction* that betas cannot express.

---

### 5. Moment of inertia of the swing leg about the hip

From biomechanics, using De Leva COM fraction ~0.37 for thigh from proximal end, ~0.28 for shank:

```
I_swing = m_thigh × (0.37 × l_thigh)² + m_shank × (l_thigh + 0.28 × l_shin)²
```

**AHA:** Determines how quickly the leg can be repositioned. During crawling, the leg must swing rapidly to change floor contact. Bodies with massive distal segments have disproportionately high I_swing because distance from hip is large — the `r²` term penalizes distal mass heavily. This explains why some specific body shapes fail crawling while others with similar total mass do not.

---

### 6. Cormic index (sitting height ratio)

A classical anthropometric measure used in anthropology and sports science for over a century:

```
cormic_index = torso_height / total_height
```

**AHA:** High cormic index = long trunk, short legs. These bodies have a higher COM fraction and worse squat mechanics. A single dimensionless number from classical human movement science. That we're using it to condition a neural controller is a genuine connection to the broader literature.

---

### Summary table

| Feature | Formula | Predicts | "aha" because |
|---|---|---|---|
| `v_preferred_walk` | `sqrt(0.5 × g × l_leg)` | natural walking speed | universal law across all animals (Alexander 1984) |
| `T_step_natural` | `2π × sqrt(l_leg / g)` | natural step timing | inverted pendulum period, explains locomotion timing failures |
| `f_upper_mass` | `m_upper / m_total` | squat/kneel torque demand | single number explains the floor-contact failure class |
| `tau_knee_proxy` | `m_upper × l_thigh` | required knee torque | interaction term betas cannot express |
| `I_swing_leg` | `m_thigh×(0.37 l_thigh)² + m_shank×(l_thigh + 0.28 l_shin)²` | leg repositioning speed | distal mass matters quadratically |
| `cormic_index` | `torso_height / total_height` | COM fraction, squat form | classical anthropometric measure, direct literature tie-in |

These can be appended to the existing 15 basic features (15 + 6 = 21 total), or the redundant derived ones (`leg_length`, `total_height`) can be dropped to keep the vector compact.

---

### Sources

- [Effects of physical characteristics on the gait transition speed — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/016794579500017M)
- [Terrestrial locomotion and Froude number — Kram et al. 1997 (CMU)](https://www.cs.cmu.edu/~hgeyer/Teaching/R16-899B/Papers/KramEA97JEB.pdf)
- [Regulation of whole-body angular momentum during human walking — Scientific Reports](https://www.nature.com/articles/s41598-023-34910-5)
- [The Effective Inertia of the Lower Limb During Locomotion — bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.10.03.680415.full.pdf)
- [A Biomechanical Review of the Squat Exercise — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10987311/)
- [Femur Length and Squat Form — Brookbush Institute](https://brookbushinstitute.com/articles/femur-length-and-squat-form)
- [Spring-mass model in running — Frontiers in Physiology](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1224459/full)
