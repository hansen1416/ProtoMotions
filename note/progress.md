# Training Progress — Morphology-Conditioned Mimic Policy

All runs use `--robot-name smpl_mor --simulator isaacgym --motion-file humos_*.pt`.
The 128 SMPL body shapes (64 β-vectors × 2 genders) span total_mass 26–144 kg and total_height 1.13–1.67 m.

---

## 1. Baseline: Direct Beta Concatenation (`mlp.py`)

**Experiment name**: `local_test` (quick smoke test), full run `hhi_1024_motion`

**Command**:
```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16
```

Full-scale run (RunPod):
```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 4096 \
    --batch-size 16384
```

**Morphology input**: `morphology_obs` = `[gender_id, beta_1/3, …, beta_10/3]` — 11-dim, appended directly to the flat observation vector before the MLP trunk.

**Architecture**: Standard 6-layer 1024-unit MLP (`MLPWithConcat`). No special conditioning mechanism — the 11-dim morphology vector is just another group of floats in the input.

**Result**: Converged to reward ≈ 0.84. This is the **baseline** all other runs are compared against.

**Known failure modes**: 65 hard clips involving floor-contact motions (crawl, kneel, squat, backward-walk) fail persistently. Analysis documented in `README.failed-motions.md`. The root hypothesis is that raw PCA betas give the policy no explicit signal about the physical constraints that govern these motions (torso mass, leg length, COM height relative to floor).

**Status**: Converged. Used as the reference point for all ablations.

---

## 2. FiLM Conditioning (`mlp_film.py`)

**Experiment name**: `local_test`

**Command**:
```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_film.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16
```

**Motivation**: FiLM (Feature-wise Linear Modulation) conditions the trunk by predicting per-layer scale (γ) and shift (β) from the morphology input, rather than concatenating morphology into the obs. It has worked well in vision for domain adaptation.

**Architecture**: A small conditioner MLP (64→64 hidden units) consumes `morphology_obs` and produces `2 × num_layers × hidden_dim` values. The trunk's activations at each layer are modulated as `h_l = h_l × γ_l + β_l`.

**Why it failed**: Two compounding issues (see `README.film-fail.md`):

1. **Fanout bottleneck**: For a 6-layer × 1024-unit actor, the conditioner must produce `2 × 6 × 1024 = 12,288` outputs from a 64-unit network. This severe compression-to-expansion mismatch dilutes gradients across all conditioner outputs, making it extremely hard to train. Worsens with more motions (wider body shape distribution, same conditioner capacity).

2. **Multiplicative instability**: Trunk gradients at layer `l` are scaled by `γ_l`. If `γ_l` drifts from 1.0 early in training — which happens when the conditioner is poorly initialized or underfit — the effective learning rate becomes shape-dependent and unstable. Noisy gamma estimates per minibatch (diverse body shapes) amplify this instability and prevent the trunk from converging.

**Result**: Reward ≈ 0.40–0.45, roughly half the baseline. The trunk could not converge to a stable feature representation under multiplicative noise from the conditioner.

**Status**: **Stopped at 1d 17h.** Too much complexity, no path to recovery without architectural change.

---

## 3. Shape Embedding + Concat (`mlp_shape_embed.py`)

**Experiment name**: `local_test`

**Command**:
```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_shape_embed.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16
```

**Motivation**: FiLM's failure was architectural. Replace multiplicative conditioning with a simple learned projection: encode the 11-dim morphology into a 64-dim embedding via a shallow MLP, then concatenate with the observation before the trunk. This is analogous to a learned positional encoding — same information, more usable nonlinear form.

**Architecture**:
```
morphology_obs (11-dim: gender + betas/3)
    → Linear(→ 64) → SiLU
    → shape_embed (64-dim)
                        │
[main obs (400–600+ dim)] ──cat──→ standard 6-layer 1024-unit trunk → output
```

Key differences from FiLM:

| Property | FiLM | Shape Embed + Concat |
|---|---|---|
| Conditioner output size | 2 × 6 × 1024 = 12,288 | 64 |
| Trunk coupling | Multiplicative (γ × h + β) | Additive (concat) |
| Gradient stability | Trunk grads scaled by γ | Trunk grads unaffected |
| Conditioner trainability | Hard (massive fanout) | Easy (small projection) |

**Files**: `protomotions/agents/common/shape_embed_mlp.py` (`ShapeEmbedMLPConfig` + `ShapeEmbedMLP`), `examples/experiments/mimic/mlp_shape_embed.py`.

**Config knobs**:
- `cond_hidden_units`: encoder depth, e.g. `[64]` (1 layer) or `[64, 64]` (2 layers, ablate via `--overrides`)
- `cond_activation`: default `silu`
- `beta_norm_scale`: betas divided by this before encoding (default `3.0`)

**Result**: Performance almost identical to the baseline (reward ≈ 0.84). The nonlinear projection did not improve over raw concat within the training budget; the trunk appears to learn an equivalent representation either way.

**Conclusion**: The learned embedding is not harmful (unlike FiLM), but it adds parameters without clear benefit at this scale. The failure modes on floor-contact motions remain — the issue is not the morphology encoding mechanism, it's the input features themselves (raw betas carry no explicit physical meaning).

**Status**: **Stopped at 1d 19h.** Performance parity with baseline confirmed; no further upside expected from this direction.

---

## 4. Physics Features (`mlp_physics.py`)

**Experiment name**: `hhi_physics_feat_1024`

**Command** (local smoke test):
```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_physics.py \
    --experiment-name hhi_physics_feat_1024 \
    --motion-file /home/hlz/datasets/humos_proto/humos_8_offset.pt \
    --num-envs 8 \
    --batch-size 16
```

Full-scale run (RunPod):
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

**Motivation**: Raw betas are PCA coefficients in appearance space — they have no direct physical interpretation. The 65 hard floor-contact clips fail because the policy lacks explicit knowledge of the physical constraints involved:
- **Crawl**: needs shoulder and hip width to avoid self-collision
- **Kneel**: needs shin length and total mass for balance during descent
- **Squat**: needs leg length, COM height, and total mass to control descent speed

Replace `morphology_obs` (11-dim, betas) with `physics_obs` (15-dim, z-scored physics features extracted from each body's MJCF). Gender is **not** passed explicitly — it is implicitly encoded in the physics features since mass, limb lengths, and widths all differ by gender.

**Architecture**: Same 6-layer 1024-unit MLP as baseline. Only the conditioning input changes (15-dim physics features instead of 11-dim betas). Starts from scratch — clean ablation.

**The 15 physics features** (z-scored across 128 training bodies):

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

Features are extracted from MJCF geometry (body `pos` vectors and geom volumes) — no simulation needed. See `README.physics-features.md` for full derivation details.

**New files**:
- `tools/extract_smpl_physics_features.py` — reads all 128 SMPL XMLs, computes 15 features, z-scores, saves `.pt`
- `protomotions/data/assets/mjcf/smpl_mor/physics_features.pt` — 128×15 feature matrix, keyed by `asset_id`
- `examples/experiments/mimic/mlp_physics.py` — experiment config

**Modified files**:
- `protomotions/simulator/isaacgym/simulator.py` — `_build_physics_features()`, builds `self.env_physics_features [num_envs, 15]`
- `protomotions/envs/context_views.py` — `env_physics_features: Optional[Tensor]` field added to `EnvContext`
- `protomotions/envs/base_env/env.py` — passes `env_physics_features` to `EnvContext` in `_build_global_context()`
- `protomotions/envs/obs/humanoid.py` — `compute_physics_obs()` pass-through
- `protomotions/envs/component_factories.py` — `physics_obs_factory()` using `EnvContext.env_physics_features`

**Ablation summary**:

| Run | Conditioning input | Dim | Result |
|---|---|---|---|
| `hhi_1024_motion` | `[gender_id, betas/3]` | 11 | Converged, reward ≈ 0.84 **(baseline)** |
| `hhi_physics_feat_1024` | `[physics_features_zscored]` | 15 | Training underway — early reward similar to baseline |

**Decision gate**: If floor-contact success (crawl/kneel subset) improves ≥10 pp over baseline, physics features become the main result. Otherwise the approach is neutral — same overall reward, different feature semantics.

**Status**: **Training underway.** Overall reward trajectory looks similar to baseline so far. Floor-contact evaluation pending convergence.

**Potential next step**: If basic physics features show no improvement, the next candidates are 6 biomechanically-grounded derived features (Froude number, natural pendulum period, upper-body mass fraction, gravity torque proxy at knee, swing leg moment of inertia, cormic index). Each predicts a specific failure mode from classical locomotion biomechanics. Details below.

---

### Physics Derivations

#### Limb lengths — from body `pos` vectors

Each body element in the MJCF has a `pos` attribute giving its **relative offset from the parent body's joint origin**. In T-pose, this vector is exactly the segment between two adjacent joints.

```
limb_length = norm(body.pos)
```

Example: `L_Knee.pos = [-0.0026, 0.0302, -0.3294]` → thigh length ≈ 0.331 m.

SMPL creates slightly asymmetric bodies (L/R are not identical), so both sides are extracted independently.

Body hierarchy for limb lengths:
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

#### Torso and head heights — summed segment lengths

```
torso_height     = norm(Torso.pos) + norm(Spine.pos) + norm(Chest.pos)
neck_head_height = norm(Neck.pos)  + norm(Head.pos)
```

These matter for crawl/kneel/squat because the policy needs to know how far the torso must drop relative to the floor.

#### Hip and shoulder widths — lateral global positions

The MJCF coordinate system places Y as the lateral axis (positive = left, negative = right). Widths are computed from **global positions** accumulated by summing `pos` vectors from Pelvis downward.

```
hip_width      = |global_y(L_Hip) - global_y(R_Hip)|
               = |L_Hip.pos_y - R_Hip.pos_y|   (both direct children of Pelvis)

shoulder_width = |global_y(L_Shoulder) - global_y(R_Shoulder)|
```

For L_Shoulder the path is: Pelvis → Torso → Spine → Chest → L_Thorax → L_Shoulder (sum `.pos_y` along the chain).

#### Total mass — from geom density × volume

Each body contains one or more collision geoms; body mass = sum of geom masses (`mass = density × volume`).

Three geom types in the SMPL XMLs:

**Capsule** (`fromto` + `size[0]` = radius `r`):
```
L      = norm(p2 - p1)
volume = π r² L + (4/3) π r³
```

**Box** (`size` = [half_x, half_y, half_z]):
```
volume = 8 × sx × sy × sz
```
MuJoCo `size` gives half-extents, so the full box is 2× in each dimension.

**Sphere** (`size[0]` = radius `r`):
```
volume = (4/3) π r³
```

Total mass spans 26–144 kg (5.5× range). This directly affects required joint torques and explains why heavier bodies fail floor-contact motions.

#### Derived features

```
leg_length   = l_thigh_length + l_shin_length
total_height = leg_length + torso_height + neck_head_height
```

Linear combinations of earlier features, but they express directly actionable quantities for the policy. The network could in principle derive these by summing, but providing them explicitly reduces the learning burden.

---

### Biomechanically-Grounded Derived Features (6 candidates)

If the 15 basic features show insufficient improvement, these derived features are grounded in classical locomotion biomechanics — each is a closed-form function of the quantities above, and each predicts a specific failure mode.

#### 1. Froude number → preferred walking speed

The single most famous result in comparative locomotion biomechanics (Alexander 1984). Animals of all sizes transition from walk to run at Froude ≈ 0.5:

```
v_preferred_walk = sqrt(0.5 × g × l_leg)
```

**Why it matters**: A 1.15 m SMPL body (l_leg ≈ 0.60 m) has a preferred walking speed of ~1.71 m/s; a 1.67 m body walks at ~2.17 m/s. The reference motion was captured from one person at one speed — a short body running that motion is mechanically forced into a sub-optimal gait regime. Raw betas tell the policy nothing about this; `v_preferred_walk` tells it everything about gait scaling in one number.

#### 2. Natural pendulum period → step timing

Walking is an inverted pendulum. The natural period of a pendulum of length L:

```
T_step = 2π × sqrt(l_leg / g)
```

**Why it matters**: Tall bodies have slower natural step timing. Short bodies take quick steps. Giving a 1.15 m body the exact same joint angle trajectory as a 1.67 m body forces it to fight its own mechanics every step — a direct predictor of tracking error in locomotion clips.

#### 3. Upper-body mass fraction → squat/kneel failure predictor

```
f_upper = mass(torso + spine + chest + neck + head + arms) / total_mass
```

**Why it matters**: During a squat, the torque the knees must resist is proportional to upper-body weight × moment arm. A body with 65% mass above the hips needs ~2× the knee torque of one with 45%. Raw betas smear this across 10 PCA axes. A single number explains crawl/kneel failure rates.

#### 4. Gravity torque proxy at the knee

```
τ_knee_proxy = m_upper × l_thigh
```

**Why it matters**: The quantity the knee actuator must overcome to hold the body during squatting/kneeling. A heavy short-thighed body and a light long-thighed body can have similar total mass but completely different squat difficulty. This feature captures the *interaction* that betas cannot express.

#### 5. Moment of inertia of the swing leg about the hip

Using De Leva COM fractions (~0.37 for thigh from proximal end, ~0.28 for shank):

```
I_swing = m_thigh × (0.37 × l_thigh)² + m_shank × (l_thigh + 0.28 × l_shin)²
```

**Why it matters**: Determines how quickly the leg can be repositioned. During crawling, the leg must swing rapidly to change floor contact. Bodies with massive distal segments have disproportionately high `I_swing` because the `r²` term penalizes distal mass heavily — explaining why some specific body shapes fail crawling while others with similar total mass do not.

#### 6. Cormic index (sitting height ratio)

A classical anthropometric measure used in sports science for over a century:

```
cormic_index = torso_height / total_height
```

**Why it matters**: High cormic index = long trunk, short legs → higher COM fraction → worse squat mechanics. A single dimensionless number from human movement science; connects to the broader literature directly.

#### Summary

| Feature | Formula | Predicts | Why non-obvious |
|---|---|---|---|
| `v_preferred_walk` | `sqrt(0.5 × g × l_leg)` | natural walking speed | universal law across all animals (Alexander 1984) |
| `T_step_natural` | `2π × sqrt(l_leg / g)` | natural step timing | inverted pendulum period, explains locomotion timing failures |
| `f_upper_mass` | `m_upper / m_total` | squat/kneel torque demand | single number explains the floor-contact failure class |
| `tau_knee_proxy` | `m_upper × l_thigh` | required knee torque | interaction term betas cannot express |
| `I_swing_leg` | `m_thigh×(0.37 l_thigh)² + m_shank×(l_thigh + 0.28 l_shin)²` | leg repositioning speed | distal mass matters quadratically |
| `cormic_index` | `torso_height / total_height` | COM fraction, squat form | classical anthropometric measure, direct literature tie-in |

Can be appended to the 15 basic features (15 + 6 = 21 total), or redundant derived ones (`leg_length`, `total_height`) can be dropped to keep the vector compact.

---

### References

- [Effects of physical characteristics on the gait transition speed — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/016794579500017M)
- [Terrestrial locomotion and Froude number — Kram et al. 1997 (CMU)](https://www.cs.cmu.edu/~hgeyer/Teaching/R16-899B/Papers/KramEA97JEB.pdf)
- [Regulation of whole-body angular momentum during human walking — Scientific Reports](https://www.nature.com/articles/s41598-023-34910-5)
- [The Effective Inertia of the Lower Limb During Locomotion — bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.10.03.680415.full.pdf)
- [A Biomechanical Review of the Squat Exercise — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10987311/)
- [Femur Length and Squat Form — Brookbush Institute](https://brookbushinstitute.com/articles/femur-length-and-squat-form)
- [Spring-mass model in running — Frontiers in Physiology](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1224459/full)

---

## Summary Table

| Approach | Input dim | Architecture change | Reward | Duration | Outcome |
|---|---|---|---|---|---|
| Raw beta concat (`mlp.py`) | 11 | None | ≈ 0.84 | Converged | **Baseline** |
| FiLM (`mlp_film.py`) | 11 | Multiplicative conditioning | ≈ 0.40–0.45 | Stopped 1d 17h | Failed — fanout + instability |
| Shape embed (`mlp_shape_embed.py`) | 11 → 64 embed | Learned projection + concat | ≈ 0.84 | Stopped 1d 19h | Neutral — no gain over baseline |
| Physics features (`mlp_physics.py`) | 15 | None (input swap) | ≈ 0.84 (early) | **Ongoing** | Pending floor-contact eval |
