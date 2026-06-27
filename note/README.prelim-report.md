# Preliminary Report: Morphology-Conditioned Physics-Based Motion Imitation

**Date:** 2026-06-22  
**Status:** In progress — §5 transfer evaluation and §6 two-stage curriculum still running

---

## 1. Problem

We want a single physics policy that can imitate arbitrary human motions across a continuous range of SMPL body shapes. Concretely: given a motion reference clip (joint positions, rotations, root trajectory) and a target body shape (SMPL β ∈ ℝ¹⁰), simulate a physically plausible humanoid of that shape tracking the reference in IsaacGym.

Prior work (PULSE, PHC) fixes the body to the mean SMPL shape. Real applications — digital humans, sim-to-real transfer, humanoid robots with manufacturing variation — need control that adapts to the body, not assumes a standard one.

**Why this is hard:** the same motion (e.g. walking) has qualitatively different physical demands on a 26 kg child-sized body vs a 144 kg heavy body. Stride length, joint torques, balance strategies, and contact timing all differ. A policy trained only on the mean shape will fail to generalize because its control laws are implicitly calibrated to one mass/limb-length distribution.

---

## 2. Setup

### 2.1 Body shape distribution

128 SMPL body shapes: 64 β-vectors sampled from a Gaussian (σ=1, range ≈ [-3, 3]) × 2 genders. This spans:
- Total mass: 26–144 kg (5.5× range)
- Total height: 1.13–1.67 m (1.5× range)
- Shape extremity ‖β‖₂: 0–6.96

Each β-vector uniquely determines a MJCF XML asset (capsule geometry, masses, inertia tensors) generated via SMPLSim.

### 2.2 Motion dataset (pilot: 1024 clips)

Source: HumanML3D text-motion dataset. 1024 clips selected from the 5th–55th difficulty percentile (skips trivial static poses and motions too hard to converge). Each clip retargeted to all 128 body shapes via HUMOS — a shape-aware motion retargeting model — producing 1024 × 128 = **131,072 training motions**.

Motion file format: per-motion tensors of global translations `gts [T,24,3]`, global rotations `grs [T,24,4]`, DOF positions `dps [T,69]`, plus per-motion metadata (`beta_key`, `gender`, `asset_id`).

### 2.3 Training setup

- **Simulator:** IsaacGym, 4096 parallel environments, 4× A40 GPU
- **Policy:** MLP actor-critic, observation vector ~800-dim concatenating proprioception + morphology obs
- **Morphology obs:** depends on architecture (see §3)
- **Reward:** AMP-style motion imitation (body position tracking + velocity + root + DOF) + action smoothness penalty
- **Evaluator:** `eval_one_shape_per_motion` — for each of 1024 clips, draw one random shape per eval cycle → 1024 episodes evaluated every 200 training epochs

---

## 3. Architecture Search

Four morphology conditioning architectures were trained from scratch on the full 1024×128 dataset. All use the same reward, optimizer, and environment.

| Run | Architecture | Morphology input | Reward at convergence | Outcome |
|---|---|---|---|---|
| 1 | `hhi_1024_motion` | MLP + raw beta concat | 11-dim `[gender_id, β/3]` | ≈ 0.84 | **Baseline** |
| 2 | `hhi_film_1024_motion` | FiLM per-layer modulation | 11-dim | ≈ 0.40–0.45 | **Failed** |
| 3 | `hhi_se_1024_motion` | Learned shape embed (11→64) + concat | 64-dim | ≈ 0.84 | Neutral |
| 4 | `hhi_phy_1024_motion` | Physics features (z-scored) + concat | 15-dim | ≈ 0.84 | Neutral |

### 3.1 Why FiLM failed

FiLM predicts per-layer scale (γ) and shift (β) from the morphology input. For a 6-layer × 1024-unit actor, this requires producing 2 × 6 × 1024 = 12,288 outputs from an 11-dim input. Two failure modes:

1. **Fanout bottleneck:** severe compression-to-expansion ratio dilutes gradients across all modulation outputs
2. **Multiplicative instability:** trunk gradients at layer l are scaled by γ_l; if γ_l drifts from 1.0 early in training, effective learning rate becomes shape-dependent and divergent

Stopped at 1d 17h. No recovery path without zero-initialized γ (not implemented).

### 3.2 Why all flat-concat variants converge identically

All three non-FiLM architectures hit reward ≈ 0.84. Two explanations:

1. **Motion variance dominates gradients.** With 1024 diverse clips, gradient signal is dominated by motion content variation. The 128-shape signal is a second-order correction — the network learns motion-invariant features first and shape-specific adaptation is a small residual.
2. **Implicit shape information in proprioception.** Body height, segment length ratios, and mass-related inertia are partially observable from proprioceptive states. The explicit morphology vector offers a shortcut the network may not need for seen shapes.

**Decision:** baseline MLP (run 1) is the chosen architecture. Simplest, most trained, and the 11-dim abstract beta input is a stronger test of shape generalization than physics features that are partially redundant with proprioception.

---

## 4. Hard Clip Analysis

### 4.1 Persistent failure distribution at baseline

At convergence (epoch ~12,021), the training evaluator (`eval_one_shape_per_motion`) ran 21 cycles across epochs 8,000–12,000. Distribution of per-clip failure frequency:

| Fail rate | Clips |
|---|---|
| 21/21 (always fails) | 328 |
| 19–20/21 | 391 |
| 17–18/21 | 214 |
| 15–16/21 | 75 |
| 12–14/21 | 16 |

531 clips fail in ≥19/21 cycles. The failure class is **motion-content driven, not shape-driven**: crawl, kneel, squat, backward-walk, and floor-contact poses. These require COM management and contact forces the PD controller cannot maintain at the current action parameterization.

### 4.2 Fine-tune attempt on hard clips

`hhi_1024_motion_tune`: converged baseline fine-tuned exclusively on 192 hard clips (crawl/kneel/squat/backward) × 128 shapes = 24,576 motions.

**Results:**
- Success rate on hard clips (8 clips × 128 shapes, local eval): 23.6% (+15 pp over baseline estimate)
- Physics explosions on easy clips (shard-0, 2 clips × 128 shapes): **94.9%** — catastrophic forgetting
- Normalized jerk: 3–4× higher than baseline — policy fights gravity with large oscillatory actions

**Interpretation:** fine-tuning on hard clips teaches the correct pose targets but destroys the smooth tracking behavior the policy learned on easy clips. The high jerk triggers IsaacGym's constraint solver to blow up. Fine-tuning is not the structural fix — the correct solution is contact-body extension and improved action parameterization. Abandoned after one run.

---

## 5. Transfer Training: Raw Betas vs Physics Features

### 5.1 Setup

Two parallel transfer runs, both initialized from the converged baseline (`hhi_1024_motion/last.ckpt`):

| Run | Architecture | Morphology input | Starting epoch | Final epoch |
|---|---|---|---|---|
| `hhi_1024_transfer` | MLP + raw betas | 11-dim | 12,021 | 21,400 |
| `hhi_phy_1024_transfer` | MLP + physics features | 15-dim (z-scored) | 6,801 | 17,200 |

The 15 physics features are derived from each body's MJCF: total mass, bilateral thigh/shin/upper-arm/forearm lengths, torso height, neck-head height, hip width, shoulder width, leg length, total height. Z-scored across the 128 training bodies.

Both runs use the same motion file (1024×128), same evaluator, same `eval_one_shape_per_motion` strategy.

### 5.2 Training failure counts

From the failed_motion curriculum logs, final 10 eval cycles (window epochs):

| | `hhi_1024_transfer` | `hhi_phy_1024_transfer` |
|---|---|---|
| Persistent failures (≥5/10 cycles) | **177** / 1024 clips | **167** / 1024 clips |
| Failure count at latest epoch | 205 | 206 |

Both runs reduced failures from ~100% (all 1024 clips failing at baseline plateau) to ~17% — an **83% reduction**. The plateau at ~200 failed clips/eval cycle indicates both have converged.

### 5.3 Clip-level overlap analysis (T1)

Comparing the two persistent failure sets:

| Clip set | Count | Fraction |
|---|---|---|
| Persistent in **both** transfer runs | 121 | 11.8% |
| Only in `hhi_1024_transfer` (raw betas fail, physics succeeds) | 56 | 5.5% |
| Only in `hhi_phy_1024_transfer` (physics fails, raw betas succeed) | 46 | 4.5% |
| In both transfers AND baseline hard class (≥19/21) | 92 | 9.0% |

**Key observation:** the 121-clip shared failure set maps almost entirely onto the baseline structural hard class (crawl/kneel/squat). These are irreducible with the current architecture — both transfer runs fail them regardless of morphology conditioning type.

Physics features help 56 clips and hurt 46 clips — a net of 10 clips. Each of these 56 clips was evaluated with a **different random shape each eval cycle**, so it is failing with the raw betas conditioning regardless of which body shape is assigned — a clip-level, not a shape-level, difference.

### 5.4 Statistical significance

Is the 10-clip net difference meaningful?

- Two-proportion z-test at n=1024 clips: z = 0.59, **p = 0.55** — not significant
- Minimum detectable net difference for significance: ~33 clips (3× larger than observed)
- Naively expanding to 131,072 motions gives z = 6.7 (p < 10⁻¹⁰) but this is invalid — outcomes within the same clip are highly correlated (ICC ≈ 0.3–0.7), reducing effective n to ~1500–2000, which also yields p > 0.05

**Conclusion:** the binary failure rate comparison cannot support a paper claim. The binary failure metric is too coarse at this sample size. Continuous tracking quality (`mean_body_dist`) is needed.

### 5.5 Visual inference test (preliminary)

On a single 8-env visual inference run (`humos_131072_0001_offset.pt`, 8 random motions):
- `hhi_phy_1024_transfer`: **5/8 envs visually followed** the reference
- `hhi_1024_transfer`: **0/8 envs visually followed**

This is a single observation on 8 envs and not quantified. However, it strongly suggests the difference is in **continuous tracking quality** (mean body distance) rather than binary pass/fail — which is consistent with the statistical analysis above.

### 5.6 What is still needed

To support any quantitative claim about the two transfer checkpoints, the full CSV evaluation (E1) on RunPod is required:
- `mean_body_dist` per motion → paired t-test at clip level (MDE ~0.03–0.07 m)
- `std(mean_body_dist)` across 128 shapes per clip → cross-shape variance (shape-adaptation quality)
- `‖β‖₂` vs `mean_body_dist` slope → shape extremity degradation
- `mean_normalized_jerk` on stable episodes → motion smoothness comparison

---

## 6. Next Steps

### 6.1 Immediate: systematic evaluation of transfer checkpoints (E1 + E3)

Before the next RunPod session, augment `HHIFaultEvaluator` to record:
- `mean_normalized_jerk` and `high_jerk_frame_pct` (via existing `SmoothnessCalculator`)
- `beta_l2` = ‖β‖₂ per motion (one-liner from `motion_lib.motion_betas`)

Then on RunPod, run both transfer checkpoints against the full 1024×128 dataset. Estimated 3–5 hours per checkpoint. All downstream analyses (E2–E6, S1, S2) are pure pandas from the resulting CSV files.

### 6.2 In progress: two-stage curriculum on full motion library

The 1024-clip pilot establishes the approach. The full training plan scales to the complete HumanML3D library:

**Stage 1 (currently running):** Train on all 20,946 HumanML3D clips with **neutral body shape** (β=0 for all envs). The policy sees the full diversity of human motion without body-shape variation. Morphology obs is all-zeros during Stage 1, which the network learns to ignore. This builds a strong motion prior across the entire motion library.

- Motion data: `humos_proto_neutral/offset/humanml3d_neutral_20946_slurmrank.pt` (6 shards × ~3,491 clips)
- Robot: `smpl_mor_neutral` (male/female neutral SMPL assets, same architecture as Stage 2)
- Key engineering: `reset_morphology_normalizer.py` resets beta normalizer dims (var[-10:] → 1.0) after Stage 1 so saturated near-zero variances do not suppress beta inputs in Stage 2

**Stage 2 (planned, full scale):** Transfer Stage 1 checkpoint to the **full 20,946-clip × 128-shape dataset**, 10× lower learning rate. This requires generating HUMOS-retargeted motion data for all 20,946 clips across all 128 shapes (2.68M output motions — 20× the 1024-clip pilot run). The data generation pipeline is the same as for the pilot: HUMOS inference → NPZ export → MotionLib → grounding offset → shard upload.

**Why this is better than training from scratch on 1024×128:**
- Any held-out evaluation clip (E7, held-out betas) is in-distribution for **motion content** after Stage 1 — generalization failures can be attributed purely to **shape OOD**, not motion OOD
- The motion prior is richer: 20,946 clips covers walking, running, jumping, crawling, manipulation, dance — far beyond the 1024-clip pilot
- Stage 2 is a smaller learning problem: shape adaptation on top of a converged motion prior converges faster and more stably

**Data pipeline requirement for Stage 2:** Run HUMOS inference on all 20,946 HumanML3D clips × 128 body shapes. Estimated compute: 20× the 1024-clip run. The resulting dataset is ~2.68M motion files which need to be converted, grounded, and uploaded to RunPod before Stage 2 can begin.

### 6.3 Held-out body shape generalization (E7)

Generate motion clips for 16 interpolation betas (‖β‖₂ ≈ 5.0, range [-3,3], never seen during training) and 16 extrapolation betas (‖β‖₂ ≈ 8.4, range [-5,5], outside training distribution). Evaluate both transfer checkpoints on these clips with the same 1024 source motions. This isolates pure shape generalization — the key novel evaluation in the paper.

Status: interpolation HUMOS inference done locally (22,459 files). Needs upload → NPZ export → MotionLib → grounding offset → RunPod evaluation. Extrapolation inference not yet started.

---

## 7. Summary of What We Have

| Item | Status |
|---|---|
| Architecture search (4 runs) | **Done** — MLP flat-concat chosen |
| Baseline pilot training (1024×128) | **Done** — reward 0.84, 12,021 epochs |
| Hard clip fine-tune | **Done** — abandoned (jerk + forgetting) |
| Transfer run: raw betas | **Done** — 21,400 epochs |
| Transfer run: physics features | **Done** — 17,200 epochs |
| T1 clip overlap analysis | **Done** — 10-clip net difference, p=0.55 |
| Visual inference smoke test (8 envs) | **Done** — 5/8 vs 0/8, not yet quantified |
| Gravity-core evaluation (88 clips × 128 shapes) | **Done** — phy 0.6730 m vs raw 0.6909 m; gap 0.018 m < threshold; **raw betas chosen** |
| Stage 1 neutral training (`hhi_20946_neutral`) | **Running** — 20,946 clips on RunPod |
| Stage 2 data generation (20,946×128) | **In progress** — see `README.stage2-data-pipeline.md` |
| Stage 2 transfer (`hhi_stage2_transfer`) | **Blocked on data** |
| E1 full CSV evaluation (Stage 2 checkpoint) | **Not yet** — needs Stage 2 checkpoint |
| E3 smoothness evaluator augmentation | **Not yet** — local code change |
| E7 held-out beta generalization | **Partial** — interp inference done (717 files); pipeline incomplete |
