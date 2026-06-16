# 3-Month Research Plan
**Start:** 2026-06-12  **Deadline:** ~2026-09-12 (ICRA 2027 submission)

---

## Paper Narrative

> "We present the first physics-based motion imitation policy that generalizes across 128 SMPL body shapes. Physics-derived morphology features (mass, COM, limb lengths) unlock shape-adaptive physical control, particularly for floor-contact motions (crawl, kneel, squat) — the primary failure mode of raw shape-parameter conditioning. The policy develops an internal representation of body physics correlated with physical properties it was never explicitly trained to predict."

---

## Current State (2026-06-16)

| Item | Status |
|---|---|
| MLP baseline `hhi_1024_motion` | Converged, reward ~0.84 |
| FiLM `hhi_film_1024_motion` | Failed (~0.40–0.45), dropped |
| ShapeEmbed `hhi_se_1024_motion` | Converged, reward ~0.84 — identical to baseline, dropped |
| Physics features `hhi_physics_feat_1024` | Converged, reward ~0.84 — no gain over baseline |
| Hard clip fine-tune `hhi_1024_motion_tune` | Converged — success_rate ~0.80 (+15 pp), reward >0.90, jerk 3–4× baseline |
| Failed motion analysis | Done — 65 clips (crawl/kneel/squat/backward) with ≥8 betas failing |
| Data pipeline | Complete — 131,072 motions, 16 local shards |
| Evaluator | Complete — `eval_one_shape_per_motion` + clip-level weight propagation |
| Held-out betas (interpolation/extrapolation) | Not yet generated |

---

## What to Drop

- **FiLM** — failed; literature confirms it's the wrong tool for fixed-topology fixed-D conditioning
- **ShapeEmbed** — same empirical result as concat; weakens the story
- **SMPL-X validation** — not blocking for 1024-clip pilot scope
- **Fine-tuning efficiency (S5)** — low ICRA impact; only if weeks 1–9 finish early
- **Hardware validation** — out of scope

================================================================================

## Month-by-Month Timeline (updated 2026-06-16)

### Month 1 (June 12 – July 12) — Training Improvements

| Week | Tasks |
|---|---|
| **1–2** | Implement A1 (residual PD) + A2 (contact reward) + A3 (phase φ) — launch combined run |
| **3** | Monitor training; compare hard clip success rate vs `hhi_1024_motion_tune` |
| **4** | Generate held-out betas via HUMOS (B1); start rollout logging for torque/energy |

### Month 2 (July 12 – Aug 12) — Analysis Sprint

| Week | Tasks |
|---|---|
| **5** | Held-out generalization eval — interpolation + extrapolation on best checkpoint |
| **6** | Torque/energy analysis: same clip × all 128 shapes. COM trajectory + stability |
| **7** | Embodiment encoding probe (B2) — activations → physical property R² |
| **8** | Stride/retargeting analysis (B3). Motion category × shape heatmap. Failure mode taxonomy |

### Month 3 (Aug 12 – Sep 12) — Paper

| Week | Tasks |
|---|---|
| **9** | Scale-up run (Track C) if Track A shows clear gains. Begin paper draft (method section) |
| **10** | Results section + all figures. Qualitative video: side-by-side multi-shape |
| **11** | Introduction, related work, conclusion. Internal review |
| **12** | Buffer: polish, video narration, submission |

================================================================================

## Paper Figures (target list)

1. System diagram: HUMOS → motionlib → multi-shape sim → morphology-conditioned policy
2. Per-shape tracking performance across 128 training betas (bar/violin plot)
3. Held-out generalization: tracking error vs beta L2 norm from training distribution (interpolation + extrapolation curves)
4. Torque/energy vs body shape for a fixed locomotion clip ("heavier → higher torques" figure)
5. **Embodiment encoding probe:** R² per physical property from linear probe on activations — "AI learned physics" figure
6. Motion category × shape extremity heatmap (crawl/kneel hardest, locomotion robust)
7. Failure mode taxonomy: fall / COM drift / joint-limit breakdown across shape extremity buckets
8. Qualitative: same motion clip across 8 body shapes side-by-side (also video)
9. Ablation: baseline vs fine-tune vs Track A combined on overall reward and floor-contact success rate

---

## Key Files and References

| Item | Path |
|---|---|
| Failed clip analysis | `note/README.failed-motions.md` |
| Ranked failure list | `results/hhi_1024_motion/persistent_failures.txt` |
| Full research plan (literature) | `note/deep-research-report.md` |
| Paper outline | `note/README.paper.md` |
| HUMOS data notes | `note/README.humos-data.md` |
| Clip text annotations | `/home/hlz/repos/hhi/data-processing/motion_id_text.json` |
| Valid motions list | `/home/hlz/repos/hhi/data-processing/valid_motions.txt` |

================================================================================

## TRACK A — Training Improvements (Performance)

These directly target the hard clip failure class (crawl/kneel/squat/backward).
Note: early termination is already reference-relative (max joint error > 0.5 m) — no fix needed there.

---

### A1. Residual PD Control
**Priority: HIGH — fixes jerk in fine-tune AND reduces hard clip learning difficulty**

```
Current:  q_target = q_neutral + scale * action
Fix:      q_target = q_ref    + scale * action
```

At `action=0` the PD controller already tracks the reference pose. Policy only learns the balance correction delta, eliminating the large corrective actions from neutral posture that cause oscillation in `hhi_1024_motion_tune`. `q_ref` is already available as `EnvContext.mimic.ref_state.dof_pos`.

Validated by PHC on 11,313 AMASS clips. Cost: ~2–3 days.

---

### A2. Contact Reward for Knees and Hands
**Priority: HIGH — direct floor-contact supervision for the failure class**

`contact_match_rew_factory` exists but only tracks foot contacts. The 192 hard clips fail at knee and hand contacts (crawl, kneel). The policy can reach approximate joint angles for a kneeling pose without knees touching the floor and still collect full tracking reward — explicit contact supervision is needed.

Extend `contact_bodies`:
```python
contact_bodies = [
    "all_left_foot_bodies", "all_right_foot_bodies",
    "L_Knee", "R_Knee",        # kneel / crawl
    "L_Wrist", "R_Wrist",      # crawl
]
```

Cost: ~1–2 days.

---

### A3. Motion Phase Variable φ in Observation
**Priority: MEDIUM — resolves temporal aliasing in squat/kneel**

```
φ = frame_idx / total_frames  ∈ [0, 1]
```

Without φ, the policy cannot distinguish going-down from coming-up in a squat — both phases share identical joint angles but require opposite control effort. Same obs → different correct action = unresolvable aliasing. Used in PULSE and Bi-Level Motion Imitation.

Cost: ~1 day (one observation key + experiment file update).

---

### A4. Per-Shape RunningMeanStd
**Priority: MEDIUM — structural correctness for multi-shape training**

The shared `RunningMeanStd` averages statistics over all 128 shapes. A 26 kg body and a 144 kg body have different root height ranges, joint velocity magnitudes, and inertia-related accelerations — the shared normalizer miscalibrates inputs for every shape.

Fix: 128 separate `RunningMeanStd` buffers keyed by `asset_id`, each updated only from envs assigned to that shape. SimBa (2024): running normalization is the most impactful obs-processing choice in RL.

Cost: ~1 day.

---

### A5. PopArt Per-Shape Return Normalization
**Priority: MEDIUM — calibrates critic value targets per shape**

Different shapes have genuinely different expected return magnitudes (easy walk vs hard squat for a heavy body). PopArt tracks a running mean/std of returns per shape and normalizes the critic value head accordingly. Used in multi-task RL (DeepMind) to handle reward scale mismatches.

Cost: ~2 days.

---

### A6. TVS Difficulty Re-Scoring
**Priority: LOW-MEDIUM — better curriculum for hard clips**

Current difficulty score (root velocity, flight ratio, DOF velocity) misclassifies squats/crawls as easy (low root velocity, no flight). Torque Variation Score (TVS, arXiv 2512.07248) is physics-grounded: it measures the torque variation required to correct small pose perturbations.

> "High-TV motions induce flat reward landscapes and vanishing policy gradients."

TVS correctly rates squats/crawls as hard. Re-weight curriculum using TVS. Cost: ~3–5 days.

================================================================================

## TRACK B — Analysis & Paper Contributions (No New Training)

These use existing checkpoints from `hhi_1024_motion` or `hhi_1024_motion_tune`.

---

### B1. Held-Out Evaluation
**Priority: CRITICAL — core generalization claim of the paper**

Generate held-out body shapes via HUMOS, evaluate both checkpoints:

1. **Interpolation** — 16–32 new random betas from `[-3, 3]` (different seed from training 128)
2. **Extrapolation** — betas in `[-5, 5]` range (scale existing betas by 5/3)

```bash
python protomotions/evaluate_hhi_faults.py \
    --checkpoint results/hhi_1024_motion/last.ckpt \
    --motion-file /path/to/held_out_interpolation.pt \
    --num-envs 64 \
    --output results/eval_mlp_interpolation.csv
```

Key metric: tracking error vs beta L2 norm from training distribution → generalization curve. This is the core paper figure.

---

### B2. Embodiment Encoding Probe
**Priority: HIGH NARRATIVE VALUE — "AI learned physics" paper figure**

At inference, record actor hidden activations for each of the 128 training shapes. Fit linear regression:
```
activation_vector → [total_mass, com_height, limb_lengths, ...]
```
Report R² per physical property. If R² > 0.8 for mass and limb length, the policy built an internal physics representation purely from imitation learning, with no explicit supervision.

No new training. Cost: ~1–2 days analysis.

---

### B3. Stride / Retargeting Analysis
**Priority: MEDIUM — visual paper contribution**

For a walking clip tracked across all 128 shapes: measure stride length and frequency per shape. If taller bodies take longer strides at similar frequency, the policy is doing implicit gait retargeting — adapting to body proportions without being told to.

No new training. Cost: ~1 day rollout analysis.

================================================================================

## TRACK C — Scale-Up (Conditional)

**Trigger:** Only after Track A shows clear gains on hard clips.

Run all 20,951 valid motions from `/home/hlz/repos/hhi/data-processing/valid_motions.txt`, same config as best Track A experiment. Scale `num_envs` to 8192, `batch_size` to 32768 (~31 GB GPU, safe on A40).

Value: "Scales to full AMASS" strengthens the system contribution for the paper.
  
  claude --resume 6e4a4a57-daf6-4774-af32-3339857a56c0