# 3-Month Research Plan
**Start:** 2026-06-12  **Deadline:** ~2026-09-12 (ICRA 2027 submission)

---

## Paper Narrative

> "We present the first physics-based motion imitation policy that generalizes across 128 SMPL body shapes. Physics-derived morphology features (mass, COM, limb lengths) unlock shape-adaptive physical control, particularly for floor-contact motions (crawl, kneel, squat) — the primary failure mode of raw shape-parameter conditioning. The policy develops an internal representation of body physics correlated with physical properties it was never explicitly trained to predict."

---

## Current State (2026-06-12)

| Item | Status |
|---|---|
| MLP baseline `hhi_1024_motion` | Converged ~epoch 12000, reward ~0.84 |
| FiLM `hhi_film_1024_motion` | Failed (~0.72), dropped |
| ShapeEmbed | Empirically identical to concat, dropped |
| Failed motion analysis | Done — crawl/kneel/squat/backward are the hard class (65 clips ≥8 betas failing) |
| Data pipeline | Complete — 131,072 motions, 16 local shards |
| Evaluator infrastructure | Ready (`evaluate_hhi_faults.py`) |
| Held-out betas (interpolation/extrapolation) | **Not yet generated** |

---

## What to Drop

- **FiLM** — failed; literature confirms it's the wrong tool for fixed-topology fixed-D conditioning
- **ShapeEmbed** — same empirical result as concat; weakens the story
- **SMPL-X validation** — not blocking for 1024-clip pilot scope
- **Fine-tuning efficiency (S5)** — low ICRA impact; only if weeks 1–9 finish early
- **Hardware validation** — out of scope

---

## Experiments

### Experiment 1 — Physics Features (highest priority)
**Goal:** Replace/augment `morphology_obs` with physics-derived features extracted from SMPL XMLs.

Features to extract (implement `scripts/extract_smpl_physics_features.py`):
- Total mass, COM height at T-pose
- Per-limb lengths: upper arm, forearm, thigh, shin, torso height, shoulder width
- ~15–20 features, z-scored across the 128 training bodies

**Why:** The failed motions (crawl/kneel/squat) fail because the policy doesn't know COM height or mass — raw betas are opaque. Physics features give the policy directly actionable information. Also the strongest interpretability story for the paper.

**Training run:** `hhi_physics_feat_1024` — MLP + physics features, 1024 clips, 4× A40. Start from scratch (cleaner science than fine-tuning).

**Decision gate:** If floor-contact success improves meaningfully (≥10 pp on crawl/kneel subset), this becomes the main result. Even a small gain is publishable because the analysis tells the story.

---

### Experiment 2 — Adaptive Sampler for Hard Clips
**Goal:** Aggressively upweight the 65 identified hard clips (crawl/kneel/squat) during training.

- Increase `failure_discount` magnitude for the identified hard clip IDs
- For floor-contact clips: floor-aware RSI — initialize from a mid-clip frame instead of always standing
- Can be baked into Experiment 1's run (same training, additional config)

**Reference:** Hard clip list at `results/hhi_1024_motion/persistent_failures.txt`. Hard clip category: ≥8 betas failing at epoch 12000.

---

### Experiment 3 — Evaluations (no new training, runs in parallel)

All use existing checkpoints from `hhi_1024_motion` and the new `hhi_physics_feat_1024`.

| Analysis | Output | Tool |
|---|---|---|
| Per-shape eval on 128 training betas | CSV → per-shape degradation curve | `evaluate_hhi_faults.py` |
| Held-out interpolation (16–32 new betas, [-3,3]) | Generalization plot vs MLP | Same evaluator |
| Held-out extrapolation (16 betas, [-5,5]) | Degradation slope | Same evaluator |
| Torque + energy per body shape (same clip) | "Heavier → higher torques" figure | Custom rollout logger |
| COM trajectory + support polygon | Stability analysis figure | Rollout data |
| Stride length vs body height | Implicit retargeting figure | Root position trajectory |
| Embodiment encoding probe | "Policy learned physics" figure | Linear ridge probe on activations |
| Motion category × shape heatmap | 2D failure structure figure | Rollout + `motion_id_text.json` |
| Failure mode taxonomy | Fall / COM drift / joint-limit breakdown | Rollout data |

---

### Experiment 4 — Full-motion Scale-up (conditional)
**Trigger:** Only if Experiment 1 shows clear gains on hard clips.  
**Run:** All 20,951 valid motions from `hhi/data-processing/valid_motions.txt`, same physics-feat config.  
**Value:** "Scales to full AMASS" — strengthens the system contribution.

---

## Month-by-Month Timeline

### Month 1 — New Training + Setup (June 12 – July 12)

| Week | Tasks |
|---|---|
| **1** | Extract physics features from SMPL XMLs. Augment `morphology_obs`. Run evaluator on `hhi_1024_motion` → baseline per-shape CSV. Generate held-out betas via HUMOS. |
| **2** | Launch `hhi_physics_feat_1024`. Implement floor-aware RSI for floor-contact clips. |
| **3** | Monitor training; compare reward curves on crawl/kneel subset specifically. |
| **4** | Run per-shape eval on physics-feat checkpoint. Start rollout logging infrastructure for torque/energy/contact. |

### Month 2 — Analysis Sprint (July 12 – Aug 12)

| Week | Tasks |
|---|---|
| **5** | Held-out generalization eval (interpolation + extrapolation) on both checkpoints. |
| **6** | Torque/energy analysis: same clip × all 128 shapes. COM trajectory + stability. |
| **7** | Embodiment encoding probe: policy layer activations → linear regression → {mass, COM, limb lengths}. **Key paper figure.** |
| **8** | Motion category × shape heatmap. Stride/step analysis. Failure mode taxonomy. |

### Month 3 — Paper (Aug 12 – Sep 12)

| Week | Tasks |
|---|---|
| **9** | Full-motion scale-up run if warranted. Begin paper draft (method section). |
| **10** | Results section + all figures. Qualitative video: side-by-side multi-shape visualization. |
| **11** | Introduction, related work, conclusion. Internal review. |
| **12** | Buffer: polish, video narration, submission. |

---

## Paper Figures (target list)

1. System diagram: HUMOS → motionlib → multi-shape sim → physics-feat policy
2. Per-shape tracking performance across 128 training betas (bar/violin plot)
3. Held-out generalization: body distance vs beta L2 norm (interpolation + extrapolation curves)
4. Torque/energy vs body shape for a fixed locomotion clip
5. **Embodiment encoding probe:** probe R² per physical property — the "AI learned physics" figure
6. Motion category × shape extremity heatmap (crawl/kneel hardest, locomotion robust)
7. Failure mode taxonomy across shape extremity buckets
8. Qualitative: same motion clip across 8 body shapes side-by-side (also video)
9. Ablation: MLP vs physics-feat on overall and floor-contact subsets

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

------

The file is 10 GB. That's because 192 clips × 128 betas × 200 frames × all tensor fields. If you want a tighter set, you can
  rerun with --min-avg-betas 8.0 to get the 65 most severe clips (~3.4 GB), or use --clip-indices for an exact hand-picked
  list. The threshold 5.0 gives you a broader hard-clip curriculum; 8.0 gives you the truly broken ones.

  Ready to use immediately:

  # Fine-tune from the converged checkpoint, sampling only from hard clips
  python protomotions/train_agent.py \
      --robot-name smpl_mor \
      --simulator isaacgym \
      --experiment-path examples/experiments/mimic/mlp.py \
      --experiment-name hhi_failed_finetune \
      --motion-file /home/hlz/datasets/humos_proto/failed_clips.pt \
      --num-envs 4096 \
      --batch-size 16384 \
      --overrides agent.config.init_from=results/hhi_1024_motion/last.ckpt

  Before launching the actual training run though Jump to bottom (ctrl+End) ↓  physics features to morphology_obs — otherwise this is just fine-tuning on the hard clips with the same input representation, which won't tell us much. Do you want to
  tackle the physics features extraction from the SMPL XMLs next?

  claude --resume 8e2b4dfa-0c8c-4a9f-8ce8-7f78c80e21bc