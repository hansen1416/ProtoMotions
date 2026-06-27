# TODO

## Current Status

| | |
|---|---|
| Stage 1 (`hhi_20946_neutral`) | **Running** — 20,946 neutral motions, `smpl_mor_neutral` + `mlp.py`, 6 shards on RunPod |
| Stage 2 data pipeline | **In progress** — `tools/prepare_stage2_data.py`; check `pipeline_log.txt` for batch progress |
| Stage 2 training | **Blocked** on data |

---

## Active — Pre-Stage-2

- **N1 (CRITICAL)** When Stage 1 converges: inspect normalizer before Stage 2
  ```bash
  python tools/reset_morphology_normalizer.py \
      --checkpoint results/hhi_20946_neutral/last.ckpt --dry-run
  # expect var[-10:] ≈ 0 on all beta dims; var[-11] (gender) ≈ 1.0
  python tools/reset_morphology_normalizer.py \
      --checkpoint results/hhi_20946_neutral/last.ckpt \
      --output results/hhi_20946_neutral/last_morph_reset.ckpt
  ```
- **N2 (DECIDED 2026-06-27)** Morphology rep: **raw betas (11-dim)** — gravity-core eval shows
  physics features only 0.018 m better (< 0.05 m threshold), not worth obs-dim change. See
  `README.gravity-core-eval.md` for full analysis.
- **N3** When Stage 2 data ready and Stage 1 converged: launch `hhi_stage2_transfer`
  with 10× lower LR, from the reset checkpoint

---

## Training Improvements (Stage 2)

- **A2 (MED)** Contact reward: extend `contact_bodies` to include `L_Knee`, `R_Knee`,
  `L_Wrist`, `R_Wrist` — needed for crawl/kneel/squat clips which are the main failure class
- **A3 (MED)** Phase obs: add `φ = frame_idx / total_frames ∈ [0,1]` as an observation key
  to resolve temporal aliasing in periodic motions

---

## Evaluator Augmentation (needed before first Stage 2 eval run)

Add to `HHIFaultEvaluator` / `evaluate_hhi_faults.py` CSV output:
- **E3-a** `mean_normalized_jerk` — via existing `SmoothnessCalculator.compute_normalized_jerk_from_pos`
- **E3-b** `high_jerk_frame_pct` — % of 0.4s windows exceeding NJ threshold 6500
- **E3-c** `beta_l2` = `‖β‖₂` per motion (one-liner from `motion_lib.motion_betas[id].norm()`)
- **E3-d** `explosion_frame` — first frame where body_dist > 5 m; −1 if none
- **E3-e** `completed` bool — all frames ran without explosion

See `README.eval-plan.md` §Part 3 for the full implementation spec.

---

## Evaluation (primary target: Stage 2 checkpoint)

Run after Stage 2 converges. All post-E1 analyses are pure pandas from the CSV — no re-simulation.

| # | Task | Blocker |
|---|---|---|
| E1 | Full CSV evaluation on RunPod (2–4 hrs per checkpoint) | Stage 2 checkpoint |
| E2 | Success/failure table: `success = root_height>0.3m AND body_dist<0.5m` | E1 |
| E4 | Per-shape success rate: group by `(gender, beta_key)` → 128 rates → histogram | E1 |
| E5 | Cross-shape variance: `std(mean_body_dist)` per clip across 128 betas | E1 |
| E6 | Shape extremity: scatter `‖β‖₂` vs `mean_body_dist`, fit OLS slope | E1 + E3-c |
| E7 | Held-out beta generalization (interp + extrap) | interp inference done; see `README.heldout-pipeline.md` |

---

## Analysis

- **B1 (HIGH)** Embodiment probe: record actor hidden activations for all 128 shapes; fit
  linear regression to predict physical properties (mass, COM height, limb lengths); report R²
- **B2 (MED)** Stride analysis: stride length/frequency vs body height for a locomotion clip

---

## Pilot Checkpoints (ablation)

`hhi_1024_transfer` and `hhi_phy_1024_transfer` are on R2 (`r2:proto-data/ckpt/`). Run E1
against the 1024×128 pilot dataset if needed for ablation comparison against Stage 2.
