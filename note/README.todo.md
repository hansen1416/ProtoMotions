# TODO

> **Note (2026-07-26):** this file had gone stale — it still described Stage 1 as running and
> Stage 2 as blocked on data. Updated below based on what's actually happened since
> (`README.note.md` §1-37 is the authoritative chronological record if in doubt).

## Current Status

| | |
|---|---|
| Stage 1 (`hhi_20946_neutral`) | **Converged.** Superseded by MoE/wide-control iterations; `hhi_moe_20946_neutral_stable` reached 95.9% success @ epoch 16400. |
| Stage 2 data pipeline | **Done.** `r2:proto-data/hhi_stage2/` (328 shards) and `hhi_stage2_per_clip/` (per-clip repackaged, used by the global clip pool) both on R2. |
| Stage 2 training | **Running** — `hhi_wide_fusion_stage2_clippool` launched 2026-07-26 on RunPod (frozen-trunk fusion adapter v4 + `GlobalClipPool`, first real run of both). |

---

## Active — Pre-Stage-2 (historical — all resolved)

- **N1 (CRITICAL) — DONE.** Normalizer reset before Stage 2, via `tools/reset_morphology_normalizer.py`.
  Confirmed still in use as of the current stage2 adapter pipeline (`README.note.md` line ~3077).
- **N2 (DECIDED 2026-06-27) — DONE.** Morphology rep: raw betas (11-dim). See `README.gravity-core-eval.md`.
- **N3 — SUPERSEDED.** The original "`hhi_stage2_transfer`, 10× LR, full fine-tune" plan was
  replaced by the frozen-trunk + adapter architecture (v1 LoRA → v2 full-concat → v3 shape-only →
  v4 concat-fusion, the version now training). See `README.note.md` §32-37.

---

## Training Improvements (Stage 2) — DONE

- **A2 (MED) — DONE.** Contact reward extended to knees/wrists. See `README.note.md` §16
  (2026-06-30, "Phase Variable φ and Contact Bodies Extension").
- **A3 (MED) — DONE.** Phase obs `φ = frame_idx / total_frames` added, same §16.

---

## Evaluator Augmentation — partially verified, needs a re-check before relying on it

Originally speced (`README.eval-plan.md` §Part 3) as additions to `HHIFaultEvaluator` /
`protomotions/evaluate_hhi_faults.py`:
- **E3-a, E3-b — DONE**, confirmed present as `SmoothnessAggregateMetric`
  (`protomotions/agents/evaluators/aggregate_metrics.py`) — same 0.4s window / 6500 threshold as spec'd.
- **E3-c, E3-d, E3-e** (`beta_l2`, `explosion_frame`, `completed`) — **not found** under these
  names in `hhi_fault_evaluator.py` as of 2026-07-26. Either implemented differently/renamed, or
  never done — verify before assuming this data exists in eval CSVs.

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
| E7 | Held-out beta generalization (interp + extrap) | **Still blocked.** Interp inference ran (717 files) but was found to be an invalid result — `infer.py` bug concatenated train/val/test splits (see `README.note.md` §11). Needs the fix + re-run; see `README.heldout-pipeline.md`. |

E1-E6: status not re-verified in this pass (not confidently confirmed done or not — check
`results/*/` eval CSVs directly rather than trusting this table).

---

## Analysis (status not re-verified in this pass)

- **B1 (HIGH)** Embodiment probe: record actor hidden activations for all 128 shapes; fit
  linear regression to predict physical properties (mass, COM height, limb lengths); report R²
- **B2 (MED)** Stride analysis: stride length/frequency vs body height for a locomotion clip

---

## Future Training Architecture (from 2026-06-30 training strategy review)

Note: written before the frozen-trunk + adapter architecture (v1-v4, `README.note.md` §32-37).
T-B1 in particular (pre-init the *trunk's* obs normalizer from morphology stats) may not directly
apply now that betas are routed through a separate, freshly-trained `beta_encoder` rather than the
frozen trunk's normalizer — re-evaluate relevance before picking this up.

- **T-B1 (MED)** Pre-initialise obs normalizer from dataset statistics before Stage 2 starts.
  Since all 128 body shapes are loaded at startup, compute `mean` and `var` of `morphology_obs`
  / `physics_obs` across the full shape set and write them directly into the checkpoint normalizer
  buffers before training begins. Avoids the hundreds of epochs needed for online accumulation to
  reach correct scale, and gives correct-scale inputs from epoch 0 for Stage 2.
  Implementation: add a `--pre-init-morphology-normalizer` flag to `train_agent.py` that reads
  the motion file's `motion_betas` / physics feature tensor, computes mean/var, and patches the
  loaded checkpoint before the training loop starts.

- **T-B3 (LOW)** Separate value heads per reward component.
  The current critic outputs a single scalar V(s) that must fit 8 reward terms with different
  timescales (contact spikes vs smooth power penalty). Multi-head critics — one V per reward
  term, summed for GAE — reduce the critic's regression burden and may stabilise advantage
  estimates. Non-trivial: requires per-term reward storage in the experience buffer and separate
  critic out-keys. Worth revisiting if Stage 2 training shows unstable critic loss.

- **T-C3 (RESEARCH)** Online body-dynamics estimator for in-context shape adaptation.
  The current policy receives a static physics feature vector at episode start and has no
  mechanism to update its internal model mid-episode based on observed dynamics. Testable
  hypothesis: compare (a) static physics features only (current) vs (b) physics features +
  a lightweight online estimator that infers body parameters from recent force/velocity
  residuals (e.g., a small RNN or attention over the last K steps of `(action, dof_vel, root_accel)`).
  The key experiment is the held-out beta (E7) evaluation: if static conditioning plateaus on
  unseen shapes, the online estimator is the natural next step. Implement only if E7 results
  show a clear shape-generalisation gap.

---

## Pilot Checkpoints (ablation)

`hhi_1024_transfer` and `hhi_phy_1024_transfer` are on R2 (`r2:proto-data/ckpt/`). Run E1
against the 1024×128 pilot dataset if needed for ablation comparison against Stage 2.
