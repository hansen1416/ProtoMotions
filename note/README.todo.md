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

## On Hold

- **H1 (MED) — small multi-shape ablation, on hold 2026-07-31.** `eval/success_rate` /
  `eval_holdout/success_rate` for `hhi_wide_stage2_scratch` (full 20,946 clips x 128 shapes) has
  been stuck in the 63-70% band around epoch 8500, matching where the abandoned adapter lineage
  (v1-v6) also plateaued (78-82%) and where the isolated `hhi_1_motion_128_shape` pilot hard-
  plateaued (65-70% from epoch ~2200 to 8500, run manually killed after stabilizing) — converging
  evidence the ceiling is a shape-conditioning capacity/architecture bottleneck, not a data-scale
  or curriculum problem, since single-shape training on the same 20,946 clips reaches 91%+ by
  epoch 4000 and 95-97% eventually.
  Proposed next step (drafted, not yet launched): isolate the shape-conditioning problem from
  data-scale/curriculum noise by training the exact same architecture (`mlp_wide.py`, unmodified)
  on a small (~100-200 clip), difficulty-stratified, full-128-shape **static** motion file — no
  `GlobalClipPool`, no residency streaming, so `eval/success_rate` is a clean full-distribution
  number from the first eval point and iteration is fast. Target: clear 95%+ where GPU-hours-per-
  clip is no longer scarce. If it still plateaus in the 65-82% band, that's strong confirmation the
  fix needs to be architectural (capacity or how `morphology_obs` is injected); if it clears 90%+,
  the full run's scale/pool mechanics need another look instead.
  Tooling ready: `tools/build_small_multishape_subset.py` (builds the small static file from R2,
  difficulty-stratified sampling) + launch command using unmodified `mlp_wide.py` — see chat log
  2026-07-31 or `README.note.md` for the full command.
  **Decision: hold until `hhi_wide_stage2_scratch` has had more time to run** — it's still slowly
  climbing (`eval_holdout/success_rate` 0.63 -> 0.70 over the last ~3600 epochs, not yet flat like
  the 1-motion pilot was), so it's not yet confirmed to have hit the same hard ceiling. Revisit
  once that run's `eval_holdout/success_rate` trend flattens or clearly breaks past ~80%.

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


------


1. PD gains are literally identical across all 128 shapes. smpl_mor.py's override_control_info sets
  stiffness/damping/effort/velocity limits purely by joint-name regex (e.g. all *_Hip_* get stiffness=800, effort_limit=500) —
  there's no scaling by that shape's actual mass/inertia. A heavy/tall shape may be structurally under-actuated (500 N·m
  isn't enough torque to support more mass), while a light/small shape is over-gained relative to its inertia (jittery, prone
  to oscillation). This is a classic multi-morphology RL pitfall — gains should scale with computed segment mass/length, not
  stay fixed.
  2. Tracking reward and success/termination criteria use absolute, non-normalized position error. mean_body_pos_error (used
  both for the hard 0.5m termination threshold and eval success) and compute_rh_rew/gt reward (gt_coef=-25, root-height
  success threshold 0.3m) all operate on raw meters of position error — never divided by that shape's actual height/limb
  length. A taller shape's joints are geometrically farther from the root at the same relative pose accuracy, so it
  accumulates more meters of error for identical skill, making it more likely to hit the 0.5m termination and less likely to
  pass the 0.3m/0.5m success test — pure geometry bias, not policy quality. This alone predicts exactly the signature you'd
  expect if shape were the bottleneck: large/tall shapes systematically failing more, independent of skill.