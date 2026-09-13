# Physical Analysis (Paper §8) — Execution Plan

Target: turn `paper/sections/08_physical_analysis.tex` from "planned analysis; no results in
this draft" into a real result. This doc is meant to be read cold on a fresh RunPod session —
it repeats the reasoning, not just the commands.

## The question (from the paper, verbatim scope)

> Not whether body mass or size correlates with joint torque (expected from mechanics), but
> whether the effect of morphology on control demand **depends on the motion being performed**.

Procedure the paper already commits to (§8): hold motion fixed, vary body, across many of the
128 trained shapes. Record per-joint torque, joint power, an aggregate control-effort measure,
and jerk during **deterministic rollouts** of the trained policy. Group clips by inherited text
caption into coarse semantic classes (walking, running, jumping, kicking, dancing, sitting, ...).
Compare **cross-morphology variation of each quantity between classes** — the hypothesis is that
dynamic motions (running/jumping/kicking) show more shape-sensitivity than quasi-static ones
(standing/sitting). Report per-class distributions, not one correlation number. Instrumentation
must be validated (and cross-body normalized) before any number is trusted.

## What already exists (reuse, don't rebuild)

| Need | Existing piece | Location |
|---|---|---|
| Matched clip x shape corpus | `small150_128shape_refined.pt` (150 clips x 128 shapes = 19,200 motions) | `data_cache/small150_128shape_refined.pt` (already local) |
| Same corpus, unrefined (fallback/comparison) | `small150_128shape.pt` | `data_cache/small150_128shape.pt` |
| The 150 clip IDs (for caption lookup) | plain text list | `data_cache/small150_128shape.clip_ids.txt` |
| Trained policy checkpoint | `hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt` (epoch 28,170) — the paper's final reported model, §76/§79 of `note/README.note.md` | `results/hhi_wide_stage2_discover_attention_slot_type_refined/` |
| Real per-DOF applied torque during a live rollout | `extras["raw/dof_forces"]`, read via PhysX force sensor tensor, `isaacgym/simulator.py:1672-1673` (`_get_simulator_dof_forces`) | already wired into env extras |
| Batched one-motion-per-env deterministic replay pattern | `tools/check_replay_torque_saturation.py` — pins N motion_ids one-per-env, steps them together, logs `torque_trace[step, env, dof]`. **This is the skeleton to extend**, not the saturation-ratio logic itself. | `tools/check_replay_torque_saturation.py:300-335` |
| Power/effort formula (torque x velocity) | `power_consumption_sum` / `power_consumption_exp` (dof_forces, dof_vel -> scalar per env) | `protomotions/envs/rewards/base.py:143,323` |
| Jerk (from rigid body position) | `SmoothnessCalculator.compute_normalized_jerk_from_pos` — already what §76/§79's "Normalized jerk" column uses | `protomotions/agents/evaluators/smoothness_calculator.py:58` |
| Cross-body torque normalization (mass-scaled effort limit) | `compute_effective_effort_limits(shape_body_masses, reference_body_masses, control_info)` — `gain_scale = shape_mass / reference_mass`, applied per DOF | `tools/check_reference_torque_feasibility.py:147` |
| Per-shape physical size/mass features (for normalization + optional covariate) | `physics_features.pt` — 15 features per asset_id (mass, limb lengths, height, etc.), z-scored | `protomotions/data/assets/mjcf/smpl_mor/physics_features.pt` |
| Per-clip captions (for semantic classification) | `annotations_processed.json` — `{clip_id: {annotations: [{text, ...}, ...]}}`, 4 captions/clip typical | `humos/annotations/humanml3d/annotations_processed.json` (local HUMOS repo; copy the ~150 relevant entries over, no need for the whole 25,204-clip file) |

**Nothing here needs HUMOS running or a new HUMOS-side inference pass.** This entire analysis
replays the already-trained ProtoMotions policy on already-existing motion data. RunPod is only
needed because it's IsaacGym + GPU, same as every other eval script in this project.

## What's genuinely new (must be built this session)

### 1. Semantic classifier over captions (new, no precedent in-repo)

150 clips is small enough to hand-verify. Plan:
- Pull each clip's captions from `annotations_processed.json` (join all 4 annotations' text per
  clip into one string for keyword matching — a clip can be multi-behavior, e.g. "a man walks
  forward, then kicks").
- Define the class vocabulary up front (don't let it emerge ad hoc mid-analysis — the paper's
  own example list is the starting point): `walking, running, jumping, kicking, dancing,
  sitting`, plus an explicit `other/unclassified` bucket for anything that doesn't keyword-match
  (do not force-fit everything into six buckets).
- Simple keyword-rule classifier first (regex/substring match per class, e.g. `walk|stroll` ->
  walking), not an LLM call or embedding classifier — this is 150 items, a human can audit the
  full assignment table in one sitting, which is the actual validation step, not classifier
  sophistication.
- **Decision needed before running this for real:** a clip can match multiple classes (the
  walk-then-kick example). Two options: (a) assign to all matched classes (a clip can appear in
  multiple per-class distributions), or (b) assign to the first/primary caption's class only.
  Recommend (a) — the paper's claim is about class-conditional distributions, and dropping
  genuinely multi-behavior clips throws away signal for no real benefit — but confirm this
  against how many clips actually multi-match before committing (if it's 5% of clips, doesn't
  matter which you pick; if it's 40%, it does).
- Output: `clip_id -> [class labels]` mapping, saved as a small JSON, checked into
  `note/` or `data_cache/` so it's inspectable and doesn't need re-deriving.

### 2. Full-coverage instrumented replay (extend, don't reuse as-is)

`check_replay_torque_saturation.py` pins ~60 hand-picked motion_ids one-per-env and logs torque
only, to test a saturation hypothesis. This analysis needs **all 19,200 (clip, shape) pairs**
(150 clips x 128 shapes, not a curated subset), logging torque **and** dof_vel **and** rigid body
position (for jerk) **and** the effective effort limit per shape (for normalization) — one round
trip per batch since 19,200 exceeds any reasonable `num_envs`.

New script: `tools/analyze_morphology_motion_interaction.py`, structured as:
- Same import-order discipline as the precedent (IsaacGym before torch, argparse before any
  project import) — copy that module-level structure exactly, it's fiddly to get right from
  scratch.
- Build `motion_lib` from `small150_128shape_refined.pt` directly (`build_motion_lib_from_config`,
  same as the precedent script) rather than going through the checkpoint's frozen `resolved_configs`
  training corpus — this motion file already **is** the matched clip x shape set, no filtering
  needed (unlike the unseen-morphology work, no `GlobalClipPool`/asset-restriction issues expected
  here since this checkpoint's own asset folder already covers all 128 trained shapes).
- Batch the 19,200 (clip, shape) pairs into rounds of `num_envs` each (pick something like 4096
  or 6144 — matches training-time env counts already proven stable on an A40; 19,200/4096 ~= 5
  rounds). Each round: reset all envs to their assigned motion_id at t=0, step deterministically
  (`sample_mean`-style, no exploration noise, matching how §79's eval protocol runs) through the
  full clip length, logging every step's `dof_forces`, `dof_vel`, and rigid body position into
  per-round tensors, same pattern as `torque_trace[step_idx] = extras["raw/dof_forces"]...` in the
  precedent.
- After each round, compute per-motion (clip, shape) summary stats immediately and discard the
  raw per-step tensors (19,200 x ~200 steps x ~69 DOFs is not huge, but no need to hold it all in
  memory at once) — accumulate into one output table keyed by `(clip_id, asset_id)`.
- Per-(clip, shape) summary stats to compute and save:
  - mean |torque| per DOF, and mean |torque| aggregated over all DOFs (raw, Nm)
  - mean |torque| / effective effort limit per DOF (the normalized version, using
    `compute_effective_effort_limits` — this is the fair cross-body number)
  - `power_consumption_sum(dof_forces, dof_vel)` averaged over the episode (raw + mass-normalized
    by total_mass from `physics_features.pt`, since power scales with body size even at matched
    "difficulty")
  - normalized jerk via `SmoothnessCalculator.compute_normalized_jerk_from_pos` (same function,
    same params as §76/§79's tables, so this number is directly comparable to the existing
    reported jerk values)
  - tracking success/gt_error too (cheap to log alongside, and lets you exclude/flag (clip, shape)
    pairs where the policy actually failed — a failed rollout's "control demand" reading is
    contaminated by whatever the policy was doing while falling, not a clean measurement of the
    motion's actual demand on that body)
- Output: one row per (clip_id, asset_id) with all of the above, e.g. a CSV or parquet —
  150 x 128 = 19,200 rows, small.

### 3. Aggregation and the actual interaction analysis (new, this is the "result")

Given the per-(clip, shape) table plus the semantic-class labels:
- Join clip_id -> semantic class(es) onto the per-motion table.
- For each metric (normalized torque, normalized power, aggregate effort, jerk) and each semantic
  class, compute the **cross-shape distribution** (not just the mean) — e.g., for each clip in a
  class, the spread of that metric across its 128 shape variants, then pool across clips in the
  class. Candidate spread statistics: std, IQR, or coefficient of variation (std/mean) — CV is
  probably right here since it's normalizing for the fact that dynamic motions may have higher
  absolute torque *and* higher variance for trivial reasons; CV asks "how much does shape matter
  *relative to* this motion's own baseline demand," which is closer to the paper's actual claim.
- Compare the per-class CV distributions to test the hypothesis: dynamic classes (running,
  jumping, kicking) predicted to have higher cross-shape CV than quasi-static classes (standing,
  sitting). Report this as **per-class distributions** (e.g. one violin/box plot per class per
  metric), exactly as the paper text already commits to ("A concise result would state that
  relationship together with per-class distributions, not a single correlation coefficient").
- Exclude or flag failed rollouts (from the tracking success column) before computing spread
  statistics — a fallen-over body's torque trace reflects a physics failure, not the motion's
  control demand on that shape.

## Instrumentation validation (do this before trusting any number — paper explicitly demands it)

1. Smoke-test the new script on a tiny slice first: 2-3 clips x 4-5 shapes, eyeball the
   per-step torque/dof_vel/position traces for one episode look physically sane (torque doesn't
   spike to absurd values at t=0, matches the general magnitude seen in
   `check_replay_torque_saturation.py`'s existing output for the same checkpoint).
2. Cross-check the mass-scaled effort limit computation against `check_reference_torque_feasibility.py`'s
   existing validated usage — same formula, same `control_info`/body-mass source, so this should
   be a copy, not a re-derivation.
3. Sanity-check jerk numbers against §79's existing per-motion jerk values for the *same*
   checkpoint and (where they overlap) similar clips — order of magnitude should match the
   reported ~1,000-1,200 normalized-jerk range, not be off by 10x (which would indicate a dt or
   unit mismatch).
4. Confirm the refined vs. unrefined motion file choice doesn't change the qualitative
   class-interaction result — if time allows, run the full 19,200-pair pass on both
   `small150_128shape_refined.pt` and `small150_128shape.pt` and check the per-class CV ranking is
   stable. If it isn't, that itself is a finding worth reporting (refinement's effect on the
   physical-analysis conclusion), not just noise to average away.

## Requirements to have ready before starting on the pod

- RunPod pod, IsaacGym-capable (same as all prior training/eval work — Ubuntu 22.04 template,
  `pip install -r requirements_isaacgym.txt` per this repo's own `CLAUDE.md`). No HUMOS
  environment needed for this section at all.
- `results/hhi_wide_stage2_discover_attention_slot_type_refined/` (checkpoint + resolved configs)
  uploaded — same R2 path used earlier: `r2:proto-data/ckpt/hhi_wide_stage2_discover_attention_slot_type_refined.zip`.
- `data_cache/small150_128shape_refined.pt` and `small150_128shape.pt` uploaded (both already
  exist locally in this repo, small enough to just scp/rclone directly, no regeneration needed).
- `protomotions/data/assets/mjcf/smpl_mor/` (the 128-shape asset folder + `physics_features.pt`)
  present — should already be part of the repo checkout on the pod, just confirm it's there.
- A small caption file: extract just the ~150 relevant entries from
  `humos/annotations/humanml3d/annotations_processed.json` locally first (no need to move the
  HUMOS repo to this pod at all) and upload that small extract.

## Open decisions to confirm with the user when this session starts for real

1. Multi-class clip assignment: assign-to-all vs. primary-class-only (see semantic classifier
   section above) — check the actual multi-match rate on the real 150 clips first, then decide.
2. Spread statistic for the interaction comparison: CV recommended above, but confirm once real
   numbers are in hand that CV doesn't behave strangely for classes with near-zero mean torque
   (e.g. pure sitting clips could have a tiny denominator).
3. Whether to run both refined and unrefined motion files (time/compute tradeoff) or commit to
   refined-only for the paper and mention unrefined as a robustness footnote at most.
4. Exact wording of "aggregate control-effort measure" for the paper — this plan computes it as
   normalized `power_consumption_sum` (mean over episode), but confirm that's the intended
   definition vs. e.g. a peak-torque or effort-limit-saturation-fraction based measure (closer to
   `check_reference_torque_feasibility.py`'s own effort framing) before locking in the paper text.
