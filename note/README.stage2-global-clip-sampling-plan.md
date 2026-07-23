# Stage 2 Global Clip-Priority Sampling — Design Plan

**Status: plan only, not yet implemented (2026-07-23).** Supersedes the shard-rotation-as-curriculum
behavior described in `note/README.stage2-streaming-loader-plan.md` — that plan solved "how do we
stream 1.1TB through limited memory," this one solves "how do we make the difficulty curriculum
survive shard rotation and cover the whole dataset instead of resetting every rotation."

## Problem with the current mechanism

`StreamingMotionLibConfig` / `MotionLibPool` (`protomotions/components/motion_lib_pool.py`) streams
328 shards (64 clips × 128 shapes = 8192 motions each, ~52MB/clip, ~1.1TB total) per the existing
plan, rotating shard-by-shard on a fixed `epochs_per_shard=64` schedule, round-robin through each
rank's disjoint shuffled slice of the 328 files.

Two problems this plan fixes:

1. **Curriculum is discarded every rotation.** `MotionManager.on_motion_lib_reloaded()`
   (`protomotions/envs/motion_manager/motion_manager.py:124-137`) does
   `self.motion_weights = self.motion_lib.motion_weights.clone()` — i.e. every 64 epochs, whatever
   difficulty signal `MimicEvaluator._update_motion_sampling_weights()` had accumulated for the
   outgoing shard's clips is thrown away and replaced with the incoming shard's static (always-1.0)
   weights. There is currently no persistent, dataset-wide notion of "which clips are hard,"
   contrary to how the `hhi_1024_motion` pilot worked (weights persisted for the life of that run
   since the whole dataset was resident at once).
2. **Clip-variant grouping is positional, not identity-based.** `_expand_to_clip_variants` in
   `protomotions/agents/evaluators/mimic_evaluator.py` (`_build_clip_expansion_index`, lines
   84-127) stacks each shape's motion-id list into a `[num_shapes, num_clips]` grid and uses
   *column index* as clip identity — documented as relying on "all shapes have the same clips in
   the same positional order." A real, stable `motion_clip_ids` field already exists in the
   packaged data (written by `tools/convert_amass_to_motionlib_with_morphology.py:74-129`, loaded
   generically via `MotionLib.load_from_file`'s `setattr` loop) but nothing reads it today.

Goal: track clip difficulty globally across all ~20,946 clips, persist it for the life of the run
(and across resumes), and let it directly drive which clips are physically resident, rather than
resident-set membership being a pure function of epoch count.

## Constraint that shapes the design

Motion data isn't just IDs — `MotionLib` holds full per-frame tensors (`gts`/`grs`/`gvs`/`gavs`/
`dps`/`dvs`) and always moves them onto the compute device at load time (`self.device` set in
`__init__`, real device passed through `component_builder.py:103`), because the simulator needs
them on-GPU every physics step. So the resident working set is bounded by **GPU VRAM**, not disk —
disk (RunPod volume) is cheap to grow and only affects the download cache, not the resident set.

**Per-clip memory footprint**: ~52MB (128 shapes), cross-checked two ways — empirically
(1.1TB / 328 shards / 8192 motions/shard ≈ 409KB/motion) and analytically (SMPL body, ~240 frames
@ 30fps × ~8s avg clip, summing all six per-frame tensors ≈ 422KB/motion). Both agree to within 5%.

**Resident pool size K = 256 clips/rank** (32,768 motions, ~13.4GB) — chosen to exactly match the
already-validated `hhi_1024_motion` pilot's per-rank footprint (1024 clips × 128 shapes ÷ 4 ranks =
256 clips/rank, `--ngpu 4`, confirmed in `note/README.runpod.md`), rather than a computed-but-
unvalidated VRAM ceiling. On 6×A40 (48GB) this is well inside budget with headroom to spare; a
computed ceiling (~80% VRAM target, ~18GB baseline model/optimizer/env overhead observed on the
v2/v3 Stage 2 runs) would support up to ~384-390 clips/rank, but that number hasn't been run on
real hardware — K=256 is the de-risked choice. Revisit upward only after confirming actual v4
baseline VRAM usage via `nvidia-smi` on a dry run.

## Design

**1. Repackage 328 batch-shards into ~20,946 per-clip files.** Pure re-partition of already-computed
data (no retargeting/physics re-run) — each of the 328 shards already distinguishes its 64 clips
via `motion_clip_ids` + `length_starts` boundaries, so splitting is mechanical and embarrassingly
parallelizable across the 328 source files. Output: one `.pt` per clip_id (~52MB avg, holding all
128 shape variants of that clip), plus a `clip_id → file_path` manifest written as a byproduct
(much simpler than a shard-mapping manifest would have been, since it's 1:1).

**2. Persistent per-rank global weight vector.** `global_clip_weights: float[~20,946]`, initialized
to 1.0, lives for the life of the run. Trivial size (~80KB) — nothing like the motion tensor data.
**Checkpointed** alongside the existing `motion_weights` checkpoint entry (`MotionManager.
get_state_dict()`/`load_state_dict()`) so a resume doesn't silently reset accumulated curriculum —
this was called out explicitly as a requirement, not an afterthought, since the entire point of
this plan is that curriculum should *not* reset.

**3. Resident clip pool replaces "one shard resident."** `MotionLibPool` (or its successor) holds
K=256 clips at a time per rank. Local disk cache sized generously above K (e.g. 2-3x) purely for
download hysteresis, so clips oscillating near the boundary don't get re-downloaded every rebuild —
disk is cheap, no reason to pinch this.

**4. Rebuild cadence tracks the evaluator, not a fixed epoch count.** Every time
`MimicEvaluator` produces a fresh success/failure signal (`eval_metrics_every`), recompute the
top-K clips by `global_clip_weights + exploration_bonus`, diff against the current resident set,
background-download anything newly promoted that isn't already cached, reassemble a monolithic
tensor block from the K clips' files (same format `MotionLib.load_from_file` already expects), and
swap it in-place — same full-env-reset + `motion_manager.on_motion_lib_reloaded()` resync as
today's shard rotation. Reusing the existing monolithic-block load path avoids adding incremental
insert/evict mutation support to `MotionLib`'s internals (concatenated tensors + `length_starts`
indexing), which would be a much more invasive change to a heavily-used class.

**5. Exploration bonus (per clip, not per shard).** Visit-count/UCB-style bonus added to a clip's
selection priority, decaying as visits accumulate, so never-yet-seen clips (default weight 1.0)
still get pulled into the pool periodically rather than the pool fixating only on already-known-hard
clips discovered early.

**6. Weight updates target the global vector, keyed by real clip identity.**
`MimicEvaluator._update_motion_sampling_weights()` switches from the positional column-index trick
to `motion_clip_ids`, applies the existing success/failure discount math
(`motion_weights_update_success_discount` / `motion_weights_update_failure_discount`, unchanged) to
`global_clip_weights[clip_id]`, then re-projects into the resident pool's local `motion_weights`
tensor used for per-step multinomial sampling.

**7. Everything below the pool boundary is unchanged**: per-env shape-compatible candidate
restriction (`sample_motions_for_asset_ids`), `resample_on_reset`, `init_start_prob`, the success/
failure discount formula itself.

## Rejected / deferred alternatives (for context, so this doesn't get re-litigated)

- **Keep shard-level round-robin, only make weights global** (an earlier, smaller-scope version of
  this plan): rejected because it still caps sampling to whatever 64-clip block happened to load,
  with no way to prioritize individual hard clips independent of their shard-mates. Superseded once
  the decision was made to remove the shard concept entirely.
- **Weighted shard *scheduling*** (bias which pre-packaged 64-clip shard loads next, without
  repackaging): considered as a lower-risk intermediate step, explicitly superseded by the decision
  to repackage per-clip instead, since per-clip files remove the coarse 64-clip grouping entirely
  rather than just reordering which group loads.
- **Larger K matching the pilot's total dataset size (1024 clips/rank)**: rejected — that reading of
  the pilot conflated "1024 total clips across the whole run" with "1024 clips resident per rank."
  The pilot's actual per-rank resident footprint was 256 clips (1024 ÷ 4 ranks); 1024 clips/rank
  would be ~53GB of motion data alone, exceeding a 48GB A40 by itself before model/optimizer/env
  memory.

## Open items before implementation

- Confirm actual v4 (fusion-adapter) baseline VRAM usage via `nvidia-smi` on a real dry run —
  the ~18GB baseline used in the K sizing math above is carried over from the v2/v3 residual-adapter
  runs' observed 20-22GB/46GB, not measured on this exact architecture.
- Decide the exploration-bonus formula precisely (visit-count decay rate) once the pool-management
  code is being written — not fixed yet, only the mechanism (UCB-style) is agreed.
- Repackaging job needs a concrete script (`tools/`-style, mirroring the existing
  `tools/prepare_stage2_data.py` conventions) — not written yet, this doc only specifies its
  input/output contract (328 shards in, ~20,946 per-clip files + manifest out).

## Explicitly not building (this iteration)

- Any change to `MotionLib`'s core tensor-indexing internals (`length_starts`, concatenated
  gts/grs/etc.) — the pool rebuild reuses the existing full-block `load_from_file` path instead.
- Variable rebuild cadence tuning beyond "track `eval_metrics_every`" — no additional throttling/
  batching of rebuilds is planned unless the naive version proves too network-heavy in practice.
- Cross-rank weight sharing/broadcast — unnecessary, since each clip's shapes are guaranteed to
  belong to exactly one rank's data at a time (repackaging preserves per-clip atomicity), so no two
  ranks ever contend for the same clip's weight.



python tools/repackage_stage2_per_clip.py \
      --r2-source r2:proto-data/hhi_stage2/ \
      --r2-dest r2:proto-data/hhi_stage2_per_clip/ \
      --workspace /workspace/repackage_prep

