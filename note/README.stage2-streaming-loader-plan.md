# Stage 2 Streaming Data Loader (shard-by-shard from R2) — Implementation Plan

**Status: implemented 2026-07-14** (see `note/README.note.md` §33 — matches this plan almost
exactly), **then superseded by the global clip pool 2026-07-24** (`note/README.note.md` §37,
`README.stage2-global-clip-sampling-plan.md`), which replaces per-rank shard rotation with a
resident per-clip pool. Kept for the shard-rotation design rationale below, which the clip pool
built on. This refines and supersedes the "What
needs to be built" section of `note/README.stage2-training-plan.md` after checking it against the
current codebase — nothing described there (`MotionLibPool`, `FileDownloader`,
`prepare_stage2_schedule.py`) was ever built.

## Context

Stage 2 trains on `r2:proto-data/hhi_stage2/` — 328 `.pt` shards × ~3.4 GB (~1.1 TB total,
20,946 clips × 128 shapes = 2.68M motions). `MotionLib.load_from_file()` materializes every frame
tensor into RAM/VRAM at construction time; it has no notion of "more data than fits." This works
for Stage 1 (20,946 clips × 1 shape, ~16 GB, one file per rank via the existing `slurmrank`
convention) and the pilot (131,072 motions, ~54 GB), but Stage 2 is ~20x too big for any single
box.

The streaming design (sliding 2-file window, background prefetch via `rclone`) was already sketched
in `note/README.stage2-training-plan.md`, but nothing was ever built. The recently-built Stage 2
architecture (`examples/experiments/mimic/mlp_wide_lora_stage2.py`, frozen-backbone + LoRA
adapter, see `note/README.note.md` §32) currently has no way to actually run on the full 128-shape
data — it can only use the small `hhi_stage1_merged6` (6-file, RAM-fits) smoke-test data
referenced in its docstring. This plan closes that gap.

I'm simplifying two things from the original doc based on what I found in the current codebase
(details below): no `schedule.json`/cursor file (rotation becomes a pure function of
`current_epoch`, so resume needs no extra state), and per-rank shard rotation reuses the same
`torch.distributed.get_rank()` convention `MotionLib` already uses for `slurmrank` files.

## Design

**Rotation is a pure function of epoch, not a persisted cursor.** At construction, list remote
files once (`rclone lsf {r2_source}`), shuffle deterministically with a fixed seed, partition
across ranks by `files[rank::world_size]` (interleaved — reuses the `torch.distributed.get_rank()`
/ `get_world_size()` pattern from `MotionLib.process_packaged_motion_file_name_multi_gpu`,
falling back to `rank=0, world_size=1` if `torch.distributed` isn't initialized, so single-GPU
debug runs work unchanged). `target_shard_idx = (current_epoch // epochs_per_shard) % len(rank_files)`.
Since `current_epoch` is already checkpointed/restored (`BaseAgent.load_parameters`), resume just
needs to load whatever shard that formula points to — no `schedule.json`, no cursor to
save/restore. (Gotcha to document: `world_size` must stay identical across resume, same caveat
`slurmrank` already has.)

**Per-rank independent shards, not one cluster-wide shared shard.** Each GPU rotates through its
own disjoint slice of the 328 files. With 328 shards and only 6 GPUs, having every rank train on
the *same* shard at a time would mean 6x redundant compute over the same 8,192 motions and 6x
slower coverage of the full 2.68M-motion set — the "per-rank shard data loading" line in the
original doc's references table already points this direction. All ranks still rotate on the same
`current_epoch` boundary (epoch count is identical across ranks in the standard PPO loop), so
gradient sync stays lockstep — they just each pull a different file when they do.

**Rotation happens at epoch boundaries with a full env reset, not mid-rollout with partial
invalidation.** The original doc's "environments reset naturally at their next episode boundary"
doesn't actually work: `load_from_file` overwrites `gts`/`grs`/etc. **in place on the same
`MotionLib` object** (confirmed by reading it — it's `setattr` onto `self`, not a new object), so
motion index `i` refers to a different motion the instant the file swaps. An env mid-episode on
motion `500` would silently start tracking the wrong reference trajectory rather than erroring.
Forcing every env to reset at the moment of rotation avoids this outright, and is cheap to add:
`BaseAgent.fit()` already does exactly this once, unconditionally, via a local `done_indices =
torch.arange(self.num_envs, ...)` right before the rollout loop starts ("force reset on fit
start"). Reusing that same mechanism for rotation.

**Same-object mutation also means the rest of the codebase needs zero changes.** `env.motion_lib`,
`motion_manager.motion_lib`, and `agent.motion_lib` are all references to one `MotionLib`
instance set up once at env-build time (`agent.py:107`, `self.motion_lib = self.env.motion_lib`).
Because rotation calls the existing `load_from_file` on that same instance, none of those
references ever need to be re-pointed — `MotionLibPool` is a drop-in subclass of `MotionLib`
(same convention as `LoRAResidualMLPWithConcat(MLPWithConcat)`), and every other consumer of
`motion_lib` keeps working unmodified.

## Files to add

### `protomotions/components/motion_lib_pool.py`
- `StreamingMotionLibConfig(MotionLibConfig)` — adds `r2_source: str`, `local_cache_dir: str`,
  `epochs_per_shard: int` (default ~50-100; tune from observed steps/epoch — the old doc's "P≈2000
  steps/file" ÷ `num_steps` per epoch, default `num_steps=32`, is the right ballpark but not
  hardcoding a false-precision number), `shard_shuffle_seed: int = 42`.
- `MotionLibPool(MotionLib)`:
  - `__init__`: resolve rank/world_size, list+shuffle+partition remote files, download this
    rank's epoch-0 shard *synchronously* (env/simulator construction needs real motion data
    immediately — can't defer), then call `super().__init__` on the downloaded local path. Kick
    off background prefetch of the next shard in this rank's rotation.
  - `maybe_rotate(current_epoch) -> bool`: compute `target_shard_idx`; if unchanged, no-op. If
    changed: block on the prefetcher for the (already-downloading) next file, call
    `self.load_from_file(local_path)` (inherited, in-place), delete the old local file, kick off
    prefetch for the shard after that, return `True`.
  - `sync_to_epoch(current_epoch)`: same as `maybe_rotate` but unconditional — used once after
    checkpoint load on resume, in case the restored epoch's target shard differs from the epoch-0
    shard loaded at construction (the unconditional fit-start reset already handles the env side
    for free; this just makes sure the *right* shard is loaded before that first rollout).
  - Local `FileDownloader`: background-thread wrapper around `rclone copy <remote> <local_dir>
    --retries=10 --retries-sleep=30s` (same retry convention as `tools/prepare_stage2_data.py`
    step 5), `is_ready()` / `wait()`. Raise a clear error at construction if `rclone` isn't on
    PATH or the remote listing fails.
  - Keeps exactly 2 local files per rank on disk (current + prefetched) — bounded regardless of
    world size, since each rank only ever touches its own slice.

### `protomotions/agents/callbacks/motion_shard_rotation.py`
- `MotionShardRotationCallback(Callback)` — same base class and file layout as
  `agents/callbacks/slurm_autoresume_srun.py`:
  - `before_play_steps(agent)`: `if agent.motion_lib.maybe_rotate(agent.current_epoch):
    agent._force_full_env_reset = True`
  - `on_load_checkpoint_end(agent)`: `agent.motion_lib.sync_to_epoch(agent.current_epoch)`

## Files to modify

### `protomotions/agents/base_agent/agent.py`
- Add `self._force_full_env_reset = False` next to the existing `self._skip_next_policy_update =
  False` (~line 152) — same pattern, same precedent.
- In `fit()`, immediately after `self.fabric.call("before_play_steps", self)` (~line 464), before
  the rollout `for step in track(...)` loop:
  ```python
  if self._force_full_env_reset:
      done_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
      self._force_full_env_reset = False
  ```

### `protomotions/train_agent.py`
- Where the callbacks list is built (~line 700, next to the existing `AutoResumeCallbackSrun`
  wiring on `args.use_slurm`), add: if `isinstance(motion_lib_config, StreamingMotionLibConfig)`,
  append the `MotionShardRotationCallback` target dict. `motion_lib_config` is already resolved
  and in scope at that point (same function, set at ~line 630; also correctly repopulated from
  `resolved_configs["motion_lib"]` on resume, so the `isinstance` check works in both the
  fresh-start and resume code paths — verify this during implementation, since the variable scope
  was traced but not every branch).

### `examples/experiments/mimic/mlp_wide_lora_stage2.py`
- Add `additional_experiment_arguments(parser)` (same pattern as
  `examples/experiments/masked_mimic/transformer.py`): `--r2-motion-source` (default
  `r2:proto-data/hhi_stage2/`), `--motion-cache-dir` (default `/workspace/motion_cache`),
  `--epochs-per-shard`, `--shard-shuffle-seed`.
- Extend `motion_lib_config(args)`: if `args.r2_motion_source` is set, return a
  `StreamingMotionLibConfig(...)`; otherwise fall back to the current plain
  `MotionLibConfig(motion_file=args.motion_file)` behavior. This keeps the file's existing
  2a smoke-test command (small `hhi_stage1_merged6` data, `--motion-file`, no streaming) working
  unchanged, and only turns on streaming for the real 2b full run.

## Explicitly not building
- `tools/prepare_stage2_schedule.py` / `schedule.json` — unnecessary once rotation is
  epoch-driven instead of cursor-driven.
- Any change to `component_builder.py`, `motion_manager/*.py`, or observation/reward code — none
  needed, per the same-object-mutation point above.

## Verification (once implemented)
1. **CPU, no cloud**: point `StreamingMotionLibConfig.r2_source` at a local directory of 2-3 tiny
   dummy `.pt` shards (`rclone copy` works fine against local paths, no R2 credentials needed).
   Confirm: construction loads shard 0; calling `maybe_rotate` at the configured epoch swaps
   `motion_lib.motion_lengths`/`gts`/etc. to shard 1's content, deletes shard 0's local copy,
   starts prefetching shard 2; `sync_to_epoch` after a simulated "resume" at a later epoch lands
   on the correct shard directly.
2. **Integration, RunPod**: run `mlp_wide_lora_stage2.py` against the real
   `r2:proto-data/hhi_stage2/` with a small `--epochs-per-shard` (e.g. 2) so several rotations
   happen in a short session. Watch for: no crash across a rotation, `eval/success_rate` and
   other metrics keep logging normally through the swap, `df -h` on the cache dir stays bounded
   (~2 shards × 3.4 GB per rank) across multiple rotations, and a mid-run resume lands on the
   epoch-correct shard (check the `sync_to_epoch` log line).
