# Stage 2 Training Plan

> **Status: superseded (2026-07-26).** This plan's transfer approach (raw-betas, full
> fine-tune, `MotionLibPool`/`FileDownloader`/`schedule.json` sliding window) was replaced by
> the frozen-trunk + adapter architecture — see `README.note.md` §32–37. The R2 path below
> (`stage2_data/`) is also stale; the actual data lives at `r2:proto-data/hhi_stage2/` (and
> `hhi_stage2_per_clip/` for the per-clip repackaged format used by the global clip pool).
> Kept for historical context on why the naive `MotionLib` approach doesn't scale.

## Context

Stage 2 trains a morphology-conditioned policy on the full 20,946 HumanML3D clips × 128 SMPL body shapes = 2.68M motions, transferred from the Stage 1 neutral checkpoint.

**Data:** 328 `.pt` files × 3.4 GB each ≈ 1.1 TB total.  
**Status:** ongoing — check `/workspace/stage2_prep/pipeline_log.txt` (or local `data/preprocessing/pipeline_log.txt`) for current batch progress.  
**Each file:** 8,192 motions = 128 shapes × 64 clips. All 128 shapes are present in every single file.

---

## Why the old MotionLib approach doesn't work

The current `MotionLib.load_from_file()` materializes all frame tensors into RAM at startup. This worked for:
- Stage 1: 20,946 clips × 1 shape, single slurmrank file per GPU
- Pilot: 1,024 clips × 128 shapes = 131,072 motions, manageable

At Stage 2 scale: 328 files × 3.4 GB = ~1.1 TB — cannot fit in any RAM budget. Even the slurmrank split (4 ranks × 82 files = ~280 GB/rank) doesn't help.

**Two data sizes:**

| Data | Size | Fits in RAM? |
|---|---|---|
| Metadata (motion_lengths, betas, weights, etc.) | 328 × 8,192 × ~200 B ≈ **540 MB** | Yes |
| Frame tensors (gts, grs, gvs, gavs, dvs, dps) | **~1.1 TB** | No |

---

## Storage architecture

Data lives in **Cloudflare R2** (`r2:proto-data/`), not Google Drive. R2 is S3-compatible with free egress and datacenter-to-datacenter speeds from RunPod (~200–500 MB/s). Google Drive has rate limits and slower throughput (~50–100 MB/s).

Upload command:
```bash
rclone copy /media/hlz/R/stage2_data/ r2:proto-data/stage2_data/ \
    --transfers=4 \
    --multi-thread-streams=16 \
    --multi-thread-chunk-size=128M \
    --progress
```

---

## Solution: Sliding file window with background prefetch

Keep only **2 files on local disk at a time** (current + 1 prefetched). MotionLib loads the current file normally. Background thread downloads the next file while training runs. On file swap, environments with invalidated motion IDs reset naturally at their next episode boundary.

```
R2 remote: [f000, f001, f002, ... f327]
                         ↓ rclone (background subprocess)
Local cache (/workspace/motion_cache/):  [f_current, f_next]  ← ~7 GB max
                         ↓ loaded into RAM
MotionLib:               [f_current, 8,192 motions, ~3.4 GB RAM]
```

**Why this works for shape diversity:** Every file already contains all 128 SMPL body shapes (128 shapes × 64 clips per file). Shape coverage is never compromised even when training on a single file.

**Download latency:**

| Storage | Speed | 3.4 GB download time |
|---|---|---|
| R2 → RunPod | ~200–500 MB/s | ~7–17 seconds |
| Google Drive → RunPod | ~50–100 MB/s | 34–68 seconds |

Training on one file takes several minutes (P = ~2,000 steps × ~0.2s/step ≈ 7 min). Download finishes long before training exhausts the file.

---

## File schedule

Computed once at training start, saved to disk for reproducibility and resume.

- Shuffle all 328 remote file paths with a fixed seed → `schedule.json`
- Track a cursor (current file index) in the checkpoint
- On resume: reload `schedule.json`, advance cursor to last completed file
- Each "epoch" = one full pass through all 328 files = all 2.68M motions seen once

```json
{
  "seed": 42,
  "files": ["r2:proto-data/stage2_data/batch_0017_0003_offset.pt", ...],
  "cursor": 0
}
```

---

## Transfer setup from Stage 1

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | Stage 1 LR × 0.1 | Standard fine-tuning |
| Normalizer | Reset + 500-step warm-up | Stage 1 stats are β=0 only |
| Morphology rep | **Raw betas (11-dim)** — same as Stage 1; decided 2026-06-27 from gravity-core eval (gap 0.018 m < 0.05 m threshold); see `README.gravity-core-eval.md` | Clean Stage 1 → Stage 2 transfer, no arch change |
| Steps per file (P) | ~2,000 | Enough for all 8,192 motions to be visited; adjust based on episode length |

---

## What needs to be built

### 1. `tools/prepare_stage2_schedule.py`
- Lists all files in `r2:proto-data/stage2_data/` via rclone
- Shuffles with fixed seed
- Writes `schedule.json`

### 2. `MotionLibPool` (new class, wraps `MotionLib`)
- Manages 2-slot local file cache
- Exposes `step(n)` — counts steps, triggers rotation when P reached
- Exposes `rotate()` — swaps current file, deletes old, signals prefetch
- Returns list of invalid motion IDs after rotation (for env reset)

### 3. `FileDownloader` (utility class)
- Thin wrapper around `rclone copy` subprocess
- Runs in background thread
- Exposes `is_ready()` and blocking `wait()`
- Handles retries on failure

### 4. Training loop hook
- Call `motion_lib_pool.step(num_steps)` after each rollout
- On rotation: call `env.resample_motions(invalid_ids)` to reset affected envs

---

## Evaluation plan (maps to paper requirements)

| Eval | What it proves | Data needed |
|---|---|---|
| E1: full 20k×128 success rate | Core claim | Stage 2 training data |
| E7: held-out interp betas | Generalization, not memorization | 22,459 interp files (generated locally) |
| Curriculum ablation | Stage 1 neutral pre-training helps | From-scratch 20k×128 run (parallel) |
| E3: smoothness | Quality beyond binary success | Jerk metric during eval rollouts |

---

## Disk budget on RunPod

| Item | Size |
|---|---|
| Current file | 3.4 GB |
| Prefetched file | 3.4 GB |
| Stage 1 checkpoint | ~1 GB |
| Stage 2 checkpoint (rolling) | ~1 GB |
| **Total** | **~9 GB** |

Every RunPod pod (even smallest) comfortably handles this.

---

## References borrowed from prior work

| Technique | Source |
|---|---|
| Two-stage curriculum (neutral → shaped) | HUMOS (Petrov et al. 2024) |
| Hard-negative motion oversampling | PHC (Luo et al. 2023) |
| 10× LR fine-tuning + normalizer reset | PHC §4 |
| Streaming file window (download-then-train) | WebDataset / LLM pretraining practice |
| Per-rank shard data loading | ProtoMotions existing infra |
