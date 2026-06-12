# Persistent Motion Failures — hhi_1024_motion

Training run: `results/hhi_1024_motion`  
Checkpoint range analyzed: epoch 8000–12000 (21 evaluation snapshots)  
Dataset: 1024 clips × 128 body shapes = 131,072 total motions, 4 GPU ranks

---

## How the Data Was Collected

### 1. Source: `failed_motions/` directory

The training run produces per-epoch, per-rank failure logs at:

```
results/hhi_1024_motion/failed_motions/
  failed_motions_epoch_<epoch>_rank_<rank>.txt
```

Each file contains a list of **global motion IDs** (integers) that terminated early during evaluation — i.e., the character fell over before the clip finished. These are written by `protomotions/agents/evaluators/mimic_evaluator.py` via `_save_failed_motions()`, which calls `_save_list_to_file()`. Failure is defined as:

```python
success_rate = 1.0 - self._motion_failed.float().mean().item()
```

A motion is marked failed when the physics rollout terminates early (character falls). It is **not** a measure of tracking fidelity — a character hovering in a half-crouch through a sitting clip without falling counts as success.

There were **240 files** total across all epochs and ranks.

### 2. ID structure

Each rank loads a separate slurmrank file from `/workspace/merged4/`. With 4 ranks and 131,072 total motions, each rank has **32,768 motions = 256 clips × 128 betas**.

The motion ID within each rank encodes:
```
clip_local_idx = motion_id // 128
beta_idx       = motion_id % 128
global_clip_idx = rank * 256 + clip_local_idx
```

### 3. Local shard mapping

The local 16 shards at `/home/hlz/datasets/humos_proto/offset/humos_131072_NNNN_offset.pt` map to the training ranks as:

| Shards | Rank | Global clip range |
|--------|------|-------------------|
| 0–3    | 0    | 0–255             |
| 4–7    | 1    | 256–511           |
| 8–11   | 2    | 512–767           |
| 12–15  | 3    | 768–1023          |

Each shard has 64 clips × 128 betas = 8,192 entries. Verified by loading `humos_131072_0000_offset.pt` and checking `motion_clip_ids[127]` == `motion_clip_ids[0]` (same clip, different beta) and `motion_clip_ids[128]` is a new clip ID.

### 4. Text annotation lookup

Clip IDs (e.g. `004197`, `M008359`) are HumanML3D-style IDs. Descriptions were looked up from:

```
/home/hlz/repos/hhi/data-processing/motion_id_text.json
```

The `M`-prefix indicates mirrored clips. The numeric portion (stripped of `M`, zero-padded to 6 digits) is the key into the JSON.

### 5. Aggregation script

A Python script was run interactively to:
1. Read all `failed_motions_epoch_*_rank_*.txt` files with epoch ≥ 8000
2. For each motion ID, compute `(rank, clip_local_idx)` → `global_clip_idx`
3. Track per-clip: number of distinct epochs failed, total beta-failures, avg betas/epoch
4. Load all 16 local shards to build `global_clip_idx → clip_id` mapping
5. Join with text annotations
6. Save ranked output to `results/hhi_1024_motion/persistent_failures.txt`

---

## Key Findings

### No clip is completely dead

The worst clip at epoch 12000 had **17/128 betas failing** (13%). Across all clips:

| Beta failure threshold | # clips at epoch 12000 |
|------------------------|------------------------|
| ≥ 64 betas (≥50%)      | 0                      |
| ≥ 16 betas (≥12.5%)    | 1                      |
| ≥ 8 betas (≥6.25%)     | 65                     |
| ≥ 4 betas (≥3.1%)      | 320                    |
| ≥ 1 beta               | 929                    |

### Persistent failures (epoch 8000–12000)

| Persistence | # clips |
|-------------|---------|
| Failed all 21 evaluated epochs | 328 |
| Failed ≥ 15 epochs             | 1008 |
| Failed ≥ 10 epochs             | 1024 (all clips) |

All 1024 clips fail at least occasionally — failures are spread across betas, not entire clips.

### Top 30 worst clips (by persistence × avg betas/epoch)

| global_idx | clip_id   | ep/21 | avg_β/ep | Description |
|-----------|-----------|-------|----------|-------------|
| 172 | 004197    | 21 | 9.8 | a person takes several steps backwards then turns around |
| 745 | M005137   | 21 | 9.1 | a person on knees waving arms |
| 604 | M000366   | 21 | 9.0 | a person walk to the right, then gets on all fours and crawls |
| 338 | 008275    | 21 | 8.9 | a kneeling person crawls forward, turns around, crawls back, stands |
| 426 | 010141    | 21 | 8.9 | a man gets on all fours and crawls around and gets back up |
| 586 | 014481    | 21 | 8.6 | a person was sideways to right with the left foot stepping over |
| 340 | 008359    | 21 | 8.5 | taking 2 steps forward then getting on hands and knees and crawling |
| 59  | 001389    | 21 | 8.5 | a person raises both his hands and then sits on his knees |
| 710 | M003969   | 21 | 8.5 | a man balancing on small object |
| 585 | 014468    | 21 | 8.4 | a person raised the both hands and after pull down the left hand |
| 609 | M000598   | 21 | 8.4 | a person crossing their legs and walking to the left |
| 752 | M005429   | 21 | 8.4 | kneeling then crawling on floor |
| 813 | M007690   | 21 | 8.4 | a man crouches with his right hand on the ground then gets back up |
| 637 | M001754   | 21 | 8.3 | a standing person slowly walks backwards then stops |
| 285 | 006830    | 21 | 8.3 | a person gets on their hands and knees |
| 614 | M000894   | 21 | 8.2 | a person starts standing with knees slightly bent and arms held out |
| 458 | 011062    | 21 | 8.2 | person gets down on both hands and knees and crawls around |
| 676 | M003066   | 21 | 8.2 | a man is kneeling |
| 337 | 008245    | 21 | 8.1 | the person gets down on all fours and starts crawling |
| 469 | 011301    | 21 | 8.1 | a person steps each leg backwards |
| 952 | M012082   | 21 | 8.1 | man draws arms in and puts them on his knees |
| 959 | M012257   | 21 | 8.1 | a person begins with arms stretched out, lowers their arms |
| 749 | M005400   | 21 | 8.1 | a person is walking forward |
| 794 | M006838   | 21 | 8.1 | a person kneels over and stays on his knees |
| 64  | 001431    | 21 | 8.0 | robot stands up and throws something with right hand then catches |
| 578 | 014364    | 21 | 8.0 | a person raises his right hand to chest height in a pronated position |
| 957 | M012199   | 21 | 8.0 | a person is side stepping to the right |
| 258 | 006074    | 21 | 8.0 | the person is crawling on his knees |
| 708 | M003950   | 21 | 8.0 | a person standing with bent knees lifts their arms to shoulder level |
| 948 | M012037   | 21 | 8.0 | a person is standing still, then raises their arms |

---

## Failure Categories

Keyword analysis on the top 100 failing clips:

| Category | Count in top 100 |
|----------|-----------------|
| crawl / all-fours | ~21 |
| walk (often as transition, e.g. "walks then kneels") | ~28 |
| backward motion | ~10 |
| sit / seated | ~10 |
| kneel | ~7 |
| squat | ~3 |

### Root cause interpretation

The policy has learned stable bipedal locomotion but struggles with:

1. **Large COM drop** — crawling, kneeling, sitting, squatting all bring the center of mass very close to or below typical standing height. The PD controller and learned policy have not reliably learned to stabilize at these low-COM configurations.

2. **Backward movement** — stepping or walking backward breaks the forward-biased policy assumptions. The character loses balance in the reverse direction.

3. **Transitions** — clips that combine standing with a floor-level phase (stand → crawl, stand → kneel) are particularly hard because the character must bridge two very different stability regimes within one episode.

### Reward dip explanation

The "sudden dips" in the training reward curve occur when the motion sampler happens to assign a large fraction of environments to these hard clips simultaneously. Since the motion manager samples uniformly by default, there is variance in how many hard clips land in each training step.

---

## Output File

Full ranked list of all 1024 clips:

```
results/hhi_1024_motion/persistent_failures.txt
```

Columns: `global_clip_idx | clip_id | epochs_failed/21 | avg_betas_failed_per_epoch | description`

---

## Suggested Next Steps

### Option 1: Exclude from next training run

The `MotionManager` supports `exclude_motions_file` and `exclude_motion_ids` in its config. To exclude the 65 worst clips (≥8 betas failing), extract their clip IDs from `persistent_failures.txt` and build an exclude file.

### Option 2: Reduce sampling weight (already partially active)

The `MimicEvaluator` has a `failure_discount` parameter that divides the sampling weight of failed motions. Tune this to reduce how often the worst clips are sampled during training.

### Option 3: Semantic filtering (recommended for full dataset)

Extend the existing `hhi/data-processing/invalid_motions.txt` to also exclude physically-impossible-to-track motions (floor-level, object-interaction). This keeps the exclusion logic in one place and applies to all future training runs.
