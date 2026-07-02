# Persistent Motion Failures — hhi_20946_neutral

Training run: `results/hhi_20946_neutral` (Stage 1 — neutral SMPL body, β = 0)
Checkpoint range analyzed: epoch 17000–20400 (18 evaluation snapshots per rank)
Dataset: 20,946 HumanML3D clips × 1 body shape (neutral), 6 GPU ranks, 3,491 clips/rank

`eval/success_rate` (tensorboard, `lightning_logs/version_0`) plateaus at **≈ 82–85%** over
epochs 19600–20400 — consistent with the persistence counts below.

---

## How the Data Was Collected

### 1. Source

```
results/hhi_20946_neutral/failed_motions/failed_motions_epoch_<epoch>_rank_<rank>.txt
```

612 files total (epoch 1000–20400 step 200, 6 ranks × 102 epochs). Each file lists integer
`motion_id`s (one per line) that failed evaluation for that rank/epoch — i.e. the character fell
/ terminated early, per `mimic_evaluator.py::_save_failed_motions`.

### 2. ID structure — differs from the `hhi_1024_motion` pilot

This run has **no beta multiplication** (single neutral shape only), so `motion_id` indexes
directly into that rank's shard's `motion_clip_ids` — no `// 128` / `% 128` beta decomposition
needed.

Each rank's local shard (`/home/hlz/datasets/humos_proto_neutral/offset/humanml3d_neutral_20946_{rank:04d}.pt`)
has exactly **3,491 motions** (6 × 3,491 = 20,946). Verified: max `motion_id` across all failure
logs for every rank is 3490 (`== len(shard) - 1`), confirming direct indexing.

```
global_clip_idx = rank * 3491 + motion_id
```

### 3. Text annotation lookup

Same as the pilot: `/home/hlz/repos/hhi/data-processing/motion_id_text.json` (22,459 entries),
keyed by the clip ID with `M`-prefix stripped and zero-padded to 6 digits.

### 4. Aggregation script

`/tmp/.../scratchpad/analyze_20946_neutral_failures.py` (ad-hoc, not committed to `tools/` —
mirrors `tools/analyse_failed_clip_overlap.py` structure; recreate if needed for the RPD run).
Loads each of the 6 shards once (~1.8–3.3 GB each), extracts only `motion_clip_ids`, frees the
shard, then joins against the aggregated per-rank fail counts. Output:
`results/hhi_20946_neutral/persistent_failures.txt`.

---

## Key Findings

### Overall persistence (18 analyzed epochs, 17000–20400)

| Threshold | # clips | % of 20,946 |
|---|---|---|
| Failed 100% of 18 epochs | 1,818 | 8.7% |
| Failed ≥ 90% of epochs | 2,073 | 9.9% |
| Failed ≥ 75% of epochs | 2,640 | 12.6% |
| Failed ≥ 50% of epochs | 3,502 | 16.7% |

No beta dimension here, so persistence-across-epochs is the only failure signal (unlike the pilot's
avg-betas-failed metric).

### Category breakdown — top 100 worst clips (18/18 epochs failed)

| Category | Count in top 100 |
|---|---|
| single-leg balance / kick / leg-swing / stretch | 40 |
| crawl / all-fours | 20 |
| balance on object / beam / climb | 14 |
| squat / crouch | 8 |
| sit | 7 |
| backward motion | 6 |
| kneel | 5 |
| lie down / get up / push-up | 3 |
| unclassified | ~7 (mostly outliers: fast run/walk speed changes, crab walk) |

(Rows overlap — a clip can match multiple keyword categories.)

### Comparison to the `hhi_1024_motion` pilot

The pilot's dominant failure categories (crawl 21%, backward 10%, sit 10%, kneel 7%, squat 3% of
its top 100) are all still present here and still severe (crawl is again the single largest
"classic" category). But at full 20,946-clip scale, a **new dominant category emerges**:
**single-leg dynamic balance** (standing on one foot, leg kicks/swings/circles, knee-to-chest,
leg-stretch clips) — 40% of the worst clips. These clips barely existed in the 1024-clip pilot
subset (which skewed toward locomotion-style content) but are common in the full HumanML3D
library (exercise/stretching/physical-therapy-style clips).

**Interpretation**: both failure families share a root cause — **narrow/unstable base of
support**. Crawl/kneel/squat/sit/backward drop the COM near or below standing height (established
in the pilot). Single-leg balance clips instead narrow the *support polygon* to one foot while
dynamically swinging the other limb — a different but related stability failure mode the
standard (non-residual) PD policy has not learned to handle.

### Relevance to the residual-PD transfer run

`hhi_20946_neutral_rpd` was fine-tuned from `hhi_20946_neutral/last.ckpt` (epoch 20000) using
residual PD control (`q_target = q_ref + scale*action`). Since ~9–17% of the *base* Stage-1
policy's clips were already persistently failing before the transfer even started — and skew
heavily toward exactly the categories residual PD was intended to fix (crawl/kneel/squat, per
`README.todo.md` §1) plus the newly-identified single-leg balance class — any transfer failure
should be checked against whether these same clip categories are still failing post-RPD, or
whether RPD introduced new failure modes (e.g. instability from `tanh(action)` residual scale,
physics explosions similar to the `hhi_1024_motion_tune` fine-tune). Re-run this same
aggregation script against `results/hhi_20946_neutral_rpd/failed_motions/` for a direct
before/after comparison.

---

## Output File

```
results/hhi_20946_neutral/persistent_failures.txt
```

Columns: `global_clip_idx | clip_id | epochs_failed/18 | description`
