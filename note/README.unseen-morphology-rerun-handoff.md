# Handoff: Re-run Unseen-Morphology Generalization Evaluation

Context for a Claude Code session starting fresh on a RunPod pod. Goal: reproduce
(and then extend the investigation of) the unseen-morphology generalization
evaluation from `note/README.unseen-morphology-results.md` (2026-09-13 run).
That run's raw output CSVs are already committed and analyzed locally, but the
intermediate motion file it depends on is not:
`data_cache/unseen_morphology_merged_0.pt` (14.5 GB) is gitignored and was
local to whichever pod produced it. **Assume it's gone on this pod** unless a
check below says otherwise, and be ready to rebuild it from R2.

## What this evaluation is

Tests whether the policy generalizes to body shapes it never trained on, by
replaying the checkpoint on 1,024 clip identities (already seen in training,
only the *body* is new for 932/1024 of them) across **32 held-out shapes**:
16 SMPL beta vectors, sampled in the same `[-3,3]^10` range as the 128 trained
shapes but excluded from training, each instantiated in both genders
(`protomotions/data/assets/mjcf/smpl_mor_interp/`, `assets.yaml`). Total
1,024 x 32 = 32,768 (clip, shape) evaluations.

## What was found last time (read the full writeup for detail/evidence)

- **28/32 shapes generalize cleanly**: ~99.6-99.8% success, flat across the
  whole evaluated distance-to-nearest-trained-shape range (2.58-5.05) — no
  decline with distance from the training set.
- **4/32 shapes fail badly** (pooled 35.25% success): `a83170f5` (male
  58.3%), `cf3d5ee7` (male 46.8%), `4d1b9df1` (female 35.8%), `1d4da893`
  (female 0.1%). In every one of the 4, the *same* beta vector on the
  *opposite* gender succeeds at 99.8-100%.
- **A 5th, milder outlier not in the paper's headline "4 failed" table**:
  `male_95749fe2` at 96.3% (38/1024 failing clips) — worth including when
  re-running.
- **Failure signature (all 4/5)**: `terminated=0%` in every case — the body
  never falls/collapses. Failing rows instead show elevated tracking error
  (`mean_gr_error`/`max_gt_error` 2-7x higher than successful rows on the
  same shape) — it's silent drift away from the reference, not instability.
- **Root-cause investigation (this session, direct MJCF asset inspection)**:
  - Confirmed male and female use the *identical* numeric beta vector at
    each flagged beta_key (verified via `assets.yaml`) — so any asymmetry
    is purely from how that vector maps through the gender-specific SMPL
    template, not from different input coefficients.
  - For 3 of 4 (`a83170f5`, `cf3d5ee7`, `1d4da893`): computed real per-body
    mass from the MJCF geometry (density x volume per geom). Found large
    per-segment mass asymmetry between the male/female asset at the same
    beta_key — up to **3.4x** locally for `a83170f5` (e.g. torso/chest/
    shoulders), and the failing gender's total body mass sits at an
    extreme percentile of the 128-shape trained-mass distribution (e.g.
    `female_1d4da893` = 122.9kg = 97th percentile). Since PD gains are
    mass-scaled per body (`rho = m_body / m_reference`, paper appendix
    `app:control`), a body 2-3x off from its "intended" per-segment mass
    gets mismatched torque authority — plausible mechanism for
    drift-without-falling.
  - **`4d1b9df1` does NOT fit this pattern** — its male/female asset
    geometry, mass, and limb lengths are essentially symmetric (ratios
    ~1.0-1.3, same as a random healthy control pair), and neither gender is
    a mass-distribution outlier. Yet `female_4d1b9df1` still fails 659/1024
    clips, and clip-by-clip, `male_4d1b9df1` succeeds on 657 of those exact
    same clips (ruling out "these are just hard motions"). **This one is
    still unexplained** — the mass/geometry story that accounts for the
    other 3 does not apply here.
  - Beta-vector magnitude does not predict failure at all: `4d1b9df1` has
    one of the *smallest* L2 norms of all 16 unique beta vectors, and the
    single largest-norm vector (`95749fe2`) is only mildly affected.
- **Not yet done** (flagged as follow-up in both the original writeup and
  this session): rendered rollout video of the failing cases to see
  qualitatively what the body does, and a dynamic (not just static-asset)
  trace of `4d1b9df1`'s rollout to find its distinct cause.

## Steps to reproduce

### 0. Check what's already on this pod first

```bash
ls -la data_cache/unseen_morphology_merged_0.pt 2>/dev/null && echo "ALREADY HERE, skip rebuild"
ls -la results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt 2>/dev/null
rclone listremotes   # confirm r2: is configured before relying on it below
```

### 1. Get the checkpoint (if not already present)

```bash
rclone copy r2:proto-data/ckpt/hhi_wide_stage2_discover_attention_slot_type_refined.zip \
    /workspace/ProtoMotions/results/
cd results && unzip hhi_wide_stage2_discover_attention_slot_type_refined.zip && cd ..
```

### 2. Rebuild the merged unseen-morphology motion file (if not already present)

Source: R2 `humos_unseen_morphology_offset/` — 128 shards
(`humos_32768_NNNN_offset.pt`, 8 clips x 32 shapes = 256 motions/shard).

```bash
mkdir -p /workspace/unseen_morph_shards
rclone copy r2:proto-data/humos_unseen_morphology_offset/ /workspace/unseen_morph_shards/

python tools/merge_motion_shards.py \
    --src /workspace/unseen_morph_shards \
    --dst data_cache \
    --num-shards 1 \
    --pattern "humos_32768_*_offset.pt" \
    --out-prefix unseen_morphology_merged
# Confirm this produces data_cache/unseen_morphology_merged_0.pt (~14.5 GB)
```

If `r2:proto-data/humos_unseen_morphology_offset/` no longer exists (check
with `rclone lsd r2:proto-data/` first), the fallback is regenerating from
scratch via `note/README.heldout-pipeline.md` Steps 3-5b (HUMOS inference ->
AMASS export -> MotionLib conversion -> frame-0 grounding) — much slower,
only do this if the R2 shards are actually gone.

### 3. Re-run the replay (reproduces the original numbers)

```bash
python tools/analyze_unseen_morphology_generalization.py \
    --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt \
    --motion-file data_cache/unseen_morphology_merged_0.pt \
    --output data_cache/unseen_morphology_generalization.csv
```

Sanity-check first with a small smoke test before the full run (see the
script's own docstring for the smoke-test invocation with `--max-clips 3
--envs-per-asset 3 --max-episode-steps 50`).

### 4. Join with split/distance metadata

```bash
python tools/aggregate_unseen_morphology_generalization.py \
    --train-ids data/splits/hhi_stage2_v1/train_ids.txt \
    --validation-ids data/splits/hhi_stage2_v1/validation_ids.txt \
    --test-ids data/splits/hhi_stage2_v1/test_ids.txt
```

Note: the script's own `--train-ids`/etc. *defaults* point at
`data_cache/hhi_stage2_v1_*_ids.txt`, which doesn't exist in this repo — the
real split files are at `data/splits/hhi_stage2_v1/*_ids.txt` (verified
present locally), so the explicit paths above are required, not optional.
Produces `data_cache/unseen_morphology_generalization.joined.csv` and
`data_cache/unseen_morphology_generalization.distance_bins.csv` — confirm
the numbers match `note/README.unseen-morphology-results.md` before trusting
anything downstream.

### 5. Render the failing shapes as video (new since last run, not yet executed)

```bash
python tools/render_unseen_morphology_failures.py \
    --motion-file data_cache/unseen_morphology_merged_0.pt \
    --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt \
    --include-mild
```

This renders one video per flagged beta key with both genders side by side
on that shape's worst-tracking clip (see the script's own docstring —
`tools/render_unseen_morphology_failures.py`, added this session, not yet
run against real data since the motion file wasn't available locally).
Requires `--simulator isaacgym`, so must run on a GPU pod with a display or
Xvfb (the underlying `record_video_mor.py` auto-starts Xvfb if no DISPLAY is
set).

## What to actually decide / investigate once data is back

1. Confirm the rebuilt `data_cache/unseen_morphology_generalization.joined.csv`
   reproduces the same per-shape success rates as
   `note/README.unseen-morphology-results.md` (sanity check the pipeline
   before trusting anything new).
2. Watch the 4 (or 5, with `--include-mild`) rendered videos. Look
   specifically at `4d1b9df1` — does the female variant visibly do something
   different (foot sliding, arm clipping through torso, a specific pose
   where it starts to diverge) that a static asset/mass analysis can't see?
3. If nothing jumps out visually, the next tool to reach for is a *dynamic*
   trace (not built yet): log per-joint torque / tracking error over time
   for `female_4d1b9df1` vs `male_4d1b9df1` on the same clip, to find where
   in the rollout they diverge.
4. Decide with the user whether/how to broaden the unseen-morphology test
   itself (more held-out shapes, repeated seeds) — this was explicitly
   deferred pending this investigation.
