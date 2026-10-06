# Instructions for the pod: paired checkpoint comparison on unseen bodies

Written 2026-10-06. Audience: a Claude Code session running on a RunPod GPU machine, started by the
user (hansen1416). Read this whole file before running anything. Do the steps in order and stop
where a gate says stop.

## 0. What this is, and what it is not

**Question.** The frozen paper checkpoint (`hhi_wide_stage2_discover_attention_slot_type_refined`,
epoch 28,170) tracks 28 of 32 held-out body shapes at about 99.75% and fails four badly (details in
section 1). A later warm-started checkpoint exists (`hhi_wide_stage2_discover_attention_slot_type_fulldata`):
a short polish pass on the same architecture that trained on all 20,951 clips, including the former
validation/test clips, on the same 128 trained bodies. The 32 held-out bodies were never used in
that fine-tune (nothing in the training experiment references `smpl_mor_interp`).

We replay both checkpoints, under an identical protocol, on the same (clip, held-out body) pairs and
report **paired changes**. This is a **checkpoint comparison**. Training duration and motion coverage
changed together, so no improvement or regression may be attributed to either factor alone.

**What it can tell us.**
- Whether the 4 failing shapes (and 1 mild outlier) move by more than checkpoint-to-checkpoint
  variation, which is estimated here from reference runs.
- Whether the other 27 shapes regress.
- Whether any change is concentrated on unseen bodies or is broad, using a trained-body control.

**What it cannot tell us.** Whether the failures are caused by defective assets. Improvement does not
rule an asset problem out, and persistence does not prove one. Do not write that conclusion. The
asset-versus-controller question needs the interventions in `note/README.unseen-morphology-rerun-handoff.md`
and is out of scope here.

**Out of scope, do not do:** training anything, touching R2 contents (read-only use: never `rclone
sync`, `delete`, `move`, `purge`, or copy *to* R2), editing existing result files, `git push`, `git
commit` (the user does their own git operations), `wandb login` or any wandb use (not needed), printing
or storing credentials anywhere in files or logs.

## 1. Background facts you must not lose

Source of truth: `note/README.unseen-morphology-results.md` (run of 2026-09-13).

- 32 held-out shapes in `protomotions/data/assets/mjcf/smpl_mor_interp/` (16 beta vectors x 2
  genders). Asset ids look like `male_95749fe2` / `female_1d4da893` (gender + beta key); verify against
  `protomotions/data/assets/mjcf/smpl_mor_interp/assets.yaml`.
- Motion data: R2 `humos_unseen_morphology_offset/`, 128 shards, merged to
  `data_cache/unseen_morphology_merged_0.pt` (14.5 GB), 1,024 clips x 32 shapes = 32,768 pairs.
- Clip split membership (`hhi_stage2_v1`): 932 clips are train-split, 47 validation, 45 test.
  **For the fulldata checkpoint the 92 validation/test clips are no longer unseen.** The headline
  comparison therefore uses only the 932 train-split clips (29,824 pairs). The 92 are reported
  separately, labelled "contaminated for the candidate", never pooled into the headline.
- Success = `max_gt_error <= 0.5 m` over the episode (max 210 steps). The analysis script's defaults
  are the original protocol: do not change `--max-episode-steps`, `--success-gt-error-threshold` or
  `--envs-per-asset` for the unseen runs.
- Original results to reproduce (train-split clips):

| Population | Success | mean gt_err |
|---|---:|---:|
| 28 clean shapes pooled | 99.75% | 0.093 m |
| `male_a83170f5` | 58.3% | |
| `male_cf3d5ee7` | 46.8% | |
| `female_4d1b9df1` | 35.8% | |
| `female_1d4da893` | 0.1% | |
| these four pooled | 35.25% | 0.76 m |
| `male_95749fe2` (mild, ~96.3% over all 1,024 clips) | ~96% | |

- Pooling all 32 shapes gives roughly 92%. The "99%" figure in conversation means the 28 clean shapes
  only. Always state which population a number refers to.
- Failing rows have `terminated=0` (silent drift, not falls).

## 2. Design

All evaluations use `tools/analyze_unseen_morphology_generalization.py` unchanged. Evaluation is
deterministic replay (policy mean action, no sampling).

**Checkpoints (all evaluated on both datasets):**

| Name | What | Role |
|---|---|---|
| `refined_run1` | refined checkpoint, `last.ckpt` | **Baseline** (frozen paper checkpoint) |
| `refined_run2` | the identical checkpoint, evaluated a second time | Reference: simulator nondeterminism |
| `ladder_*` (0-2) | other checkpoints from the refined run, as close to epoch 28,170 as exist | Reference: checkpoint-to-checkpoint variation |
| `fulldata` | fulldata polish checkpoint, one frozen file | **Candidate** |

**Datasets:**

| Code | Motion file | Shapes | Clips | Purpose |
|---|---|---|---|---|
| `unseen` | `data_cache/unseen_morphology_merged_0.pt` | 32 held-out | 1,024 (headline subset: 932 train-split) | the question |
| `ctrl` | `small150_128shape_refined.pt` (built in step 4) | 128 trained | 150, all forced into the train split | matched trained-body control |

The control compares each checkpoint to itself across datasets only through its own paired delta:
Delta(unseen) is set against Delta(trained), each paired within its own dataset. The two datasets
come from different refinement pipelines and that does not matter for a within-dataset paired delta.

## 3. Pre-flight (the user does this before starting you; verify each item)

1. Pod uses the project image `hansen1416/hhi-protomotions-isaacgym:v1` (IsaacGym preinstalled),
   at least 1 GPU with 24 GB or more, at least 100 GB free on `/workspace`.
2. `rclone` has a working remote called `r2` pointing at the Cloudflare R2 bucket `proto-data`
   (endpoint `https://a17f581e2d142fd42fd7169cd4c48c8c.r2.cloudflarestorage.com`). Credentials are the
   user's: if `rclone lsd r2:proto-data` fails, ask the user to run `rclone config` themselves. Do not
   ask them to paste secrets into the chat if avoidable.
3. The repo is cloned at `/workspace/ProtoMotions` on a branch that contains
   `tools/compare_checkpoints_unseen_morphology.py` and this file. If the script is missing, tell the
   user: they must push or copy it. Do not rewrite it from memory.
4. The fulldata checkpoint is reachable. You must know **which file** (`last.ckpt` or a named epoch)
   and **where** (a local path on this pod, or an R2 key under `ckpt/`). If the user has not said,
   ask. Also ask what `--training-max-steps` / how many epochs the polish ran, and whether training is
   still running anywhere. Do not guess.

Kickoff prompt for the user to paste:

> Read note/README.unseen-morphology-checkpoint-comparison.md and follow it exactly. The fulldata
> checkpoint is at <PATH OR R2 KEY>. Ask me before anything the document says to ask about. Do not
> commit or push.

## 4. Procedure

Work from `/workspace/ProtoMotions`. Use `tmux` (or `nohup ... &`) for anything longer than a few
minutes and tail the log instead of polling in a loop.

### Step 0: environment check

```bash
nvidia-smi
python -c "from protomotions.utils.simulator_imports import import_simulator_before_torch as f; f('isaacgym'); import torch; print(torch.cuda.is_available())"
rclone lsd r2:proto-data | head
df -h /workspace
git status --short && git log --oneline -3
export OUT=results/analysis/ckpt_compare_$(date +%F) && mkdir -p $OUT && echo $OUT
```

Stop and report if IsaacGym does not import or CUDA is unavailable.

### Step 1: split ids and their integrity

`*_ids.txt` are not in git (they live under a versioned R2 prefix); `split_metadata.json` is tracked
(possibly as a git-lfs pointer, check that it is JSON, run `git lfs pull` if not).

```bash
mkdir -p data/splits/hhi_stage2_v1
rclone lsf r2:proto-data/hhi_stage2_per_clip_refined/splits/hhi_stage2_v1/
rclone copy r2:proto-data/hhi_stage2_per_clip_refined/splits/hhi_stage2_v1/ data/splits/hhi_stage2_v1/ \
    --include "*_ids.txt" --include "split_metadata.json" --s3-no-check-bucket
wc -l data/splits/hhi_stage2_v1/*_ids.txt
python - <<'EOF'
import hashlib, json
d = "data/splits/hhi_stage2_v1/"
meta = json.load(open(d + "split_metadata.json"))
for split, info in meta["id_lists"].items():
    h = hashlib.sha256(open(d + info["file"], "rb").read()).hexdigest()
    print(split, info["clip_count"], "OK" if h == info["sha256"] else "HASH MISMATCH")
EOF
cat data/splits/hhi_stage2_v1/validation_ids.txt data/splits/hhi_stage2_v1/test_ids.txt > $OUT/valtest_ids.txt
```

If a hash mismatches or the lists are missing, stop and ask the user. (The split is deterministic and
regenerable with `tools/create_hhi_split_manifests.py`, but only do that with the user's approval.)

### Step 2: checkpoints, frozen and identified

Never evaluate a path that something else might still overwrite. Copy each checkpoint into its own
frozen directory together with its configs, record identity, and evaluate the copy.

```bash
ls -la results/hhi_wide_stage2_discover_attention_slot_type_refined/ 2>/dev/null
# refined, if absent:
rclone copyto r2:proto-data/ckpt/hhi_wide_stage2_discover_attention_slot_type_refined.zip \
    /workspace/refined.zip --s3-no-check-bucket -P && (cd results && unzip -n /workspace/refined.zip)
# fulldata: only from the location the user gave you. List R2 first if it is an R2 key.
rclone lsf r2:proto-data/ckpt/ | grep -i -E "fulldata|refined"
```

Freeze (repeat per checkpoint; `NAME` in `refined`, `fulldata`, `ladder_a`, ...):

```bash
freeze () {  # freeze NAME SRC_CKPT
  d=results/frozen/$1; mkdir -p $d
  cp "$2" $d/model.ckpt
  cp "$(dirname "$2")/resolved_configs_inference.pt" $d/ || echo "MISSING resolved_configs_inference.pt for $1"
  cp "$(dirname "$2")/resolved_configs.pt" $d/ 2>/dev/null
  sha256sum $d/model.ckpt | tee $d/model.ckpt.sha256
}
freeze refined results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt
freeze fulldata <PATH TO THE FULLDATA CKPT THE USER NAMED>
```

The analysis script reads `<checkpoint dir>/resolved_configs_inference.pt`, so each frozen directory
needs that file. For ladder checkpoints in the same source directory, the same configs apply.

Identity and sanity record, written to `$OUT/checkpoint_identity.txt`:

```bash
python - <<'EOF' | tee $OUT/checkpoint_identity.txt
from protomotions.utils.simulator_imports import import_simulator_before_torch as f; f("isaacgym")
import torch, glob, os
for d in sorted(glob.glob("results/frozen/*")):
    ck = torch.load(d + "/model.ckpt", map_location="cpu", weights_only=False)
    print(d, "size", os.path.getsize(d + "/model.ckpt"), "epoch", ck.get("epoch"),
          "step_count", ck.get("step_count"), "keys", list(ck.keys())[:8])
    try:
        c = torch.load(d + "/resolved_configs_inference.pt", map_location="cpu", weights_only=False)
        ml = c["motion_lib"]
        print("   robot asset folder:", c["robot"].asset.asset_folder_name)
        print("   motion lib:", {k: getattr(ml, k, None) for k in ("manifest_name", "validation_manifest_name", "eval_holdout_size")})
    except Exception as e:
        print("   config read failed:", repr(e))
EOF
```

Verify and write down: `refined` epoch is 28,170 (the paper checkpoint; if it differs, stop and ask);
`fulldata` epoch and `step_count` (the polish was a warm start, so the epoch counter may continue from
28,170 or restart; report what you see, derive the polish length only if the user confirms how the
counter behaves); `fulldata` motion manifest is `.../full_manifest.jsonl` with no validation manifest; neither
checkpoint's robot asset folder is `smpl_mor_interp` (the eval script repoints it itself). Ask the
user to confirm the fulldata identity (file, epoch, budget) from this record before spending GPU time.

**Ladder (noise band).** List the other `*.ckpt` files in the refined results directory. Pick up to
two with epochs as close to 28,170 as exist (prefer one within the last ~10% of training and one
earlier). If none exist, say so in the report: the noise band then rests on the rerun alone, which only
measures simulator nondeterminism.

### Step 3: the unseen-body motion file

```bash
ls -la data_cache/unseen_morphology_merged_0.pt 2>/dev/null && echo "ALREADY HERE"
mkdir -p /workspace/unseen_morph_shards
rclone copy r2:proto-data/humos_unseen_morphology_offset/ /workspace/unseen_morph_shards/ \
    --transfers=4 --multi-thread-streams=16 --multi-thread-chunk-size=128M --s3-no-check-bucket -P
python tools/merge_motion_shards.py --src /workspace/unseen_morph_shards --dst data_cache \
    --num-shards 1 --pattern "humos_32768_*_offset.pt" --out-prefix unseen_morphology_merged
ls -la data_cache/unseen_morphology_merged_0.pt     # expect about 14.5 GB
```

Delete the shard directory afterwards only if disk is tight, and only `/workspace/unseen_morph_shards`.

### Step 4: the trained-body control motion file

The 150-clip small corpus is difficulty-stratified and was forced into the train split when the
`hhi_stage2_v1` split was made.

```bash
python tools/build_small_multishape_subset.py --num-clips 150 \
    --output /workspace/motion_cache/small150_128shape_refined.pt
# sidecar: /workspace/motion_cache/small150_128shape_refined.clip_ids.txt
comm -23 <(sort /workspace/motion_cache/small150_128shape_refined.clip_ids.txt) \
         <(sort data/splits/hhi_stage2_v1/train_ids.txt) | wc -l     # must print 0
```

If any control clip is not in the train split, stop and ask. Expected motions: 150 x 128 = 19,200.

### Step 5: smoke tests (tiny; must pass before any full run)

```bash
CK=results/frozen/refined/model.ckpt
python tools/compare_checkpoints_unseen_morphology.py --self-test        # must end with SELF-TEST PASSED
python tools/analyze_unseen_morphology_generalization.py --checkpoint $CK \
    --motion-file data_cache/unseen_morphology_merged_0.pt \
    --max-clips 3 --envs-per-asset 3 --max-episode-steps 50 --output $OUT/smoke_unseen.csv
python tools/analyze_unseen_morphology_generalization.py --checkpoint $CK \
    --motion-file /workspace/motion_cache/small150_128shape_refined.pt \
    --unseen-asset-folder mjcf/smpl_mor/ --unseen-asset-info mjcf/smpl_mor/assets.yaml \
    --max-clips 3 --envs-per-asset 3 --max-episode-steps 50 --output $OUT/smoke_ctrl.csv
python - <<'EOF'
import csv, os
out = os.environ["OUT"]
for f, shapes in (("smoke_unseen.csv", 32), ("smoke_ctrl.csv", 128)):
    rows = list(csv.DictReader(open(f"{out}/{f}")))
    print(f, len(rows), "rows,", len({r['asset_id'] for r in rows}), "shapes (expect", shapes, ")")
EOF
```

Fix real failures. `--self-test` was written without being executed on the author's machine, so if it
fails it is probably a bug in the helper, not in the data: fix it minimally and tell the user what you
changed. If the control smoke test fails on asset paths or ids, fall back to adapting
`tools/analyze_morphology_motion_interaction.py` (same 150 x 128 corpus, trained assets) and tell the user.

### Step 6: baseline run 1 and the reproduction gate

Define the runners once (keep the settings identical across every run of a dataset):

```bash
run_unseen () {  # run_unseen NAME FROZEN_CKPT
  python tools/analyze_unseen_morphology_generalization.py --checkpoint "$2" \
    --motion-file data_cache/unseen_morphology_merged_0.pt --output $OUT/unseen_$1.csv \
    > $OUT/unseen_$1.log 2>&1
}
run_ctrl () {    # run_ctrl NAME FROZEN_CKPT
  python tools/analyze_unseen_morphology_generalization.py --checkpoint "$2" \
    --motion-file /workspace/motion_cache/small150_128shape_refined.pt \
    --unseen-asset-folder mjcf/smpl_mor/ --unseen-asset-info mjcf/smpl_mor/assets.yaml \
    --envs-per-asset 32 --output $OUT/ctrl_$1.csv > $OUT/ctrl_$1.log 2>&1
}
```

Run the baseline first, in a tmux pane (the functions above live in that shell, so run it there and
tail `$OUT/unseen_refined_run1.log` from a second pane):

```bash
run_unseen refined_run1 results/frozen/refined/model.ckpt     # expect 32,768 rows
```

**Gate.** On the 932 train-split clips, per-shape success for `refined_run1` must be close to the
section 1 table: each of the four flagged shapes within 5 percentage points, `male_95749fe2` within
5 pp of ~96%, and all remaining shapes pooled within 0.5 pp of 99.75%. Compute it:

```bash
python - <<'EOF'
import csv, os
out = os.environ["OUT"]
train = set(open("data/splits/hhi_stage2_v1/train_ids.txt").read().split())
by = {}
for r in csv.DictReader(open(f"{out}/unseen_refined_run1.csv")):
    if r["clip_id"] in train:
        by.setdefault(r["asset_id"], []).append(int(r["success"]))
rates = {a: sum(v) / len(v) for a, v in by.items()}
for a, s in sorted(rates.items(), key=lambda kv: kv[1])[:8]:
    print(f"{a:22s} {100 * s:6.2f}%  n={len(by[a])}")
flag = {"male_a83170f5", "male_cf3d5ee7", "female_4d1b9df1", "female_1d4da893", "male_95749fe2"}
print("flagged ids present:", sorted(flag & set(by)))      # all five must appear; fix the ids if not
others = [v for a, v in by.items() if a not in flag]
print("remaining shapes pooled (%):", 100 * sum(map(sum, others)) / sum(map(len, others)), "n shapes", len(others))
# Expect 27 remaining shapes at ~99.75-99.8%. The opposite-gender twins of the flagged shapes are
# among them and should each sit near 100%.
EOF
```

If the gate fails, **stop**, report the numbers and the checkpoint identity, and ask the user. Do not
continue with a baseline that does not reproduce. If an original CSV is available
(`data_cache/unseen_morphology_generalization.csv` or `.joined.csv`; ask the user), also confirm the
row count (32,768) and compare row by row on `success`.

### Step 7: the remaining runs

Run sequentially (one GPU). Order matters only in that the candidate comes after the gate.

```bash
run_unseen refined_run2 results/frozen/refined/model.ckpt     # same checkpoint again
run_unseen fulldata     results/frozen/fulldata/model.ckpt
# ladder checkpoints you froze in step 2, e.g.:
# run_unseen ladder_a results/frozen/ladder_a/model.ckpt
run_ctrl refined_run1 results/frozen/refined/model.ckpt
run_ctrl refined_run2 results/frozen/refined/model.ckpt
run_ctrl fulldata     results/frozen/fulldata/model.ckpt
# run_ctrl ladder_a ...  (same ladder checkpoints as on the unseen set)
```

Row counts to expect: unseen 32,768 per run; ctrl 150 x 128 = 19,200 per run. If a run is short of rows
or crashed (GPU OOM: lower `--envs-per-asset` consistently for **every** run of that dataset including
a fresh baseline run, and tell the user), do not compare it.

### Step 8: analysis

Headline (train-split 932 clips) with the five flagged shapes called out. Group ids must be verified
against `assets.yaml` (use the exact `asset_id` strings that appear in the CSV):

```bash
python tools/compare_checkpoints_unseen_morphology.py \
  --baseline refined_run1=$OUT/unseen_refined_run1.csv \
  --reference refined_run2=$OUT/unseen_refined_run2.csv \
  --candidate fulldata=$OUT/unseen_fulldata.csv \
  --clip-ids-file data/splits/hhi_stage2_v1/train_ids.txt \
  --group flagged4=male_a83170f5,male_cf3d5ee7,female_4d1b9df1,female_1d4da893 \
  --group mild=male_95749fe2 \
  --output-prefix $OUT/compare_unseen_train932 | tee $OUT/compare_unseen_train932.txt
# add one  --reference ladder_a=$OUT/unseen_ladder_a.csv  per ladder run
```

Contaminated subset, reported separately:

```bash
python tools/compare_checkpoints_unseen_morphology.py ...same runs... \
  --clip-ids-file $OUT/valtest_ids.txt --output-prefix $OUT/compare_unseen_valtest92 \
  | tee $OUT/compare_unseen_valtest92.txt
```

Trained-body control (all 150 clips are train-split; success is near ceiling, so read the error deltas
first):

```bash
python tools/compare_checkpoints_unseen_morphology.py \
  --baseline refined_run1=$OUT/ctrl_refined_run1.csv \
  --reference refined_run2=$OUT/ctrl_refined_run2.csv \
  --candidate fulldata=$OUT/ctrl_fulldata.csv \
  --bottom-k 8 --print-shapes 20 \
  --output-prefix $OUT/compare_ctrl150 | tee $OUT/compare_ctrl150.txt
```

`bottom8_by_baseline` is chosen from the baseline run alone, so it is exposed to regression to the
mean: say that in the report, and prefer the `all` row for conclusions.

### Step 9: interpret by these rules (fixed in advance; do not adapt them to the results)

1. **Flagged shapes.** For `flagged4` (and `mild`), the candidate moved beyond the noise band only if
   the script prints `beyond noise band: True` (its CI excludes 0 **and** its |delta| exceeds the
   largest |delta| of any reference run). Otherwise report "not distinguishable from checkpoint
   variation". Also list per-shape deltas, since pooled numbers hide a single shape moving.
2. **Regression.** Report every shape whose success CI lies entirely below -2 pp. A clean "no
   regression among the other 27" is a finding.
3. **Broad versus concentrated.** Compare candidate deltas, unseen `rest` (the 27 non-flagged shapes)
   against trained `all` and `bottom8`, on `mean_gt_error` and success. Similar sign and size on both
   datasets means broad tracking change. A change in `flagged4` or `rest` much larger than the trained
   control means concentrated on unseen bodies. A trained-body delta smaller than its own reference
   noise is not evidence of anything.
4. **The 92 validation/test clips** are shown only in their own table, labelled contaminated for the
   candidate. Do not average them into the headline.
5. **Do not re-select shapes** using the new results. The flagged set is fixed from the 2026-09-13 run.
6. **Wording.** Use "checkpoint comparison". Do not say the improvement is due to more training or to
   more motion coverage. Do not say the result confirms or rules out an asset defect.

### Step 10: deliverables

Write `$OUT/REPORT.md` and copy it to `note/README.unseen-morphology-checkpoint-comparison-results.md`
(untracked; the user will commit it). Contents, in this order:

1. One-paragraph answer, with the population each number refers to.
2. Provenance: branch and commit, the identity table (file, sha256, epoch, size) for every checkpoint,
   exact commands run, wall-clock per run, row counts, any deviation from this document.
3. Gate result (baseline reproduction) as a table next to the section 1 numbers.
4. Headline table: for `flagged4`, each flagged shape, `mild`, `rest`, `all`: baseline %, candidate %,
   delta with 95% CI, pairs fixed/broken, gt/gr deltas. Reference-run deltas beside it.
5. Regression check for the other 27 shapes.
6. Trained-body control table and the broad-versus-concentrated reading under rule 3.
7. The 92-clip contaminated table, labelled.
8. Limitations: ladder availability, single seed, one candidate checkpoint, nondeterminism, the 4
   flagged shapes not independent samples of "shape space", shape deltas are per-shape facts.
9. Files list. Raw CSVs and logs stay in `$OUT`.

Final chat message to the user: three to five lines, the answer plus the path to the report. Do not
commit or push anything.

## 5. Troubleshooting

- **IsaacGym before torch.** Any ad hoc Python that loads a checkpoint or a resolved config must call
  `import_simulator_before_torch("isaacgym")` before `import torch`.
- **`resolved_configs_inference.pt` missing** next to a checkpoint: copy it from the training results
  directory of the same run. Never borrow it from a different experiment.
- **GPU out of memory:** lower `--envs-per-asset` (unseen default 128 gives 4,096 envs). Because envs
  are independent the metrics should not change, but keep it identical for every run of that dataset
  and re-run the baseline if you changed it after step 6.
- **Fewer rows than expected:** a clip shorter than 2 steps is skipped by the script. The compare helper
  drops clips that lack any shape in any run and prints how many.
- **rclone slow or throttled:** add `--retries=10 --retries-sleep=30s` and lower `--transfers`.
- **Unsure about anything not covered here:** ask the user. Do not improvise on checkpoint identity,
  split membership, or what gets written to R2.
