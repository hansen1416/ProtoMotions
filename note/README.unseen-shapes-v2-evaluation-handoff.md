# Handoff: evaluate two checkpoints on the new unseen-shape datasets (inside / outside)

Audience: a Claude Code session on a RunPod IsaacGym pod (root in a container), started by the user (hansen1416).
Written 2026-10-09. Read this whole file first. It builds on a procedure that already exists in the repo, so also read
`note/README.unseen-morphology-checkpoint-comparison.md` (the 2026-10-06 plan; it is on branch `feature/hhi`) and
reuse its mechanics. Where this file and that one differ, this file wins for the new datasets. Do the steps in
order and **stop where a gate says stop**. Nothing here has been executed by the author.

## 0. Question and design

Does the policy track bodies it never trained on, and how does that change with distance outside the training
range? Two checkpoints under one protocol, as before:

| Name | Checkpoint | Role |
|---|---|---|
| `refined_run1`, `refined_run2` | `hhi_wide_stage2_discover_attention_slot_type_refined` (the frozen paper checkpoint, expected epoch 28,170), evaluated twice | **Baseline**; the rerun measures simulator nondeterminism |
| `fulldata` | `hhi_wide_stage2_discover_attention_slot_type_fulldata`, file **`epoch_34000.ckpt`** | **Candidate**. The file was pre-registered on 2026-10-06 in `note/README.unseen-morphology-fulldata-evaluation.md`, before any unseen-body result existed. Do not pick another file by looking at results |

Sources on R2 (`r2:proto-data/ckpt/`): `hhi_wide_stage2_discover_attention_slot_type_refined.zip` (2.5 GB) and
`hhi_wide_stage2_discover_attention_slot_type_fulldata.zip` (3.9 GB). `fulldata` trained on all 20,951 clips, including the
validation and test clips, so those are "contaminated" for it.

Datasets (both 256 clips x 128 shapes = 32,768 pairs; identical clips): `inside` (64 new betas x 2 genders, uniform
in [-3,3]) and `outside` (64 betas x 2 genders in four tiers with max |beta| in (3,3.5], (3.5,4], (4,4.5], (4.5,5]).
Clips: 233 train, 12 validation, 11 test, recorded in `clips_256_keyids.json`. **Headline population = the 233
train-split clips** (clean for both checkpoints). The 23 validation/test clips are reported separately, labelled
contaminated for `fulldata`. Plus a trained-body control (150 clips x 128 trained shapes) exactly as in the 10-06 note.
This is a **checkpoint comparison**: training length and clip coverage changed together, so never attribute a
difference to either alone.

## 1. Rules

- No `git commit/push/pull/checkout/stash`; do not modify tracked files (new files only; if a tool has a real bug,
  fix it minimally and tell the user exactly what changed). No wandb, no training.
- R2: read freely; **writes only `rclone copy` into `r2:proto-data/unseen_shapes_v2/eval/`** after checking it is
  empty. Never `sync/delete/move/purge`; no other prefix. Never print credentials.
- Any ad hoc Python that loads a checkpoint or config must call
  `from protomotions.utils.simulator_imports import import_simulator_before_torch as f; f("isaacgym")` before `import torch`.
- Analysis groups are **fixed before any result is seen** (section 5) and never re-selected afterwards.
- Long runs in `tmux` or `nohup ... &` with a log; sequential on the single GPU. Record wall-clock per run.
- Ask the user before anything not covered here, and before spending GPU on extra checkpoints (the "ladder").

## 2. Inputs (verify; fetch from R2 if missing)

1. The dataset-build job should have produced, per dataset `D` in `inside outside`:
   `r2:proto-data/unseen_shapes_v2/<D>_refined/` (4 refined MotionLib shards of 8,192 motions, plus
   `dataset_manifest.json`). If these prefixes do not exist yet, stop and tell the user; do not evaluate unrefined or
   partial data.
2. Body assets in the repo (untracked): `protomotions/data/assets/mjcf/smpl_mor_unseen_{inside,outside}/` with 128 XMLs
   and an `assets.yaml` each. Missing? `rclone copy r2:proto-data/unseen_shapes_v2/xml/smpl_mor_unseen_<D>
   protomotions/data/assets/mjcf/smpl_mor_unseen_<D> --s3-no-check-bucket`.
3. Metadata: `rclone copy r2:proto-data/unseen_shapes_v2/inputs /workspace/unseen_v2/inputs` (`clips_256_keyids.json`,
   `beta_manifest.csv`) and `r2:proto-data/unseen_shapes_v2/physical_report.csv`.
4. Environment: IsaacGym imports, GPU with at least 40 GB, at least 100 GB free (two merged motion files are about
   14.5 GB each; delete shard directories after merging if tight, never the merged files), `rclone lsd r2:proto-data` works.

## 3. Setup

```bash
cd /workspace/ProtoMotions && export OUT=results/analysis/unseen_v2_$(date +%F) && mkdir -p $OUT/groups && echo $OUT
cp /workspace/unseen_v2/inputs/clips_256_keyids.json $OUT/   # also copy beta_manifest.csv, physical_report.csv here
python - <<'EOF'
import json, os
out = os.environ["OUT"]
c = json.load(open(f"{out}/clips_256_keyids.json"))
tr = sorted(k for k, v in c.items() if v["split"] == "train")
vt = sorted(k for k, v in c.items() if v["split"] != "train")
open(f"{out}/train233_ids.txt", "w").write("\n".join(tr) + "\n"); open(f"{out}/valtest23_ids.txt", "w").write("\n".join(vt) + "\n")
print(len(tr), len(vt))        # must print 233 23
EOF
```

**Checkpoints (frozen and identified).** Follow step 2 of the 10-06 note (download the two zips into
`/workspace`, `unzip -l` them first, extract under `results/`, then its `freeze` function and the
`checkpoint_identity.txt` snippet). Required checks, then **ask the user to confirm** before any GPU run:
- `refined`: use `last.ckpt`; epoch must be 28,170 (else stop and ask).
- `fulldata`: the zip must contain `epoch_34000.ckpt` next to `resolved_configs_inference.pt`. If it does not, or if
  it holds several candidates, stop and ask which file; do not choose using results. Report its epoch and `step_count`.
- Each frozen directory needs its own `resolved_configs_inference.pt` (from the same run, never another experiment).

**Merged motion files.** For each `D`:

```bash
mkdir -p /workspace/shards_$D && rclone copy r2:proto-data/unseen_shapes_v2/${D}_refined/ /workspace/shards_$D/ --include "*.pt" --transfers=4 --s3-no-check-bucket -P
python tools/merge_motion_shards.py --src /workspace/shards_$D --dst data_cache --num-shards 1 --pattern "*_refined.pt" --out-prefix unseen_v2_$D
# expect data_cache/unseen_v2_${D}_0.pt, about 14.5 GB. If the shard filenames differ, adapt the pattern; the folder must hold only that dataset's shards.
```

Gate S: loading either merged file must show 32,768 motions, 256 unique clips, 128 unique shapes, every
(clip, shape) pair present (the analysis script itself raises on missing pairs), asset ids matching the dataset's
`assets.yaml`.

**Trained-body control file** and **noise-band ladder**: build exactly as steps 3 to 4 of the 10-06 note. Ask the user
whether to include ladder checkpoints before using GPU time on them.

## 4. Runs

Protocol is unchanged: deterministic replay, success = max position error at most 0.5 m over at most 210 steps
(script defaults; do not change `--max-episode-steps` or `--success-gt-error-threshold`). The only change for 128 shapes:
**`--envs-per-asset 32`**, which gives 128 x 32 = 4,096 environments and 8 rounds (256 / 32), the same load as the
09-13 run. Use the identical setting for every run of a dataset.

```bash
run_v2 () {  # run_v2 DATASET NAME FROZEN_CKPT
  python tools/analyze_unseen_morphology_generalization.py --checkpoint "$3" \
    --motion-file data_cache/unseen_v2_$1_0.pt \
    --unseen-asset-folder mjcf/smpl_mor_unseen_$1/ --unseen-asset-info mjcf/smpl_mor_unseen_$1/assets.yaml \
    --envs-per-asset 32 --output $OUT/$1_$2.csv > $OUT/$1_$2.log 2>&1
}
```

Smoke test first per dataset (add `--max-clips 3 --envs-per-asset 3 --max-episode-steps 50`; expect 3 x 128 = 384 rows
and 128 shapes). Fix real failures. **Gate R (sanity before the long runs):** `python tools/compare_checkpoints_unseen_morphology.py --self-test`
passes, and the trained-body control for `refined` gives success near ceiling (the 10-06 note says to read error
deltas first). Then, sequentially: for `D` in `inside`, `outside`: `refined_run1`, `refined_run2`, `fulldata`; then the
trained-body control for the same three checkpoints; then any ladder runs the user approved.

Expected rows: 32,768 per run (19,200 for the control). A short or crashed run is not compared. If a GPU OOM forces a
lower `--envs-per-asset`, change it for every run of that dataset and rerun the baseline. **Gate N:** `refined_run1`
vs `refined_run2` per-pair `success` should agree on nearly every pair (report the fraction and the pooled success
of each); if they differ materially, report it before interpreting anything.

Optional but recommended: if the user supplies `data_cache/unseen_morphology_generalization.joined.csv` (the 09-13
baseline, 32,768 rows) and you download the old merged file as in step 3 of the 10-06 note, rerun the 32 old shapes
with `refined` and compare to that CSV (28 clean shapes about 99.75%; four flagged shapes 36 to 58%). It proves the
checkpoint and tooling reproduce the earlier result. Ask the user before downloading the 14.5 GB.

## 5. Pre-registered groups (write now, from the metadata, before looking at any result)

Asset ids are `<gender>_<beta_key>`, both genders of each key. Build one text file per group under `$OUT/groups/`
(check the comparator's group-file format in its code or `--help` first) from `beta_manifest.csv` and `physical_report.csv`:

- `outside_tier0` ... `outside_tier3`: all 32 assets of that tier (`tier` column of the manifest).
- `outside_ratio_out`: outside betas whose male/female mass ratio at the same beta lies outside **0.56 to 3.21**
  (the training range of that ratio; about 13 of 64 betas), both genders each.
- `mass_out` and `height_out`: assets (any dataset) whose total mass is outside 26.4 to 144.4 kg, or height outside
  1.13 to 1.67 m, from `physical_report.csv`.
- `all` is automatic; `rest` is the automatic remainder; add `--bottom-k 8` and note it is chosen from the baseline alone
  (regression to the mean), so prefer `all` for conclusions.

## 6. Analysis

Per dataset, headline on the 233 train-split clips:

```bash
python tools/compare_checkpoints_unseen_morphology.py \
  --baseline refined_run1=$OUT/<D>_refined_run1.csv --reference refined_run2=$OUT/<D>_refined_run2.csv \
  --candidate fulldata=$OUT/<D>_fulldata.csv --clip-ids-file $OUT/train233_ids.txt \
  --group ... (the group files above, with the groups that apply to <D>) \
  --bottom-k 8 --output-prefix $OUT/compare_<D>_train233 | tee $OUT/compare_<D>_train233.txt
```

Repeat with `--clip-ids-file $OUT/valtest23_ids.txt` (separate table, labelled contaminated for `fulldata`), and for the
trained-body control as in the 10-06 note. Add `--reference` rows for any ladder runs.

**Per-shape table** (descriptive; this is the degradation-curve data): join the `refined_run1` and `fulldata` per-shape
success and mean position error with `beta_manifest.csv` (tier, max |beta|, L2, nearest-trained distance) and
`physical_report.csv` (mass, height, male/female ratio) into `$OUT/per_shape_<D>.csv`. Report success against
tier, mass and height. Beta distance alone is not the physical story: only about 10 of 128 outside bodies are beyond the
training mass range.

**Interpretation rules (fixed in advance):**
1. A candidate change counts only if the comparator prints `beyond noise band: True` for that group (CI excludes 0
   and |delta| exceeds every reference run's |delta|). Otherwise "not distinguishable from checkpoint variation".
2. Report every shape whose success CI lies entirely below -2 pp. A clean "no regression" is a finding.
3. Compare candidate deltas on `inside`/`outside` against the trained-body control: similar sign and size means broad
   tracking change; larger on new bodies means concentrated on new bodies.
4. Never pool the 23 validation/test clips into the headline.
5. Wording: "checkpoint comparison". Do not claim more training or more coverage is the cause, and do not claim a result
   confirms or rules out an asset defect or a bad HUMOS reference. Failures on `outside_ratio_out` shapes are an
   open question (extreme male/female asymmetry), not an explanation.

## 7. Deliverables

Write `$OUT/REPORT.md` and copy it to `note/README.unseen-shapes-v2-evaluation-results.md` (untracked; the user commits).
Order: (1) one-paragraph answer, stating the population each number refers to; (2) provenance: commit, identity table
(file, sha256, epoch, size) per checkpoint, commands, wall-clock per run, row counts, deviations; (3) gates S, R, N (and the
old-set reproduction if done); (4) headline tables per dataset and group with CIs and noise band; (5) per-shape and
tier/mass/height summaries; (6) trained-body control; (7) the 23-clip contaminated table, labelled; (8) limitations (single
seed, ladder availability, nondeterminism, tier groups are not independent samples of shape space). Then upload CSVs,
logs, per-shape tables and the report with `rclone copy` to `r2:proto-data/unseen_shapes_v2/eval/` (checked empty) and
verify with `rclone size`. Final chat message: three to five lines with the answer and the report path.
