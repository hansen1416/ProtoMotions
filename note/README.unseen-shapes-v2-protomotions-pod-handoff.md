# Handoff for the ProtoMotions pod: finish the two unseen-shape datasets

Audience: a Claude Code session running on a RunPod pod (root in a container), started by the user (hansen1416).
Written 2026-10-09. You cannot see any other notes, so this file is self-contained. Read it all before running
anything, do the steps in order, and **stop where a gate says stop**. Nothing below has been run by the author
except what is listed under "State". Treat any mismatch as information, not something to paper over.

## 0. Goal

Two datasets, `inside` and `outside`, each 256 clips x 128 body shapes (64 betas x 2 genders) = 32,768 motions.
The raw HUMOS motions and the body XMLs already exist. Your job is the ProtoMotions side: raw HUMOS `.pt` ->
AMASS-style NPZ -> `.motion` -> MotionLib shards -> frame-0 grounding offset (IsaacGym) -> refinement (CPU) ->
verify -> upload. Final product per dataset: **4 refined MotionLib shards of 8,192 motions** (64 clips x 128
shapes each), plus the offset-only shards, on R2. No training, no evaluation, no conclusions about results.

## 1. State when you start

Pod: A40 46 GB, **96 CPU cores**, Python 3.8.10, IsaacGym imports fine, torch 2.4.1+cu121 (cuda True), numpy 1.24.4,
scipy 1.10.1, about 100 GB free on `/workspace`. Repo `/workspace/ProtoMotions` on branch `feature/hhi`
(a558984, clean except the untracked new asset folders). `rclone` has the `r2:` remote (bucket `proto-data`).

| Item | Location |
|---|---|
| Raw HUMOS output, 256 `.pt` per dataset (each clip holds 64 beta keys x male/female) | `/workspace/unseen_v2/humos_raw/{inside,outside}/` |
| Betas, clip list, manifest | `/workspace/unseen_v2/inputs/` (`all_betas_unseen_*.pt`, `clips_256_keyids.json`, `beta_manifest.csv`) |
| Body XMLs + `assets.yaml` (128 each, untracked) | `protomotions/data/assets/mjcf/smpl_mor_unseen_inside/` and `smpl_mor_unseen_outside/` |
| `inside` NPZ export, **done** | `/workspace/unseen_v2/inside/npz/HUMOS/*.npz` (32,768 files, about 2 GB) and `/workspace/unseen_v2/inside/npz/unseen_v2_inside.yaml` |
| `inside` `.motion` conversion | **partial**: about 330 files exist in `npz/HUMOS/` (the 5 newest were deleted after a kill) |
| `outside` | nothing done yet |

The stock converter is a **single-process loop and too slow**: about 3 files/s, so about 3 hours per dataset,
using only about 6 of 96 cores. That is the reason this handoff exists. The first thing to do is parallelize it.

## 2. Rules

- **Do not modify any tracked file in the repo.** New files only (put your scripts in `/workspace/unseen_v2/scripts/`).
  **No `git commit`, `git push`, `git pull`, `git checkout`, `git stash`.** The user does git themselves.
- **R2 writes: only `rclone copy`, only into new prefixes under `r2:proto-data/unseen_shapes_v2/`** named in section 9,
  and only after checking the target prefix is empty. Never `sync`, `delete`, `deletefile`, `move`, `purge`, and never
  write into any other prefix. Never print or store credentials.
- No wandb, no training. Python is 3.8: no `X | Y` type hints, no `match`, no `list[int]` annotations.
- Run anything longer than about 2 minutes with `nohup ... > logfile 2>&1 &`, log under `/workspace/unseen_v2/<dataset>/`,
  and poll. Everything below is resumable unless noted.
- Hard failures get reported, never worked around by silently dropping shapes, clips or steps.
- Do not delete anything under `humos_raw/`, `inputs/` or the XML folders. Deleting intermediates is allowed only
  as described in section 8.

## 3. Step A: preflight (gate A)

Report one table; stop if a "must" fails.

1. `cd /workspace/ProtoMotions && git status --short` shows only the two untracked `smpl_mor_unseen_*` folders (must).
2. `ls protomotions/data/assets/mjcf/smpl_mor_unseen_inside/*.xml | wc -l` is 128, same for `outside`, each with `assets.yaml` (must).
3. `ls /workspace/unseen_v2/humos_raw/inside | wc -l` is 256, same for `outside` (must).
4. `find /workspace/unseen_v2/inside/npz -name '*.npz' | wc -l` is 32768 (must); `ls -la .../unseen_v2_inside.yaml` exists (must).
5. `nproc` is about 96, `df -h /workspace` has at least 60 GB free (must), `nvidia-smi` shows the A40.
6. Inspect the real formats before scripting: `head -40 /workspace/unseen_v2/inside/npz/unseen_v2_inside.yaml`;
   `sed -n 1,130p data/scripts/convert_amass_to_proto.py` (look at `load_motion_configs` and how it keys files);
   `sed -n 225,372p tools/convert_amass_to_motionlib_with_morphology.py` (the packaging wrapper).

## 4. Step B: export `outside` to NPZ (about 30 s)

```bash
cd /workspace/ProtoMotions && mkdir -p /workspace/unseen_v2/outside && python -u tools/export_humos_to_amass_npz.py \
  --input-dir /workspace/unseen_v2/humos_raw/outside --out-root /workspace/unseen_v2/outside/npz \
  --yaml-name unseen_v2_outside.yaml --fps 30.0 --skip-existing > /workspace/unseen_v2/outside/export.log 2>&1
```

These flags match the script that built the training corpus. Do **not** pass `--apply-offset-height` (the offset
step handles grounding). Gate: 32,768 `.npz` files and the YAML exists.

## 5. Step C: NPZ -> `.motion`, parallelized (the slow step)

Facts about `data/scripts/convert_amass_to_proto.py` (verify them in step A item 6):
- It is a typer CLI: `python data/scripts/convert_amass_to_proto.py <root> --humanoid-type smpl --output-fps 30 --motion-config <yaml>`.
  It must run **from the repo root** (it reads `protomotions/data/assets/mjcf/smpl_humanoid.xml` relatively).
- It walks every subfolder of `<root>` (here `HUMOS/`), globs `**/*.npz`, sorts, and converts one file at a time on CPU.
- It **skips a file whose `.motion` already exists** (unless `--force-remake`). The `.motion` is written next to the NPZ.

Recipe (adapt after reading the code; one dataset at a time, `inside` first):
1. Make the list of NPZ files that have no `.motion` yet. Zero-byte `.motion` files count as missing: delete them first.
2. Split the list round-robin into `W` groups (W = 32 is a good start with 96 cores).
3. For each group make `/workspace/unseen_v2/<D>/w<i>/HUMOS/` and **hard link** (`ln`) that group's NPZ files into it
   (same filesystem, no extra disk). The YAML path keys must still resolve: check how `load_motion_configs` keys
   entries and keep relative paths (`HUMOS/<name>.npz`) identical.
4. Launch W workers from the repo root, one thread each:
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python -u data/scripts/convert_amass_to_proto.py <wdir> --humanoid-type smpl --output-fps 30 --motion-config <yaml> > <wdir>.log 2>&1 &`
5. Poll the total `.motion` count. Expect a large speedup over 3 files/s; report the measured rate.
6. When all workers exit: **move** the new `.motion` files from each `w<i>/HUMOS/` into the original `npz/HUMOS/`
   (`mv`, same names), then delete only the `w<i>` directories (they hold hard links and the moved outputs; the
   originals stay). Never `rm -rf` the original `npz/HUMOS/`.
7. Gate C: `.motion` file count equals 32,768 per dataset and none is zero bytes. If a few are missing or empty,
   rerun the single-process converter on the dataset root once (it skips the finished ones).

## 6. Step D: package into MotionLib shards (CPU)

```bash
cd /workspace/ProtoMotions && nohup python -u tools/convert_amass_to_motionlib_with_morphology.py \
  /workspace/unseen_v2/<D>/npz /workspace/unseen_v2/<D>/proto \
  --motion-config /workspace/unseen_v2/<D>/npz/unseen_v2_<D>.yaml \
  --humanoid-type smpl --output-fps 30 --device cpu --batch-size 8192 > /workspace/unseen_v2/<D>/convert.log 2>&1 &
```

It first re-runs the `.motion` conversion (everything is skipped, so it only scans) and then packages. The
YAML lists whole clips contiguously (128 variants per clip) and 8192 = 64 x 128, so each shard should hold 64 whole
clips. The stage-2 corpus used exactly this batch size. If packaging hits a ZIP64 or memory error, use
`--batch-size 4096` (32 clips). Gate D: 4 `.pt` shards in `proto/`, each with 8192 motions, 64 distinct
`motion_clip_ids`, and exactly 128 motions per clip; `motion_asset_ids` all present in
`protomotions/data/assets/mjcf/smpl_mor_unseen_<D>/assets.yaml`. If packaging is slow, you may run the 4 shards
in parallel by giving each its own root of hard links and a filtered YAML, as in step C.

After Gate D passes, delete `<D>/npz/` (about 2 GB, and the `.motion` files) only if disk is below 40 GB free.

## 7. Step E: frame-0 grounding offset (IsaacGym, GPU)

For each shard (`compute_humos_frame0_offsets.py` imports IsaacGym before torch itself):

```bash
cd /workspace/ProtoMotions && python tools/compute_humos_frame0_offsets.py \
  --motion-file /workspace/unseen_v2/<D>/proto/<shard>.pt \
  --asset-root protomotions/data/assets/mjcf/smpl_mor_unseen_<D> \
  --out-motion-file /workspace/unseen_v2/<D>/offset/<shard>_offset.pt --overwrite
```

Smoke test first on one shard with `--limit 128` (read `--help`; write to a scratch output) to check IsaacGym
loads the new bodies. **The `outside` set has unusual bodies (tall, heavy, strongly gendered)**: if some
`asset_id` fails to load or produces NaN, record which ones and report them. Do not drop them silently. Run shards
one at a time on the single GPU, in the background, and report the time per shard.

**Why this step is required even though `refine_humos_motion.py`'s docstring says it replaces it.** In
`step5_ground_penetration_correction`, refinement only lifts a motion whose lowest point is below `TARGET_Z = 0.005`;
it never lowers a motion that starts floating. The training corpus was grounded with this offset first and refined
second. Keep that order.

## 8. Step F: refinement (CPU)

```bash
cd /workspace/ProtoMotions && python tools/refine_humos_motion.py \
  --motion-file /workspace/unseen_v2/<D>/offset/<shard>_offset.pt \
  --asset-root protomotions/data/assets/mjcf/smpl_mor_unseen_<D> \
  --out-motion-file /workspace/unseen_v2/<D>/refined/<shard>_refined.pt --device cpu --report \
  2>&1 | tee /workspace/unseen_v2/<D>/refined/<shard>_report.log
```

Smoke test with `--limit 128` first. The 4 shards can run in parallel (4 background processes, e.g.
`OMP_NUM_THREADS=8` each). Keep every `--report` log. Gate F: the refined shards have the same motion counts and
clip structure as the offset shards, no non-finite values in `gts, grs, gvs, gavs, dps, dvs, lrs`, and the report
shows penetration at about zero.

Optional (about 30 minutes of work, skip if the metric functions in `refine_humos_motion.py` are not reusable):
write `/workspace/unseen_v2/reference_quality_by_asset.csv` with per-`asset_id` mean stance-foot speed and
fraction of frames below the floor, on the offset-only and refined shards. It helps tell a bad HUMOS reference
(likely for extreme `outside` bodies) from a controller limit later.

**Disk policy (about 100 GB total, raw inputs take 18 GB).** Each stage per dataset is about 14 GB (4 shards).
Do `inside` completely, upload (section 9), verify the upload, then delete that dataset's `proto/` and `offset/`
and `refined/` before starting `outside`. Never delete a stage before its successor passed its gate and (for
the final ones) the upload was verified.

## 9. Step G: verify and upload

Final verification per dataset (refined shards): 4 shards x 8192 = 32,768 motions; 256 distinct clip ids matching
`/workspace/unseen_v2/inputs/clips_256_keyids.json` (233 train, 12 validation, 11 test; the file records each
clip's split); 128 motions per clip; 64 distinct beta keys, each with both genders; all asset ids in the dataset's
`assets.yaml`. Write `/workspace/unseen_v2/<D>/dataset_manifest.json` (shard names, sizes, counts, the git commit of
the repo checkout, and tool commands used).

Upload layout, each prefix **checked empty first** (`rclone lsf <prefix> --s3-no-check-bucket` prints nothing):

| Destination under `r2:proto-data/unseen_shapes_v2/` | Content |
|---|---|
| `<D>_offset/` | frame-0-only shards (control: same processing as the 2026-09-13 run) |
| `<D>_refined/` | final shards, `--report` logs, `dataset_manifest.json` (**the evaluation input**) |

```bash
[ -z "$(rclone lsf r2:proto-data/unseen_shapes_v2/<D>_refined/ --s3-no-check-bucket)" ] && \
  rclone copy /workspace/unseen_v2/<D>/refined r2:proto-data/unseen_shapes_v2/<D>_refined/ --transfers=4 --s3-no-check-bucket --progress
```

Verify each upload with `rclone size` (object count and bytes equal the pod's).

## 10. Report back to the user

One message: (1) a table of gates A to G with pass/fail and evidence; (2) the measured conversion rate before and
after parallelization and the time per stage; (3) any asset, clip or shard that failed anywhere (IsaacGym load,
NaN, refine error), by id; (4) exact R2 paths and sizes; (5) every deviation from this file and why. Do not
interpret results or write paper text.
