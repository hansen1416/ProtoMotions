# Instructions for the pod: build the two unseen-shape datasets (v2)

Written 2026-10-08. Audience: a Claude Code session running on a RunPod GPU machine that has the HUMOS
environment, started by the user (hansen1416). Read this whole file before running anything. Do the steps
in order and **stop where a gate says stop**. Nothing here has been executed yet: the author had no Python
available, so every command and every "expect N" below is derived from reading the code and notes, not from a
run. Treat a mismatch as information, not as something to paper over.

## 0. What this builds

Two MotionLib datasets for testing the frozen policy on body shapes it never trained on, replacing the
16-beta / 32-shape set used on 2026-09-13 (`note/README.unseen-morphology-results.md`).

| Dataset | Bodies | Meaning |
|---|---|---|
| `inside` | 64 new betas x 2 genders = 128 shapes | Uniform in [-3, 3]^10, the same sampling cube as the 128 trained shapes, but new draws. Held-out, same distribution. |
| `outside` | 64 new betas x 2 genders = 128 shapes | Every beta has max abs coordinate in (3, 5]. Four graded tiers of 16 betas (32 shapes) each, so success can be plotted against how far outside the training cube a body is. |

Both datasets use the **same 256 clips**, so inside vs outside is a paired comparison on identical motions.
Each dataset is 256 clips x 128 shapes = 32,768 motions (about 14.5 GB merged, same size as the 09-13 run).

This job builds data only. **No training, no evaluation, no conclusions about results.**

## 1. Locked design (confirmed with the user on 2026-10-08)

| Item | Decision |
|---|---|
| Training bodies (for distances) | 64 betas, uniform [-3, 3]^10, seed 46. Read them from `protomotions/data/assets/mjcf/smpl_mor/assets.yaml` (plain text). |
| Seeds | inside betas: **1001**. outside betas: **2001**. clip subsample: **31337**. (46 = training, 99 = the old 16-beta interp/extrap set. The new seeds must differ from both.) |
| inside sampling | `rng = np.random.default_rng(1001)`; 64 x 10 draws uniform on [-3, 3]. Prefer calling SMPLSim's own `sample_betas_uniform(batch_size=64, low=-3.0, high=3.0, rng=rng)` if importable, so the convention matches the training set. |
| outside sampling | `rng = np.random.default_rng(2001)`. Tier k = 0..3 has band (3 + 0.5k, 3.5 + 0.5k]. For each of 16 betas per tier: draw `u ~ U[-1,1]^10` and `r ~ U(lo, hi]`, then `beta = u * r / max(abs(u))`. This makes max abs coordinate exactly r. Cap is 5.0. |
| Beta key | SMPLSim's own convention: after drawing all betas, draw one key per beta from the same RNG with `deterministic_hex4(rng)` (`rng.integers(0, 2**32, dtype=np.uint32)` as 4 big-endian bytes, hex). Same as `sample_betas` in SMPLSim's `run_heldout.py`. Must be unique across the new 128 betas AND across every key already in `smpl_mor/assets.yaml` and `smpl_mor_interp/assets.yaml`. For the outside tiers, draw all keys after all betas, in tier order. |
| Genders | Both genders for every beta (same beta vector in male and female). Do not drop one: the male/female pairing is what exposed the asset-artifact failures on 09-13. |
| Clips | 256 clips = 233 train + 12 validation + 11 test, a stratified seeded subsample (seed 31337) of the 1,024 pilot clips used on 09-13 (932 / 47 / 45). Split labels come from `hhi_stage2_v1`. |
| Post-processing | **Frame-0 grounding offset (IsaacGym), then `tools/refine_humos_motion.py` (CPU)**, in that order. See section 8 for why refine alone is not enough. |
| Shards | `--batch-size 8192` = 64 clips x 128 shapes, so 4 shards per dataset, and all 128 shapes of a clip stay in one shard. |
| Asset folders | `protomotions/data/assets/mjcf/smpl_mor_unseen_inside/` and `.../smpl_mor_unseen_outside/` (128 XML + `assets.yaml` each). |
| Betas files | `protomotions/data/assets/all_betas_unseen_inside.pt` and `all_betas_unseen_outside.pt` (each exactly 64 entries). |

## 2. Rules

- **No `git commit`, `git push`, `git pull`, `git lfs push`.** The user does their own git operations. New files
  in the working tree are fine. Do not modify existing files in the repo, only add new ones.
- **R2 writes only under `r2:proto-data/unseen_shapes_v2/`**, and only with `rclone copy`. Never `sync`,
  `delete`, `deletefile`, `move`, `purge`, or copy into any other prefix. Never print or store credentials.
- No wandb, no training.
- Use `python -m pip`, never bare `pip` (the pod has several Pythons; this has bitten before).
- `torch.load(..., weights_only=False)` for every MotionLib / HUMOS file.
- One dataset at a time; delete intermediates (raw HUMOS output is uploaded first, NPZ is deleted after convert).
- A hard failure is reported, never worked around by silently dropping shapes, clips or steps.

## 3. Step 0 - Preflight (Gate A)

Check all of these, report the results in one table, and stop if any "must" fails.

1. `nvidia-smi` shows a GPU. `df -h /workspace`: **must have at least 100 GB free** (peak per dataset is
   roughly 9 GB raw + 28 GB NPZ + 15 GB raw MotionLib + 15 GB offset + 15 GB refined).
2. ProtoMotions repo at `/workspace/ProtoMotions` (or tell the user where it is), on a checkout that contains
   this file. HUMOS at `/workspace/humos`. If either is missing, stop and ask.
3. `grep -n "keyids-file" /workspace/humos/humos/infer.py` finds the flag. If not, the pod has the unpatched
   `infer.py`: fetch the patch with
   `rclone copy r2:proto-data/humos/humos_infer_patch.tar.gz . --s3-no-check-bucket` and extract it as
   described in `note/README.humos-runpod-setup.md` section 8. That tarball also contains `pilot_1024_motions.json`,
   which shows the expected keyids-file format.
   **Also check `humos/stats/humanml3d/humos3dfeats/` holds `*_mean.pt` and `*_std.pt` files.** `Normalizer.load()`
   lists that directory, so inference fails without it. It is gitignored and was not in the 2026-09-19 backup,
   so it may be missing. If it is, regenerate it with `humos/prepare/motion_stats.py` (CPU is fine with
   `--device cpu`; it reads `datasets/humos3dfeats` and the train split), then run the **stats reproduction
   check**: run `infer.py` on one pilot clip with the old interp betas file (`humos/all_betas_interp.pt`) and
   compare the output with the same clip's file in `r2:proto-data/humos_unseen_morphology/` (the 09-13 raw
   output, same checkpoint and betas): same beta key and gender, `pose_body`, `trans` and `root_orient` must
   agree to about 1e-4. If they do not, the regenerated stats differ from the originals: stop and report the
   max difference, because every clip generated with wrong stats would be subtly different from the training
   corpus.
4. `rclone lsf r2:proto-data/humos_unseen_morphology/ | wc -l` prints 1024 (this is the pilot clip list).
5. `rclone lsf r2:proto-data/unseen_shapes_v2/` is **empty or does not exist**. If it has content, stop and tell
   the user; do not overwrite.
6. The Python in the HUMOS env can `import torch, numpy, yaml, scipy, tqdm` and the ProtoMotions tools run:
   `cd /workspace/ProtoMotions && python tools/export_humos_to_amass_npz.py --help`. Install what is missing
   with `python -m pip`. Do **not** pip-install IsaacGym.
7. **IsaacGym check (decides step 8):** `python -c "from isaacgym import gymapi"` in the HUMOS env. If that
   fails, look for a separate conda env with IsaacGym (`conda env list`; `tools/prepare_stage2_data.py` has an
   `--isaacgym-env` option for exactly this split). Record which interpreter will run the offset step. If there
   is no IsaacGym anywhere on the pod, that is not a stop yet: do steps 1 to 7, upload the un-offset shards (see
   section 9), and tell the user that the offset+refine phase needs an IsaacGym pod.
8. **Install SMPLSim into the same Python env as HUMOS (needed for step 3).** The original work used one env
   (`smplsim`) for HUMOS inference, SMPLSim XML generation and NPZ conversion, so this is expected to work:
   ```bash
   cd /workspace
   git clone --branch feature/hhi https://github.com/hansen1416/SMPLSim.git
   python -m pip install -e /workspace/SMPLSim      # pulls mujoco, lxml, vtk, numpy-stl, torchgeometry, smplx fork
   mkdir -p /workspace/SMPLSim/data
   ln -sfn /workspace/humos/body_models/smpl /workspace/SMPLSim/data/smpl   # SMPL_{NEUTRAL,MALE,FEMALE}.pkl
   ```
   `SMPL_Robot` reads `data/smpl` **relative to the working directory**, so always run generation with the
   current directory set to `/workspace/SMPLSim`. SMPLSim pins the `ZhengyiLuo/smplx` fork; after installing,
   re-run the HUMOS smoke test from `note/README.humos-runpod-setup.md` section 7 to confirm `infer.py` still
   works. If installing SMPLSim breaks HUMOS (or the clone fails), stop and report the error; do not downgrade
   or pin packages on your own. The chumpy fork from the HUMOS setup note is needed by both.
9. Git LFS: in this repo `*.pt` files and possibly some XMLs are LFS pointers. A `.pt` of about 130 bytes is a
   pointer, not data. This job only needs `assets.yaml` files (plain text), so you do not need `git lfs pull`.

## 4. Step 1 - Clip list

1. Pilot clip ids: `rclone lsf r2:proto-data/humos_unseen_morphology/ | sed 's/\.pt$//'` (1024 ids such as
   `000007`, `M000123`).
2. Split membership: `data/splits/hhi_stage2_v1/{train,validation,test}_ids.txt`. These are gitignored, so they
   may be absent on the pod. If absent, try `tools/create_hhi_split_manifests.py` (read its docstring first) or
   look under `r2:proto-data/` with `rclone lsf` for a splits file. If you cannot recover the split labels, stop
   and ask the user. Do not invent labels.
3. Sanity: the 1024 ids must split 932 train / 47 validation / 45 test with 0 unmatched. If not, stop.
4. Subsample with `numpy.random.default_rng(31337)`: **233 train, 12 validation, 11 test**, sampled without
   replacement within each split. Sort the final ids. Write `data_cache/unseen_v2/clips_256.json` (one record
   per clip: id and split), and write the keyids-file for `infer.py` in the exact format of
   `pilot_1024_motions.json` as `data_cache/unseen_v2/clips_256_keyids.json`.

## 5. Step 2 - Sample the betas

Write a new script `tools/sample_unseen_betas.py` (full Apache-2.0 header, see `CLAUDE.md`) that implements
section 1 exactly and writes:

- `protomotions/data/assets/all_betas_unseen_inside.pt` and `all_betas_unseen_outside.pt`: a dict
  `{beta_key: float32 tensor of shape (10,)}` with exactly 64 entries each. **Mirror the format of an existing
  `all_betas_interp.pt`** (the copy at `/workspace/humos/humos/all_betas_interp.pt` is the one `infer.py` was
  smoke-tested with). Inspect its key type, dtype, and shape and match them.
- `data_cache/unseen_v2/beta_manifest.csv`: one row per beta with `dataset, tier, beta_key, linf, l2,
  nearest_trained_l2` (L2 distance to the closest of the 64 training betas), plus the 10 coordinates.

Acceptance asserts (the script must fail loudly):

- inside: 64 betas, `max |beta| <= 3.0`, coordinate RMS within 1.73 +/- 0.15 (uniform on [-3,3] has 1.732).
- outside: 4 tiers x 16; for tier k, `3 + 0.5k < linf <= 3.5 + 0.5k`; all `linf <= 5.0`; every beta has at
  least one coordinate beyond 3.
- keys unique, and disjoint from all keys in `smpl_mor/assets.yaml` and `smpl_mor_interp/assets.yaml`.
- no new beta within L2 < 1e-3 of any training beta.

Print a short summary (per dataset and tier: count, linf range, L2 range, nearest_trained_l2 range). For
reference, the old interp set had nearest-trained distances of 2.58 to 5.05.

## 6. Step 3 - Generate the MJCF assets (SMPLSim)

SMPLSim's `run.py` produced the training XMLs and `run_heldout.py` produced the old interp/extrap XMLs. Their
`robot_cfg` dicts are identical (checked when this file was written), and `run_heldout.py`'s `sample_betas` and
`generate_xml` are the pieces to reuse.

**Do not run `run_heldout.py` or `run.py` as they are.** Both hard-code output locations (`../ProtoMotions/...`
and `../humos/humos/...`), and `run_heldout.py --mode interp` writes into `smpl_mor_interp/`, which would
overwrite the committed 09-13 assets. Write a **new** driver file (do not edit SMPLSim's existing files), e.g.
`/workspace/SMPLSim/run_unseen_v2.py`, that imports `deterministic_hex4`/`sample_betas` and the
`SMPL_Robot` construction from `run_heldout.py`, takes explicit `--betas-file` and `--output-folder` arguments,
and writes `{gender}_{beta_key}_smpl.xml`. Copy the `robot_cfg` dict verbatim from `run_heldout.py` (do not
retype it). If the sampling code in step 2 lives in `tools/sample_unseen_betas.py`, the driver only needs to
load the betas files and generate XMLs.

**Gate B (reproduction check) before generating anything new:** regenerate one existing training shape from its
betas (use `female_093098f0`; its betas are in `smpl_mor/assets.yaml`) into a scratch folder and compare with the
committed `protomotions/data/assets/mjcf/smpl_mor/female_093098f0_smpl.xml`. They must match to a **relative**
1e-6 on every numeric attribute, with identical structure. Compare relatively, not absolutely: toe `geom`
densities are about 8e6, where float32 rounding alone gives absolute differences of several units. If they do
not match, the driver is not reproducing the training pipeline: stop and report which attributes differ. Do the
same for one male shape.

**Gate B result (2026-10-09, pod, HUMOS's `body_models/smpl` files symlinked as `SMPLSim/data/smpl`): passed.**
For `female_093098f0` and `male_093098f0`, no structural differences; the only numeric differences were in
`geom.density` (6 of 24 geoms), at most about 5e-7 relative (toe densities of about 8e6 differing by 0.7 and
3.8). All positions, sizes and joint values matched to better than 1e-6. So HUMOS's SMPL files reproduce the
training XMLs, and the original `smpl_model.zip` (`gdrive:ckpt/smpl_model.zip`) is not needed.

Then generate, per dataset, into the new asset folders (128 XMLs each), and build the manifests:

```bash
cd /workspace/ProtoMotions
python tools/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor_unseen_inside \
    --betas-file protomotions/data/assets/all_betas_unseen_inside.pt \
    --out protomotions/data/assets/mjcf/smpl_mor_unseen_inside/assets.yaml
python tools/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor_unseen_outside \
    --betas-file protomotions/data/assets/all_betas_unseen_outside.pt \
    --out protomotions/data/assets/mjcf/smpl_mor_unseen_outside/assets.yaml
```

Expect 128 entries in each `assets.yaml`, and the `betas` there must equal the betas-file values.

**Gate C (asset sanity):** use the raw (un-z-scored) feature extraction in
`tools/extract_smpl_physics_features.py` to get `total_mass` and `total_height` for every new XML, and the same
for the 128 training XMLs as the reference range. Write `data_cache/unseen_v2/physical_report.csv`
(`asset_id, dataset, tier, gender, beta_key, total_mass_kg, total_height_m`, plus per beta key the male/female
mass ratio). **Do not write a `physics_features.pt` into the new asset folders**: that tool z-scores against
whatever folder it is given, and a wrong normaliser would silently corrupt a physics-feature policy.

- Hard stop: any XML that fails to parse, has NaN, or has non-positive mass or height.
- Report only (flag, do not drop): shapes whose mass or height lies outside the training range; per tier, the
  mass/height ranges; any beta whose male/female mass ratio lies outside the **training range of the same
  ratio, 0.56 to 3.21** (measured on the 128 training bodies; half of them exceed 1.5, so 1.5 is not a useful
  threshold). Previously the failing shapes on 09-13 showed large male/female mass asymmetry, so this table
  will be needed for interpreting results. Result on 2026-10-09: inside 0 of 64 betas outside that range,
  outside about 13 of 64 (ratios 0.32 to 6.52).

## 7. Smoke test (Gate D)

Run the whole chain (sections 8.1 to 8.5) on **2 clips** (one train, one validation) for the `inside` dataset
into a scratch directory, ending in a refined file. Expect 2 x 128 = 256 motions. Check frame counts are sane
(the 09-13 episodes were about 198 steps at 30 fps), no NaN, every `motion_asset_ids` entry exists in
`smpl_mor_unseen_inside/assets.yaml`, and the refine step prints its before/after report. Fix problems here, on
2 clips, not on 256. Delete the scratch directory afterwards.

## 8. Steps 4 to 8 - Full run (do `inside` completely, then `outside`)

Let `D` be `inside` or `outside`; work under `/workspace/unseen_v2/D/`.

### 8.1 HUMOS inference

```bash
cd /workspace/humos
python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file /workspace/ProtoMotions/protomotions/data/assets/all_betas_unseen_D.pt \
    --keyids-file /workspace/ProtoMotions/data_cache/unseen_v2/clips_256_keyids.json \
    --local-out-dir /workspace/unseen_v2/D/humos_raw
```

`infer.py` produces every beta in the file for both genders per clip, so expect 64 x 2 = 128 variants per clip.
Use `--cfg` (not `--config`). Expect the log line `Restricted to 256/256 requested keyids`. The run is
resumable (existing output files are skipped).

Verify: exactly 256 `.pt` files; each loads and has `male` and `female` dicts with the 64 beta keys of that
dataset's betas file; no NaN in `pose_body`, `trans`, `root_orient`. **Upload the raw output now** as insurance:
`rclone copy /workspace/unseen_v2/D/humos_raw r2:proto-data/unseen_shapes_v2/humos_raw/D/ --transfers=4
--s3-no-check-bucket --progress`, then verify object count and size match.

### 8.2 Export to AMASS-style NPZ

Mirror what `tools/prepare_stage2_data.py` did for the training corpus. Note it did **not** pass
`--apply-offset-height` (the older held-out README did; the training corpus did not, and the frame-0 offset step
handles grounding), so do not pass it:

```bash
cd /workspace/ProtoMotions
python tools/export_humos_to_amass_npz.py \
    --input-dir /workspace/unseen_v2/D/humos_raw \
    --out-root /workspace/unseen_v2/D/npz \
    --yaml-name unseen_v2_D.yaml \
    --fps 30.0 \
    --skip-existing
```

Expect 256 x 128 = 32,768 NPZ files and a YAML listing them. If `prepare_stage2_data.py` at this commit uses
different flags from those shown, the script wins: re-read lines around its "Step 2/5" and copy them.

### 8.3 Convert to MotionLib shards

```bash
python tools/convert_amass_to_motionlib_with_morphology.py \
    /workspace/unseen_v2/D/npz /workspace/unseen_v2/D/proto \
    --motion-config /workspace/unseen_v2/D/npz/unseen_v2_D.yaml \
    --humanoid-type smpl --output-fps 30 --device cpu --batch-size 8192
```

(The older held-out README shows `--motion-yaml`/`--assets-yaml`, a stale interface. The flags above are from the
current script and from `prepare_stage2_data.py`.) Expect 4 shards. Each shard should hold 8192 motions, 64
distinct clip ids, and 128 motions per clip. If the packaging OOMs, lower `--batch-size` but keep it a multiple of
128. Delete the NPZ directory once the shards verify.

Check the shards carry the asset metadata the offset and refine steps need: `motion_asset_ids`, `motion_beta_keys`,
`motion_genders`, `motion_clip_ids`, `motion_betas`.

### 8.4 Frame-0 grounding offset (IsaacGym)

Run on each shard, with the interpreter chosen in preflight item 7:

```bash
python tools/compute_humos_frame0_offsets.py \
    --motion-file /workspace/unseen_v2/D/proto/<shard>.pt \
    --asset-root protomotions/data/assets/mjcf/smpl_mor_unseen_D \
    --out-motion-file /workspace/unseen_v2/D/offset/<shard>_offset.pt \
    --overwrite
```

If IsaacGym cannot load some of the new bodies (the outside tiers may include very tall, short, heavy or light
bodies), record which `asset_id`s fail and report; do not drop them silently. Upload the offset-only shards to
`r2:proto-data/unseen_shapes_v2/D_offset/` (they are the same processing as the 09-13 run, so they are also the
control that lets the user measure what refinement changed).

**Why this step stays even though `refine_humos_motion.py`'s docstring says it replaces it.** Reading the code
(`step5_ground_penetration_correction`), refinement only lifts a motion when its lowest point is below
`TARGET_Z = 0.005`; it never lowers a motion that starts floating. The frame-0 offset shifts in both directions.
The training corpus went offset first and refinement second (`note/README.paper-structure-and-keypoints.md`:
"the existing corpus had frame-0 grounding; the newer refinement also addresses later frames"), so offset then
refine is the matching order. If you find evidence that this reading is wrong, say so and stop rather than
choosing silently.

### 8.5 Refine (CPU)

```bash
python tools/refine_humos_motion.py \
    --motion-file /workspace/unseen_v2/D/offset/<shard>_offset.pt \
    --asset-root protomotions/data/assets/mjcf/smpl_mor_unseen_D \
    --out-motion-file /workspace/unseen_v2/D/refined/<shard>_refined.pt \
    --device cpu --report 2>&1 | tee /workspace/unseen_v2/D/refined/<shard>_report.log
```

`tools/refine_humos_r2_per_clip.py` shows how the training corpus invoked it (around lines 250 to 262); keep the
same flags and only change the asset root. Keep every `--report` log.

For the outside dataset the reference itself may be bad (HUMOS was never trained on bodies this extreme), and a
tracking failure later could be a bad reference rather than a controller limit. So, **if it is cheap** (about 30
minutes of work), reuse the metric functions in `refine_humos_motion.py`'s report code to write
`data_cache/unseen_v2/reference_quality_by_asset.csv`: per `asset_id`, mean stance-foot speed and fraction of
frames below the floor, computed on the offset-only shards (before refinement) and on the refined shards. If the
functions are not reusable without a rewrite, skip this and just say so; the saved `--report` logs are the
fallback.

## 9. Verification and upload (Gate E)

For each dataset, on the 4 refined shards:

- 8192 motions per shard, 32,768 total; 256 distinct clip ids matching `clips_256.json`; every clip has exactly
  128 motions; every `motion_asset_ids` value is in that dataset's `assets.yaml`; 64 distinct beta keys, each with
  both genders; no non-finite values in `gts, grs, gvs, gavs, dps, dvs, lrs`.
- Clip split counts by motion: 233 / 12 / 11 clips, i.e. 29,824 / 1,536 / 1,408 motions.
- Write `data_cache/unseen_v2/dataset_manifest.json`: shard filenames and sizes, motion counts, the seeds, the
  clip list, tier membership of every asset id, and the git commit hash of the repo checkout the build ran on.

Upload with `rclone copy` only (layout; every destination is under `r2:proto-data/unseen_shapes_v2/`):

| Path | Content |
|---|---|
| `humos_raw/inside/`, `humos_raw/outside/` | 256 raw HUMOS `.pt` each (uploaded in 8.1) |
| `inside_offset/`, `outside_offset/` | frame-0-only shards (uploaded in 8.4) |
| `inside_refined/`, `outside_refined/` | final shards plus the `--report` logs; **these are the evaluation inputs** |
| `assets/` | a tarball of `smpl_mor_unseen_inside/`, `smpl_mor_unseen_outside/`, both betas files, `beta_manifest.csv`, `physical_report.csv`, `clips_256.json`, `dataset_manifest.json` |

Verify each upload (object count and total size equal to the pod). The XMLs, `assets.yaml` files and betas files
also need to end up in the repo (git-lfs) for evaluation; the user will pull them from `assets/` and do the
git operations themselves.

## 10. Report back to the user

One message with:

1. A table of every gate (A to E) with pass/fail and the evidence.
2. The beta summary (per dataset and tier) and the physical report summary: which shapes are outside the
   training mass/height range, and which beta keys have a male/female mass ratio outside 0.56 to 3.21.
3. Any shape or clip that failed anywhere (IsaacGym load, NaN, refine error), by id.
4. Whether the offset step ran, in which interpreter, and whether the reference-quality CSV was produced.
5. Exact R2 paths and sizes, and anything you deviated from in this file and why.

Do not interpret results or write paper text. Do not delete anything on R2.

## Note for the evaluation step (not part of this job)

`tools/analyze_unseen_morphology_generalization.py` defaults to the old `smpl_mor_interp` folder, so evaluation
needs its asset-folder and asset-info arguments pointed at the new folders. The original protocol should stay
unchanged (`--max-episode-steps`, `--success-gt-error-threshold`, `--envs-per-asset`). With 128 shapes per
dataset, 128 envs per asset is 16,384 envs; plan the number of rounds or shards per run accordingly. Merge the
four refined shards per dataset with `tools/merge_motion_shards.py --num-shards 1`, as the 09-13 run did. The
validation and test clips are unseen only for the frozen paper checkpoint; the `fulldata` checkpoint trained on
them, so for that checkpoint report train-split clips only.
