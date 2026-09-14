# HUMOS Inference on RunPod (Ubuntu 22.04 template)

Goal: run HUMOS inference (not training — a pretrained checkpoint already exists locally) on a
RunPod pod, since the local machine's `nvidia-smi` isn't working. Used to generate the
unseen-morphology dataset (pilot clips x held-out interp betas) for paper §6.5.1.

Local HUMOS repo: `/home/hlz/repos/humos`. Checkpoint used:
`logs/humos/q6zbv2tu/checkpoints/latest-epoch=1599.ckpt`.

Pod identity during this setup: `root@c747541afd7e:/workspace/humos` (root in a container).

## 1. Why a minimal package, not the full 76GB repo

```bash
cd /home/hlz/repos/humos
du -sh --max-depth=1 .
```

Breakdown: `datasets/` 73GB (raw training data, not needed for inference), `body_models/` 2.2GB
(SMPL/SMPLH, needed), `logs/` 633MB (only need the one checkpoint subdir, 316MB), `humos/` 38MB
(the package), `.git/` 97MB, `wandb/` 67MB, `aitviewer_humos/` 65MB, `website/` 27MB,
`bosRegressor/` 20MB, `assets/` 1.7MB.

Confirmed `SMPL_PATH = "./body_models/smpl"` in `humos/utils/constants.py` is a **relative** path,
so a minimal copy is safe as long as the directory layout is preserved and scripts run from the
repo root.

## 2. Build and transfer the minimal inference package

Locally, package only what's needed for inference (~2.6GB):

```bash
cd /home/hlz/repos/humos
tar -czf humos_infer.tar.gz \
    humos/ body_models/ assets/ bosRegressor/ \
    logs/humos/q6zbv2tu/ setup.py requirements.txt
```

Explicitly excluded: `datasets/` (73GB raw data), `.git/`, `wandb/`, `aitviewer_humos/` (initially
assumed visualization-only — this assumption turned out to be wrong, see step 4), `website/`, the
rest of `logs/`.

Upload `humos_infer.tar.gz` to the pod under `/workspace/humos/` (e.g. via the RunPod file browser,
`scp`, or `runpodctl send`/`receive` — adjust to your pod's connection method).

## 3. Extract on the pod

rclone copy r2:proto-data/humos/humos_infer.tar.gz ./ \
--transfers=2 --multi-thread-streams=16 --multi-thread-chunk-size=128M \
--s3-no-check-bucket --progress

```bash
tar -xzf humos_infer.tar.gz --no-same-owner
```

## 4. Python environment

HUMOS needs **Python 3.10** (`conda create -n humos_p310 python=3.10` per HUMOS's own
`README.md:27`). The Ubuntu 22.04 RunPod template ships Python 3.10 as default `python3`/`python`,
so no venv/conda needed inside the container.

Pod GPU: NVIDIA A40, driver 570.195.03, CUDA 12.8 (`nvidia-smi`). Driver is backward-compatible with
older CUDA runtimes, so cu121 wheels are fine.

Install torch + requirements + the package itself. **Always use `python -m pip`, not bare `pip`** —
this container has multiple Python installs (`pip` alone resolved to Python 3.12's pip while
`python` is 3.10, silently installing packages into the wrong interpreter's site-packages):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 5. Missing runtime dependencies not covered by requirements.txt

`requirements.txt` doesn't fully capture `infer.py`'s actual import graph. These surfaced one at a
time as `ModuleNotFoundError`s when running the smoke test (see step 7); resolved here in one batch
for future setups.

**`aitviewer`** (`infer.py:28`, `from aitviewer.models.smpl import SMPLLayer`) — locally this is
installed editable from the `aitviewer_humos/` submodule
(`https://github.com/sha2nkt/aitviewer_humos.git`, a fork of ETH's aitviewer). It was excluded from
the transfer package assuming it was visualization-only; it isn't — `infer.py` imports it at module
level for the SMPL layer wrapper. No need to transfer the 65MB directory; install the fork directly
from GitHub instead:

```bash
python -m pip install "git+https://github.com/sha2nkt/aitviewer_humos.git"
```

(Pulls in GUI/render deps — `moderngl`, `moderngl-window`, `PyQt5`, `usd-core`, etc. — all
pip-installable headless; only becomes an issue if a minimal container lacks system libs `PyQt5`'s
wheel needs, not encountered here.)

**`wandb`, `scikit-learn`, `transformers`, `ipdb`, `pyyaml`** — also missing, installed together:

```bash
python -m pip install wandb scikit-learn transformers ipdb pyyaml
```

(`tqdm`, `roma`, `opencv-contrib-python-headless` came in transitively via the aitviewer/pytorch-lightning
install chains, so weren't needed explicitly.)

**`chumpy`** (pulled in by `smplx` when loading the pickled SMPL model file, inside `SMPLLayer.__init__`) —
`requirements.txt` has it commented out, and stock chumpy (PyPI 0.70/0.71) is broken against
numpy>=1.20 (relies on removed `np.bool`/`np.int` aliases). Locally it's installed from a patched
personal fork (`hansen1416/chumpy`), not PyPI. Install the same fork:

```bash
python -m pip install "git+https://github.com/hansen1416/chumpy.git"
```

`human_body_prior` (also commented out in `requirements.txt`) was checked and is **not** in
`infer.py`'s import chain — likely training/data-prep-only, not expected to be needed here.

## 6. Data dependencies under the excluded `datasets/` (73GB)

`TextMotionDataset.__init__` needs two things from `datasets/` that aren't part of `humos/`'s own
`annotations/`/`stats/` subdirs (those 37MB/284KB dirs are already inside the transferred `humos/`
package and sufficient for split lists + normalizer stats):

- **`datasets/splits/`** (66MB) — `mld_test_split_keyids.txt` (hardcoded path in
  `humos/src/data/text_motion.py`) and `identity_dict_test_split_smpl.pkl` (used by the model in
  `humos/src/model/tmr_cyclic_origin.py:227`).
- **`datasets/humos3dfeats/`** (2.7GB) — per-clip AMASS motion feature `.tensor` files, read lazily
  by `AMASSMotionLoader` as the dataloader iterates (not touched at dataset-construction time since
  `cfg_template.yml` sets `TM_LOADER.PRELOAD: False`).

Package and transfer just these two directories (~2.77GB) the same way as step 2:

```bash
# locally
cd /home/hlz/repos/humos
tar -czf humos_datasets_subset.tar.gz datasets/splits datasets/humos3dfeats
```
rclone copy humos_datasets_subset.tar.gz r2:proto-data/humos/ \
--transfers=1 --multi-thread-streams=16 --multi-thread-chunk-size=128M \
--s3-no-check-bucket --progress

rclone copy r2:proto-data/humos/humos_datasets_subset.tar.gz ./ \
--transfers=1 --multi-thread-streams=16 --multi-thread-chunk-size=128M \
--s3-no-check-bucket --progress

```bash
# on the pod, from /workspace/humos
tar -xzf humos_datasets_subset.tar.gz --no-same-owner
```

python -m pip install --no-build-isolation "git+https://github.com/hansen1416/chumpy.git"

## 7. Smoke test

```bash
python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file humos/all_betas_interp.pt \
    --local-out-dir /tmp/humos_smoketest
```

Error progression seen while getting here (in order): `ModuleNotFoundError: torch` (python/pip
mismatch, fixed in step 4) -> `ModuleNotFoundError: aitviewer` (step 5) ->
`ModuleNotFoundError: wandb` (step 5) -> `FileNotFoundError: ./datasets/splits/mld_test_split_keyids.txt`
(step 6, in progress at time of writing).

## 8. `infer.py` keyid-subset patch and the real run

`infer.py` had no way to restrict which clips it processes — it always iterates the full
concatenated train/val/test split (20k+ clips), and for every clip generates **both genders x every
beta in `--betas-file`** (nested loop in `run_inference`). So "N clips x B betas" actually means
"N x B x 2" variants, and there was no way to target just the 1024-clip pilot set
(`pilot_1024_motions.json`, already used for `hhi_1024_motion`/`hhi_1024_transfer`/etc. elsewhere in
the project) without a code change.

Patched `humos/infer.py` (local repo, then re-transferred to the pod) to add a `--keyids-file` flag:
`run_inference` now takes a `keyids_filter` set and, if given, filters each split's
`dataset.keyids` down to the intersection before building the `ConcatDataset`. Verified the keyid
formats match exactly (`pilot_1024_motions.json` and the HUMOS split files both use zero-padded
6-digit IDs, `M`-prefixed for mirrored clips).

Transferred just the patched file + the pilot JSON (both tiny) via R2, same pattern as the larger
transfers:

```bash
# locally
cd /home/hlz/repos/humos
cp /home/hlz/repos/ProtoMotions/pilot_1024_motions.json .
tar -czf humos_infer_patch.tar.gz humos/infer.py pilot_1024_motions.json
rclone copy humos_infer_patch.tar.gz r2:proto-data/humos/ --s3-no-check-bucket --progress
```

```bash
# on the pod, from /workspace/humos
rclone copy r2:proto-data/humos/humos_infer_patch.tar.gz . --s3-no-check-bucket --progress
tar -xzf humos_infer_patch.tar.gz --no-same-owner
```

Real run (1024 clips x 16 interp betas x 2 genders = 32,768 variants):

```bash
python -u humos/infer.py \
    --cfg humos/configs/cfg_template.yml \
    --betas-file humos/all_betas_interp.pt \
    --keyids-file pilot_1024_motions.json \
    --local-out-dir /workspace/humos_unseen_morphology_output
```

Completed cleanly: `Restricted to 1024/1024 requested keyids` at startup (no pilot clips lost to
the HumanML3D support-object filter), output = 1024 `.pt` files (one per clip keyid, each containing
all 16 betas x 2 genders), 8.9GB total.

## 9. Output uploaded to R2, pod released

```bash
rclone copy /workspace/humos_unseen_morphology_output \
    r2:proto-data/humos_unseen_morphology/ \
    --transfers=4 --multi-thread-streams=16 --multi-thread-chunk-size=128M \
    --s3-no-check-bucket --progress
```

Verified post-upload: 1024 objects, 8.868 GiB on R2 (matches pod exactly); spot-checked
`000007.pt` loads cleanly with `torch.load` and has the expected structure
(`{"male": {beta_key: {betas, gender, root_orient, pose_body, trans, offset_height, joints_pos,
root_vel, root_ang_vel, dof_vel}, ...16 betas}, "female": {...}, "text": [...]}`). Pod released
after this check — R2 at `r2:proto-data/humos_unseen_morphology/` is now the source of truth for
this data.

## Open items / not yet done

- Apply the known `infer.py` train/val/test split-concatenation bug fix (documented in
  `note/README.note.md` §11, from the earlier E7 held-out pipeline work) — check whether it's
  relevant now that keyid filtering happens post-concatenation (it may already be moot, since we
  filter by keyid rather than relying on split membership).
- Downstream pipeline: AMASS NPZ export, MotionLib conversion, frame-0 grounding offset (pattern in
  `note/README.heldout-pipeline.md` steps 4-5b, adapted for this 1024x16x2 subset) — not started.
- Update paper §6.5.1 wording from the placeholder "~20,000 clips" plan to the actual 1024x16x2
  scoped design, once the downstream pipeline produces real numbers.
