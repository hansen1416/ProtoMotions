# Unseen shapes v2: HUMOS and SMPLSim part (done 2026-10-09)

What was built and how. The remaining steps (export, convert, offset, refine, verify) are in
`note/README.unseen-shapes-v2-dataset-build.md` section 8.2 onward and run on a ProtoMotions + IsaacGym pod.

## Result

Two datasets, each 64 betas x 2 genders = 128 shapes, on the same 256 clips (32,768 raw HUMOS motions each).
Everything is on R2 under `r2:proto-data/unseen_shapes_v2/`:

| Path | Content |
|---|---|
| `humos_raw/inside/`, `humos_raw/outside/` | 256 HUMOS `.pt` each, 8.9 GB each, all 200 frames at 20 fps, no NaN |
| `inputs/` | `all_betas_unseen_{inside,outside}.pt`, `clips_256_keyids.json`, `beta_manifest.csv` |
| `xml/smpl_mor_unseen_{inside,outside}/` | 128 MJCF XMLs + `assets.yaml` each; `xml/run_unseen_v2.py` is the generator |
| `physical_report.csv`, `infer.log` | per-asset mass/height; inference log |

## Design (locked)

- **inside:** uniform [-3, 3]^10, seed 1001. **outside:** four tiers of 16 betas, max |beta| in (3,3.5], (3.5,4],
  (4,4.5], (4.5,5], seed 2001, built as `u * r / max|u|`. Keys use SMPLSim's `deterministic_hex4`.
- **clips:** 233 train + 12 validation + 11 test (seed 31337), a subset of the 1,024 pilot clips (932/47/45).
  Validation/test clips are unseen only for the frozen paper checkpoint; `fulldata` trained on them.
- The input-generation script was run inline and not saved; the spec above is the record. Betas range:
  inside L2 3.28-6.79, outside 5.10-11.20 (training 3.9-7.0).

## HUMOS environment (verified recipe, RunPod)

1. Pod default Python was 3.12 but HUMOS needs **3.10** (`aitviewer` requires <3.11):
   `apt-get install -y python3.10 python3.10-venv && python3.10 -m venv /workspace/venv310`.
2. Download and extract **inside `/workspace/humos`** (extracting in `/` puts `requirements.txt` outside the package):
   `humos_infer.tar.gz` (code, `body_models`, checkpoint `q6zbv2tu`), `humos_datasets_subset.tar.gz`
   (`datasets/splits`, `humos3dfeats`), all from `r2:proto-data/humos/`.
3. `python -m pip install`: torch (cu121), `-r requirements.txt`, `-e .`, `git+https://github.com/sha2nkt/aitviewer_humos.git`,
   `wandb scikit-learn ipdb pyyaml`, **`transformers==4.46.3`** (5.x fails on `AutoModel` with this stack),
   chumpy fork with `--no-build-isolation`. Exact versions: `r2:proto-data/humos/pod_pip_freeze.txt`.
4. **Normalizer stats are gitignored and were lost.** Regenerated with
   `python humos/prepare/motion_stats.py --device cpu` (19,362 train clips, 30 min) into
   `humos/stats/humanml3d/humos3dfeats/`; backed up as `r2:proto-data/humos/humos_stats.tar.gz`.
5. Checks: one-clip smoke test passed (8 s/clip). Reproduction check on clip `000007` with the 09-13 betas:
   run vs run difference 0.0 (deterministic), difference to the 09-13 output 2.5e-3 max, so the regenerated
   stats are close to the originals but not proven bit-identical.

## Inference

`humos/infer.py --cfg humos/configs/cfg_template.yml --betas-file ... --keyids-file clips_256_keyids.json
--local-out-dir ...` once per dataset: about 4.5 s per clip, 18 min per dataset, `Restricted to 256/256`.

## SMPLSim XMLs

- Separate venv `/workspace/venv_smplsim` (py3.10, CPU torch): clone `hansen1416/SMPLSim` branch `feature/hhi`,
  `pip install -e`, chumpy fork, symlink `SMPLSim/data/smpl` to `humos/body_models/smpl`. Run from the SMPLSim directory.
- `run_unseen_v2.py` imports `run_heldout.generate_xml` (same robot config as the training XMLs) and writes only
  to `--out-dir`. **Never run `run_heldout.py` itself**: its hard-coded paths overwrite `smpl_mor_interp`.
- **Gate B (reproduction):** `female_093098f0` and `male_093098f0` regenerated vs the committed XMLs: only
  `geom.density` differs, at most ~5e-7 relative (toe densities are ~8e6). Compare relatively, not absolutely.
  HUMOS's SMPL files therefore reproduce the training XMLs.
- 256 XMLs in 16 min 47 s; `assets.yaml` from `tools/generate_smpl_mor_asset_info.py`.

## Physical report (mass / height; training 26.4-144.4 kg, 1.13-1.67 m)

| Set | Mass kg | Height m | Beyond training range |
|---|---|---|---|
| inside | 27.7-154.5 | 1.10-1.67 | 1 mass, 5 height |
| outside tier 0 | 51.8-171.4 | 1.11-1.63 | 2 / 1 |
| outside tier 1 | 26.5-133.8 | 1.14-1.71 | 0 / 1 |
| outside tier 2 | 31.2-181.8 | 1.11-1.66 | 4 / 1 |
| outside tier 3 | 21.9-211.1 | 1.07-1.82 | 4 / 4 |

- Outside in beta space is only mildly outside physically (10/128 beyond the mass range): analyse against
  mass and height as well as beta distance.
- Male/female mass ratio at the same beta: training 0.56-3.21, inside 0.59-3.14, outside 0.32-6.52. About 13 of
  64 outside betas fall outside the training ratio range (e.g. `26b2c805` 6.5x, `4ee058c0` 4.0x); inspect these
  first if an outside shape fails oddly. Ratios above 1.5 are normal (half of training).

## Next

On a ProtoMotions + IsaacGym pod: pull `humos_raw`, `inputs` and `xml` from R2, then export to NPZ, convert to
MotionLib (4 shards of 8,192 motions per dataset), frame-0 offset (IsaacGym), refine, verify, upload.
