# Paired unseen-body evaluation of the full-data checkpoint

Plan prepared 2026-10-06. Evaluation has not been executed.

## Question and checkpoint

Does additional training on the original 128 bodies improve zero-shot tracking
on the same 32 unseen bodies, including the four previously difficult bodies?
Use `epoch_34000.ckpt`, the latest numbered checkpoint currently present locally,
chosen before evaluating unseen bodies. Do not select among checkpoints using
these evaluation results. Keep `resolved_configs_inference.pt` beside it.
The remembered approximately 99% training score is not yet independently verified.

Full-data training includes all original train/validation/test motion identities.
All 32 evaluation bodies must remain excluded from training. This experiment is
unseen-body generalization on known motion identities, not held-out-motion
generalization or adaptation through training on new bodies.

## Inventory checked locally

- Present: 32 actual MJCF XML assets and `smpl_mor_interp/assets.yaml`.
- Present: baseline raw and joined CSVs under `data_cache/`.
- Present: candidate epoch 34000 and its inference configuration under
  `results/hhi_wide_stage2_discover_attention_slot_type_fulldata/`.
- Missing locally: `data_cache/unseen_morphology_merged_0.pt` (~14.5 GB).
- Source recorded in the original run: 128 R2 shards in
  `r2:proto-data/humos_unseen_morphology_offset/`. Remote availability is unverified:
  the Windows rclone executable failed to start with an Illegal System DLL Relocation error.
- Original split ID text files are missing locally. The baseline joined CSV
  already retains original split membership; no split regeneration is needed
  for the paired comparison.

The original unseen references have frame-zero grounding, not the full training
reference refinement. Reuse them unchanged to isolate the checkpoint comparison.
Do not rerun HUMOS or alter body assets before this experiment.

## Run on the Linux IsaacGym GPU environment

Run from the repository root. Transfer the candidate checkpoint AND its
`resolved_configs_inference.pt` to the same relative folder on the pod first.
Ensure the baseline joined CSV and the 32 assets are also present.
Allow disk space for both downloaded shards and the merged file, plus ample
host RAM for merging/loading. The full replay uses 4,096 simulated environments.

Check R2 and recover the original references if the merged file is absent:

```bash
rclone lsf r2:proto-data/humos_unseen_morphology_offset/ --max-depth 1
rclone copy r2:proto-data/humos_unseen_morphology_offset/ /workspace/unseen_morph_shards/
# Confirm exactly 128 humos_32768_*_offset.pt files before merging.
python tools/merge_motion_shards.py \
  --src /workspace/unseen_morph_shards --dst data_cache --num-shards 1 \
  --pattern 'humos_32768_*_offset.pt' --out-prefix unseen_morphology_merged
```

Use a fresh result folder; retain file hashes and the source revision:

```bash
mkdir -p results/unseen_fulldata_epoch34000
sha256sum results/hhi_wide_stage2_discover_attention_slot_type_fulldata/epoch_34000.ckpt \
  results/hhi_wide_stage2_discover_attention_slot_type_fulldata/resolved_configs_inference.pt \
  data_cache/unseen_morphology_merged_0.pt \
  data_cache/unseen_morphology_generalization.joined.csv \
  > results/unseen_fulldata_epoch34000/input_hashes.txt
git rev-parse HEAD > results/unseen_fulldata_epoch34000/source_revision.txt
python tools/analyze_unseen_morphology_generalization.py \
  --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_fulldata/epoch_34000.ckpt \
  --motion-file data_cache/unseen_morphology_merged_0.pt \
  --max-clips 3 --envs-per-asset 3 --max-episode-steps 50 \
  --output results/unseen_fulldata_epoch34000/smoke.csv
```

Check that the smoke test completes with 96 finite pair records; its shortened
horizon is not a performance result. Then run the original full protocol:

```bash
python tools/analyze_unseen_morphology_generalization.py \
  --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_fulldata/epoch_34000.ckpt \
  --motion-file data_cache/unseen_morphology_merged_0.pt \
  --envs-per-asset 128 --max-episode-steps 210 --success-gt-error-threshold 0.5 \
  --output results/unseen_fulldata_epoch34000/replay.csv
python tools/compare_unseen_morphology_checkpoints.py \
  --baseline data_cache/unseen_morphology_generalization.joined.csv \
  --candidate results/unseen_fulldata_epoch34000/replay.csv \
  --output results/unseen_fulldata_epoch34000/paired_summary.csv
```

The comparator rejects missing/extra/duplicate pairs, non-finite errors, and
different episode horizons. It reports all-body and per-body results, paired
recoveries/regressions, and historical motion-split groups. The latter are NOT
held-out groups for the candidate. Review the four severe shapes and the mild
`male_95749fe2` outlier without excluding any from the overall result.

If the simulator/runtime or evaluator has changed since the original run,
replay the original epoch-28170 checkpoint on the rebuilt corpus as well and
use that contemporaneous baseline. Record discrepancies rather than assuming
bitwise reproducibility across GPU environments.

## Following the primary replay

Prepare a trained-body control using these same 1,024 clip identities and
their existing trained-body reference variants. This control is not yet packaged
locally; freeze the selected trained bodies before running it. Compare original
and candidate checkpoints on identical pairs. This separates broad motion
tracking improvement from changes specific to unseen bodies.

The existing unseen evaluator logs position/rotation errors and termination,
not jerk. Do not claim this comparison measures smoothness without additional
instrumentation. Inspect representative recovered and persistent failures.
