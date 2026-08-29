cd /workspace && git clone -b feature/hhi https://github.com/hansen1416/ProtoMotions.git && cd ProtoMotions

pip install gdown && apt update && apt install curl zip -y && curl https://rclone.org/install.sh | bash

<!-- 1024-raw.zip -->
gdown 14IYbHhMxKARQ9nnEitJXyHwGGP0SFKHG

<!-- 1024-phy.zip -->
gdown 1FNqL69Xu46bW3wzQNt6ndMwiStCR6d23

<!-- download data from R2 -->
rclone config

4, 7

https://a17f581e2d142fd42fd7169cd4c48c8c.r2.cloudflarestorage.com

<!-- copy 1024 motions -->
rclone copy r2:proto-data/merged4/ /workspace/merged4/ \
  --transfers=4 \
  --multi-thread-streams=16 \
  --multi-thread-chunk-size=128M \
  --progress

<!-- copy 20946 neutral -->
rclone copy r2:proto-data/20946_neutral_offset/ /workspace/20946_neutral_offset/ \
    --transfers=4 \
    --multi-thread-streams=16 \
    --multi-thread-chunk-size=128M \
    --progress

<!-- copy 20946 with 2 shapes -->
rclone copy r2:proto-data/hhi_stage1_merged6/ /workspace/hhi_stage1_merged6/ \
    --transfers=4 \
    --multi-thread-streams=16 \
    --multi-thread-chunk-size=128M \
    --progress

pip install -e . && wandb login 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 && python tools/extract_smpl_physics_features.py

python tools/build_small_multishape_subset.py \
  --num-clips 150 \
  --output /workspace/motion_cache/small150_128shape.pt

python tools/build_mixed_source_corpus.py \
--canonical-file /workspace/motion_cache/150_128shape_canonical/150_128shape_canonical_offset.pt \
--humos-file /workspace/motion_cache/small150_128shape.pt \
--output /workspace/motion_cache/150_128shape_mixed_source_weighted.pt

------

# neutral and transfer

```bash
nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor_neutral \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp.py \
  --experiment-name hhi_20946_neutral \
  --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_20946_neutral > /tmp/train_neutral.log 2>&1 &
```

------

```bash
python -u protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 4 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_1024_motion
```

```bash
nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion_tune \
    --motion-file /workspace/difficult-motions/failed_clips.pt \
    --checkpoint results/hhi_1024_motion/last.ckpt \
    --num-envs 4096 \
    --batch-size 16384 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_1024_motion_tune > /tmp/train_hhi_tune.log 2>&1 &
```

```bash
nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp.py \
  --experiment-name hhi_1024_transfer \
  --motion-file /workspace/merged4/humos_slurmrank.pt \
  --checkpoint results/hhi_1024_motion/last.ckpt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 4 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_1024_transfer > /tmp/train_hhi_transfer.log 2>&1 &
```

------

```bash
python -u protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_film.py \
    --experiment-name hhi_film_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 8192 \
    --batch-size 32768 \
    --ngpu 4 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_film_1024_motion
```

```bash
python -u protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_shape_embed.py \
    --experiment-name hhi_se_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 4 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_se_1024_motion
```

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_shape_embed.py \
  --experiment-name hhi_se_1024_motion \
  --motion-file /workspace/merged4/humos_slurmrank.pt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 4 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_se_1024_motion > /tmp/train_se.log 2>&1 &
      
Then note the PID it prints. Close your window freely — the process keeps running on the server.

To check on it later:
tail -f /tmp/train_se.log   # watch live log
ps aux | grep train_agent   # confirm still running

To stop it: kill 1437


nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_physics.py \
  --experiment-name hhi_phy_1024_motion \
  --motion-file /workspace/merged4/humos_slurmrank.pt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 4 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_phy_1024_motion > /tmp/train_se.log 2>&1 &

```bash
nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_physics.py \
  --experiment-name hhi_phy_1024_transfer \
  --motion-file /workspace/merged4/humos_slurmrank.pt \
  --checkpoint results/hhi_phy_1024_motion/last.ckpt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 4 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_phy_1024_transfer > /tmp/train_hhi_transfer.log 2>&1 &
```

```bash
nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_moe.py \
    --experiment-name hhi_moe_20946_neutral \
    --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
    --num-envs 6144 \
    --batch-size 24576 \
    --ngpu 6 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_moe_20946_neutral > /tmp/hhi_moe_20946_neutral.log 2>&1 &
```

nohup python -u protomotions/train_agent.py --robot-name smpl_mor_neutral --simulator isaacgym --experiment-path examples/experiments/mimic/mlp_moe_stable.py --experiment-name hhi_moe_20946_neutral_stable --checkpoint results/hhi_moe_20946_neutral/last.ckpt --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt --num-envs 6144 --batch-size 24576 --ngpu 6 --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl --wandb-group hhi_moe_20946_neutral_stable > /tmp/hhi_moe_20946_neutral_stable.log 2>&1 &

nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor_neutral \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide.py \
    --experiment-name hhi_wide_20946_neutral \
    --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
    --num-envs 6144 \
    --batch-size 24576 \
    --ngpu 6 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_20946_neutral > /tmp/hhi_wide_20946_neutral.log 2>&1 &


nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor_neutral \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_explore.py \
  --experiment-name hhi_wide_20946_neutral_explore \
  --checkpoint results/hhi_wide_20946_neutral/score_based.ckpt \
  --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_20946_neutral_explore > /tmp/hhi_wide_20946_neutral_explore.log 2>&1 &

---

## Stage2

> **Status (2026-07-26):** the active run is `hhi_wide_fusion_stage2_clippool` (last command
> below) — the global clip pool + fusion adapter v4 combo, now training for real for the first
> time. The three `--r2-motion-source r2:proto-data/hhi_stage2/` commands above it (lora/residual/
> plain-fusion stage2) are earlier shard-streaming iterations, superseded by the clip-pool
> approach. Kept as a fallback launch reference only — `hhi_stage2/` itself is intentionally kept
> on R2 for this reason (see R2 cleanup decision in project notes), not deleted alongside
> `hhi_stage1/`.
>
> Also note: training on RunPod was blocked for a stretch by a host-level `futex_lock_pi` crash
> inside IsaacGym's PhysX CPU dispatcher, reproduced identically on both 6xA40 and 6xRTX A6000
> RunPod configs (see `README.note.md` §37.2). Retrying the exact same launch commands on RunPod
> again on 2026-07-26 worked without the fix ever being identified — so if training locks up with
> the same symptom on a fresh pod, that's a known (if unresolved) RunPod-side flake, not a code bug.

0. One-time prerequisite (skip if you've already run this against hhi_wide_20946_neutral):
python tools/reset_morphology_normalizer.py \
  --checkpoint results/hhi_wide_20946_neutral/score_based.ckpt \
  --output results/hhi_wide_20946_neutral/last_morph_reset.ckpt

1. The streaming smoke test — deliberately using a small --epochs-per-shard (2 instead of the production default 64) so
several rotations actually happen during a short test window, since that's the whole point of this run: exercising the
rotation-safety fixes end to end.

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_lora_stage2.py \
  --experiment-name hhi_wide_lora_stage2 \
  --checkpoint results/hhi_wide_20946_neutral/last_morph_reset.ckpt \
  --r2-motion-source r2:proto-data/hhi_stage2/ \
  --motion-cache-dir /workspace/motion_cache \
  --epochs-per-shard 64 \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_lora_stage2 > /tmp/hhi_wide_lora_stage2.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_lora_stage2.py \
  --experiment-name hhi_wide_residual_stage2 \
  --checkpoint results/hhi_wide_20946_neutral/last_morph_reset.ckpt \
  --r2-motion-source r2:proto-data/hhi_stage2/ \
  --motion-cache-dir /workspace/motion_cache \
  --epochs-per-shard 64 \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_residual_stage2 > /tmp/hhi_wide_residual_stage2.log 2>&1 &

nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2.py \
    --experiment-name hhi_wide_fusion_stage2 \
    --checkpoint results/hhi_wide_20946_neutral/last_morph_reset.ckpt \
    --r2-motion-source r2:proto-data/hhi_stage2/ \
    --motion-cache-dir /workspace/motion_cache \
    --epochs-per-shard 64 \
    --num-envs 6144 \
    --batch-size 24576 \
    --ngpu 6 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_fusion_stage2 > /tmp/hhi_wide_fusion_stage2.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2.py \
  --experiment-name hhi_wide_fusion_stage2_clippool \
  --checkpoint results/hhi_wide_20946_neutral/last_morph_reset.ckpt \
  --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \
  --global-clip-pool-cache-dir /workspace/motion_cache \
  --global-clip-pool-size 256 \
  --global-clip-pool-rebuild-every 64 \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_fusion_stage2_clippool > /tmp/hhi_wide_fusion_stage2_clippool.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2.py \
  --experiment-name hhi_wide_fusion_stage2_clippool \
  --checkpoint results/hhi_wide_20946_neutral/last_morph_reset.ckpt \
  --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \
  --global-clip-pool-cache-dir /workspace/motion_cache \
  --global-clip-pool-size 256 \
  --global-clip-pool-rebuild-every 64 \
  --global-clip-pool-selection-temperature 1.0 \
  --global-clip-pool-weight-ema-alpha 0.1 \
  --global-clip-pool-difficulty-scores-path data/preprocessing/valid_ids_sorted_by_difficulty.txt \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_fusion_stage2_clippool > /tmp/hhi_wide_fusion_stage2_clippool.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2_unfrozen.py \
  --experiment-name hhi_wide_fusion_stage2_unfrozen \
  --checkpoint results/hhi_wide_fusion_stage2_clippool/last.ckpt \
  --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \
  --global-clip-pool-cache-dir /workspace/motion_cache \
  --global-clip-pool-size 256 \
  --global-clip-pool-rebuild-every 64 \
  --global-clip-pool-selection-temperature 1.0 \
  --global-clip-pool-weight-ema-alpha 0.1 \
  --global-clip-pool-difficulty-scores-path data/preprocessing/valid_ids_sorted_by_difficulty.txt \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_fusion_stage2_unfrozen > /tmp/hhi_wide_fusion_stage2_unfrozen.log 2>&1 &

python tools/deepen_fusion_trunk.py --checkpoint results/hhi_wide_fusion_stage2_unfrozen/last.ckpt --output results/hhi_wide_fusion_stage2_unfrozen/last_deepened1.ckpt --num-new-layers 1 --verify

python tools/deepen_fusion_trunk.py --checkpoint results/hhi_wide_fusion_stage2_unfrozen/last.ckpt --output results/hhi_wide_fusion_stage2_unfrozen/last_deepened1.ckpt --num-new-layers 1 --verify

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2_deepened.py \
  --experiment-name hhi_wide_fusion_stage2_deepened1 \
  --checkpoint results/hhi_wide_fusion_stage2_unfrozen/last_deepened1.ckpt \
  --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \
  --global-clip-pool-cache-dir /workspace/motion_cache \
  --global-clip-pool-size 256 \
  --global-clip-pool-rebuild-every 64 \
  --global-clip-pool-selection-temperature 1.0 \
  --global-clip-pool-weight-ema-alpha 0.1 \
  --global-clip-pool-difficulty-scores-path data/preprocessing/valid_ids_sorted_by_difficulty.txt \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_fusion_stage2_deepened1 > /tmp/hhi_wide_fusion_stage2_deepened1.log 2>&1 &


nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_stage2_scratch.py \
  --experiment-name hhi_wide_stage2_scratch \
  --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \
  --global-clip-pool-cache-dir /workspace/motion_cache \
  --global-clip-pool-size 256 \
  --global-clip-pool-rebuild-every 256 \
  --global-clip-pool-eval-holdout-size 128 \
  --global-clip-pool-weight-floor 0.05 \
  --global-clip-pool-random-fraction 0.2 \
  --num-envs 6144 \
  --batch-size 24576 \
  --ngpu 6 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_stage2_scratch > /tmp/hhi_wide_stage2_scratch.log 2>&1 &

# AMP style

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_amp.py \
  --experiment-name hhi_wide_150motion_128shape_amp \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 1 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_amp > /tmp/train_150motion_amp.log 2>&1 &

# Explore

nohup python -u protomotions/train_agent.py     --robot-name smpl_mor     --simulator isaacgym     --experiment-path examples/experiments/mimic/mlp_wide_discover.py     --experiment-name hhi_wide_150motion_128shape_discover     --motion-file /workspace/motion_cache/small150_128shape.pt     --num-envs 4096     --batch-size 16384     --ngpu 1     --use-wandb     --wandb-project hhi-protomotions     --wandb-entity yugoamaryl     --wandb-group hhi_wide_150motion_128shape_discover > /tmp/train_150motion_discover.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_softterm.py \
  --experiment-name hhi_wide_150motion_128shape_discover_softterm \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 1 \
  --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_discover_softterm > /tmp/train_150motion_discover_softterm.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_sharpen.py \
  --experiment-name hhi_wide_150motion_128shape_discover_sharpen \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 4096 --batch-size 16384 --ngpu 1 \
  --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_discover_sharpen > /tmp/hhi_wide_150motion_128shape_discover_sharpen.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_explore.py \
  --experiment-name hhi_wide_150motion_128shape_discover_explore \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 4096 --batch-size 16384 --ngpu 1 \
  --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_discover_explore > /tmp/hhi_wide_150motion_128shape_discover_explore.log 2>&1 &

nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide_discover_historical.py \
    --experiment-name hhi_wide_150motion_128shape_discover_historical \
    --motion-file /workspace/motion_cache/small150_128shape.pt \
    --num-envs 4096 --batch-size 16384 --ngpu 1 \
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_150motion_128shape_discover_historical > /tmp/hhi_wide_150motion_128shape_discover_historical.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_lookahead.py \
  --experiment-name hhi_wide_150motion_128shape_discover_lookahead \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 4096 --batch-size 16384 --ngpu 1 \
  --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_discover_lookahead > /tmp/hhi_wide_150motion_128shape_discover_lookahead.log 2>&1 & 


nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_historical_lookahead.py \
  --experiment-name hhi_wide_150motion_128shape_discover_historical_lookahead \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 6144 --batch-size 24576 --ngpu 1 \
  --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_discover_historical_lookahead > /tmp/hhi_wide_150motion_128shape_discover_historical_lookahead.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_window_match.py \
  --experiment-name hhi_wide_150motion_128shape_discover_window_match \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 6144 --batch-size 24576 --ngpu 1 \
  --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_discover_window_match > /tmp/hhi_wide_150motion_128shape_discover_window_match.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_relaxed_eval.py \
  --experiment-name hhi_wide_150motion_128shape_discover_relaxed_eval \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --num-envs 6144 --batch-size 24576 --ngpu 1 \
  --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_128shape_discover_relaxed_eval > /tmp/hhi_wide_150motion_128shape_discover_relaxed_eval.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_relaxed_rh.py \
--experiment-name hhi_wide_150motion_128shape_discover_relaxed_rh \
--motion-file /workspace/motion_cache/small150_128shape.pt \
--num-envs 6144 --batch-size 24576 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_128shape_discover_relaxed_rh > /tmp/hhi_wide_150motion_128shape_discover_relaxed_rh.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_worstk.py \
--experiment-name hhi_wide_150motion_128shape_discover_worstk \
--motion-file /workspace/motion_cache/small150_128shape.pt \
--num-envs 6144 --batch-size 24576 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_128shape_discover_worstk > /tmp/hhi_wide_150motion_128shape_discover_worstk.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_historical_lookahead.py \
--experiment-name hhi_wide_13clip_128shape_discover_historical_lookahead \
--motion-file /workspace/small_motion_cache/hard_clips_discover_lineage.pt \
--num-envs 6144 --batch-size 24576 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_13clip_128shape_discover_historical_lookahead > /tmp/hhi_wide_13clip_128shape_discover_historical_lookahead.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention.py \
--experiment-name hhi_wide_150motion_discover_attention \
--motion-file /workspace/motion_cache/small150_128shape.pt \
--num-envs 4096 --batch-size 16384 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_discover_attention > /tmp/hhi_wide_150motion_discover_attention.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention_physics.py \
--experiment-name hhi_wide_150motion_discover_attention_physics \
--motion-file /workspace/motion_cache/small150_128shape.pt \
--num-envs 4096 --batch-size 16384 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_discover_attention_physics > /tmp/hhi_wide_150motion_discover_attention_physics.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention_bigger.py \
--experiment-name hhi_wide_150motion_discover_attention_bigger \
--motion-file /workspace/motion_cache/small150_128shape.pt \
--num-envs 2048 --batch-size 8192 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_discover_attention_bigger > /tmp/hhi_wide_150motion_discover_attention_bigger.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_stage2_discover_attention.py \
--experiment-name hhi_wide_stage2_discover_attention \
--global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \
--global-clip-pool-cache-dir /workspace/motion_cache \
--global-clip-pool-size 256 --global-clip-pool-rebuild-every 256 \
--global-clip-pool-eval-holdout-size 128 --global-clip-pool-weight-floor 0.05 \
--global-clip-pool-random-fraction 0.2 \
--num-envs 2048 --batch-size 8192 --ngpu 6 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_stage2_discover_attention > /tmp/hhi_wide_stage2_discover_attention.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor \
--simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention_dofreward.py \
--experiment-name hhi_wide_150motion_128shape_discover_attention_dofreward \
--motion-file /workspace/motion_cache/150_128shape_canonical/150_128shape_canonical_offset.pt \
--num-envs 4096 \
--batch-size 16384 \
--ngpu 1 \
--use-wandb \
--wandb-project hhi-protomotions \
--wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_128shape_discover_attention_dofreward > /tmp/hhi_wide_150motion_128shape_discover_attention_dofreward.log 2>&1 &

1. Build the combined corpus:
python tools/build_mixed_source_corpus.py \
--canonical-file /workspace/motion_cache/150_128shape_canonical/150_128shape_canonical_offset.pt \
--humos-file /workspace/motion_cache/small150_128shape.pt \
--output /workspace/motion_cache/150_128shape_mixed_source.pt

  2. Launch:
nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention_source_switched.py \
--experiment-name hhi_wide_150motion_128shape_discover_attention_source_switched \
--motion-file /workspace/motion_cache/150_128shape_mixed_source.pt \
--num-envs 2048 --batch-size 8192 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_128shape_discover_attention_source_switched > /tmp/hhi_wide_150motion_128shape_discover_attention_source_switched.log 2>&1 &


nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention_source_weighted.py \
--experiment-name hhi_wide_150motion_128shape_discover_attention_source_weighted \
--motion-file /workspace/motion_cache/150_128shape_mixed_source_weighted.pt \
--num-envs 2048 --batch-size 8192 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_128shape_discover_attention_source_weighted > /tmp/hhi_wide_150motion_128shape_discover_attention_source_weighted.log 2>&1 &

nohup python -u protomotions/train_agent.py \
  --robot-name smpl_mor \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp_wide_discover_attention.py \
  --experiment-name hhi_wide_150motion_discover_attention_refined \
  --motion-file /workspace/motion_cache/small150_128shape_refined.pt \
  --num-envs 4096 \
  --batch-size 16384 \
  --ngpu 1 \
  --use-wandb \
  --wandb-project hhi-protomotions \
  --wandb-entity yugoamaryl \
  --wandb-group hhi_wide_150motion_discover_attention \
  > /tmp/hhi_wide_150motion_discover_attention_refined.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention_slot_type.py \
--experiment-name hhi_wide_150motion_discover_attention_slot_type_refined \
--motion-file /workspace/motion_cache/small150_128shape_refined.pt \
--num-envs 4096 --batch-size 16384 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_discover_attention_slot_type_refined \
> /tmp/hhi_wide_150motion_discover_attention_slot_type_refined.log 2>&1 &

nohup python -u protomotions/train_agent.py \
--robot-name smpl_mor --simulator isaacgym \
--experiment-path examples/experiments/mimic/mlp_wide_discover_attention_adaln.py \
--experiment-name hhi_wide_150motion_discover_attention_adaln_refined \
--motion-file /workspace/motion_cache/small150_128shape_refined.pt \
--num-envs 4096 --batch-size 16384 --ngpu 1 \
--use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
--wandb-group hhi_wide_150motion_discover_attention_adaln_refined \
> /tmp/hhi_wide_150motion_discover_attention_adaln_refined.log 2>&1 &

----



python tools/analyze_shape_failure_correlation.py --results-dir /workspace/ProtoMotions/results/hhi_wide_150motion_128shape_seggain --motion-file /workspace/motion_cache/small150_128shape.pt --output /workspace/ProtoMotions/results/hhi_wide_150motion_128shape_seggain/failure_correlation_analysis.txt

python tools/select_visualization_clips.py \
  --results-dir results/hhi_wide_150motion_128shape_discover_historical_lookahead \
  --motion-file /workspace/motion_cache/small150_128shape.pt \
  --checkpoint results/hhi_wide_150motion_128shape_discover_historical_lookahead/score_based.ckpt \
  --script-output results/hhi_wide_150motion_128shape_discover_historical_lookahead/render_visualize.sh

bash results/hhi_wide_150motion_128shape_discover_historical_lookahead/render_visualize.sh
-----


  docker pull hansen1416/hhi-protomotions-isaacgym:v1

  docker run -d \
  --name proto \
  --network=host \
  --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt:/mnt \
  hansen1416/hhi-protomotions-isaacgym:v1 \
  tail -f /dev/null

docker exec -it proto /bin/bash

----------

# Archive

apt update && apt install -y tmux
tmux new -s hhi
tmux new -t hhi
tmux kill-session -t hhi

tmux ls

tmux kill-server

tmux capture-pane -p -S -5000 > /tmp/tmux_log.txt
  cat /tmp/tmux_log.txt | grep -i "error\|traceback\|exception" | head -50

---

mkdir results && chmod 777 -R results && cd results && mv ../1024-raw.zip ./ && unzip 1024-raw.zip && mv results/hhi_1024_motion ./ && rm 1024-raw.zip && rm -r results && cd ../

------

apt-get update
apt-get install -y ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update

apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

--

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

---------


docker run -it --name hhi-protomotions \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  hansen1416/hhi-protomotions-isaacgym:v1 \
  /bin/bash


  ------------

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo systemctl enable --now docker




curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

codex resume 019ff1f9-8219-7931-a78c-8625ce2d843e