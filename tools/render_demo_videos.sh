#!/usr/bin/env bash
# Renders the 30 hand-picked demo clips (simple -> fancy) for the MorphMimic project page,
# each showing the SAME clip across 16 different trained body shapes side by side.
#
# Run on a GPU pod (record_video_mor.py requires --simulator isaacgym). Run from the
# ProtoMotions repo root.
#
# Prerequisites on the pod:
#   - the final checkpoint (see note/README.runpod.md for the R2 download command)
#   - the 150-clip refined motion file, rebuilt directly on the pod via:
#       python tools/build_small_multishape_subset.py \
#         --num-clips 150 --output data_cache/small150_128shape_refined.pt
#     (reads from r2:proto-data/hhi_stage2_per_clip_refined/ -- the raw/unrefined R2 copy
#     was deleted during the September cleanup, so this is the only way to get it now)
#
# CHECKPOINT/MOTION_FILE default to the paths above but can be overridden without editing
# this file, e.g. if you built the motion file under a different path/name:
#   CHECKPOINT=/workspace/... MOTION_FILE=/workspace/motion_cache/small150_128shape.pt \
#     bash tools/render_demo_videos.sh

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt}"
MOTION_FILE="${MOTION_FILE:-data_cache/small150_128shape_refined.pt}"
OUT_DIR="${OUT_DIR:-output/videos/demo30}"
NUM_SHAPES="${NUM_SHAPES:-16}"

mkdir -p "$OUT_DIR"

# tier:index:clip_id -- index is the motion-index of that clip's first (shape 0) variant;
# --same-motion finds the other 127 shape-variants of the same clip automatically.
CLIPS=(
  "01_static:0:M004501"
  "02_static:128:004226"
  "03_static:13696:M000508"
  "04_static:17536:M009226"
  "05_walk:6656:007081"
  "06_walk:8960:007329"
  "07_walk:6016:M003872"
  "08_walk:9216:000667"
  "09_walk:12672:M007973"
  "10_dynamic_walk:7168:M006362"
  "11_dynamic_walk:14976:007891"
  "12_medium:17920:M002632"
  "13_medium:12160:M010318"
  "14_medium:11904:010504"
  "15_medium:8448:M007469"
  "16_medium:11776:M003925"
  "17_medium:16512:008125"
  "18_medium:11648:M010197"
  "19_jump:12416:M008156"
  "20_jump:13952:M002534"
  "21_jump:13440:M013848"
  "22_jump:7936:M009021"
  "23_jump:19072:003773"
  "24_kick:5248:M001062"
  "25_kick:15360:M008265"
  "26_kick:3968:M013236"
  "27_kick:18816:014400"
  "28_dance:17280:M014401"
  "29_dance:18944:008912"
  "30_dance:15104:M003998"
)

for entry in "${CLIPS[@]}"; do
  IFS=':' read -r tier idx clip_id <<< "$entry"
  out="${OUT_DIR}/${tier}_${clip_id}.mp4"
  if [ -f "$out" ]; then
    echo "skip (exists): $out"
    continue
  fi
  echo "=== rendering ${tier} (clip ${clip_id}, motion-index ${idx}) -> ${out} ==="
  python -u protomotions/record_video_mor.py \
    --checkpoint "$CHECKPOINT" \
    --simulator isaacgym \
    --motion-file "$MOTION_FILE" \
    --motion-index "$idx" \
    --num-envs "$NUM_SHAPES" \
    --same-motion \
    --compact-spawn-spacing 2.0 \
    --fps 30 \
    --output "$out" \
    --overrides motion_lib._target_=protomotions.components.motion_lib.MotionLib
done

echo "Done. Videos in ${OUT_DIR}/"
