#!/usr/bin/env bash
# Record 20 example videos from hhi_wide_20946_neutral (Stage 1, neutral shape, single actor
# per clip). 10 "easy" + 10 "harder but still-successful" motions, selected from
# results/persistent_failures_final_wide.txt (fail counts across 43 eval snapshots, epochs
# 200-8600) -- all 20 succeed in the large majority of analyzed snapshots, just spanning
# easy (near-0% fail) to occasionally-challenging (up to ~21% fail: kneel/squat/single-leg
# balance/backward walk/kicks).
#
# Usage:
#   bash tools/record_smoke_videos_20946_neutral.sh
#   CHECKPOINT=... MOTION_DIR=... OUT_DIR=... bash tools/record_smoke_videos_20946_neutral.sh
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-results/hhi_wide_20946_neutral/last.ckpt}"
MOTION_DIR="${MOTION_DIR:-/workspace/20946_neutral_offset}"
OUT_DIR="${OUT_DIR:-output/videos/hhi_wide_20946_neutral_smoke}"
mkdir -p "$OUT_DIR"

# global_idx = rank * 3491 + local_motion_index (6 ranks, 3491 clips/rank, see
# note/README.failed-motions-20946-neutral.md). Each rank has its own shard file:
# ${MOTION_DIR}/humanml3d_neutral_20946_<rank:04d>.pt

# global_idx  category  rank  local_idx  tag
MOTIONS=(
  "323   easy 0 323  imitates_biting_waves_hand"
  "1413  easy 0 1413 stands_waves_arm"
  "2970  easy 0 2970 walks_presses_button"
  "3135  easy 0 3135 walks_slowly_reaches_arm"
  "499   easy 0 499  walks_cleans_hands"
  "2704  easy 0 2704 beckoning_waving"
  "3538  easy 1 47   dances_moving_arms"
  "3606  easy 1 115  walks_hands_at_side"
  "3717  easy 1 226  dopily_walks_silly_gestures"
  "3765  easy 1 274  claps_both_hands"
  "4749  hard 1 1258 kneels_lifts_heavy_object"
  "4566  hard 1 1075 kicks_forward_left_backward_right"
  "5039  hard 1 1548 walks_backwards_arms_out"
  "6289  hard 1 2798 walks_on_one_foot_turns"
  "6857  hard 1 3366 balances_on_left_foot"
  "7080  hard 2 98   steps_kicks_forward_right_leg"
  "9937  hard 2 2955 kicks_backward_left_foot"
  "10183 hard 2 3201 does_squats"
  "10441 hard 2 3459 hops_one_foot_walks_backward_forward"
  "16747 hard 4 2783 knee_to_elbow_then_squat"
)

for entry in "${MOTIONS[@]}"; do
  read -r gidx cat rank local tag <<< "$entry"
  motion_file="${MOTION_DIR}/humanml3d_neutral_20946_$(printf '%04d' "$rank").pt"
  out="${OUT_DIR}/${cat}_g${gidx}_${tag}.mp4"
  echo "=== [$cat] global_idx=$gidx rank=$rank local_idx=$local -> $out ==="
  python protomotions/record_video_mor.py \
    --checkpoint "$CHECKPOINT" \
    --simulator isaacgym \
    --motion-file "$motion_file" \
    --motion-index "$local" \
    --output "$out"
done

echo "Done. Videos in $OUT_DIR"
