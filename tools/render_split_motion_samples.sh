#!/usr/bin/env bash
# Render five representative training clips and five validation clips from the
# refined Stage-2 MotionLib dataset. Each video shows the first eight body-shape
# variants of the same base motion, with red reference-pose markers.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
viz_root="${1:-/workspace/motion_viz}"
remote_root="r2:proto-data/hhi_stage2_per_clip_refined"

train_clip_ids=(000000 006748 013471 M006604 M014614)
validation_clip_ids=(000022 006589 013534 M006574 M014615)

mkdir -p "${viz_root}/clips"
mkdir -p "${viz_root}/videos/train"
mkdir -p "${viz_root}/videos/validation"

render_split() {
    local split_name="$1"
    shift

    local clip_id
    local motion_file
    local output_file
    for clip_id in "$@"; do
        motion_file="${viz_root}/clips/${clip_id}.pt"
        output_file="${viz_root}/videos/${split_name}/${split_name}_${clip_id}.mp4"

        echo "[${split_name}] Downloading ${clip_id}.pt"
        rclone copyto \
            "${remote_root}/${clip_id}.pt" \
            "${motion_file}"

        if [[ ! -s "${motion_file}" ]]; then
            echo "Downloaded motion file is missing or empty: ${motion_file}" >&2
            return 1
        fi

        echo "[${split_name}] Rendering ${output_file}"
        python3 "${repo_root}/tools/render_motionlib_video.py" \
            --motion-file "${motion_file}" \
            --robot smpl_mor \
            --start 0 \
            --batch-size 8 \
            --target-markers \
            --camera-distance-scale 1.4 \
            --output "${output_file}"
    done
}

render_split train "${train_clip_ids[@]}"
render_split validation "${validation_clip_ids[@]}"

echo "Videos saved under ${viz_root}/videos"
