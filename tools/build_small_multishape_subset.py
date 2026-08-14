# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Build a single static, multi-shape motion file out of a small, difficulty-stratified subset of
the Stage 2 per-clip R2 dataset (~20,951 clips x 128 shapes).

Why: the Stage 2 adapter lineage (v1-v6) and the from-scratch GlobalClipPool run both plateaued
around 63-82% success, while the same architecture on a single body shape reaches 95-97%. An
isolated ablation (one motion, all 128 shapes -- `hhi_1_motion_128_shape`) reproduces the same
low ceiling (plateaus ~65-70%), which points at shape-conditioning capacity, not curriculum/data
scale, as the bottleneck (see note/README.note.md). Before spending more GPU-hours confirming that
at the full 20,946-clip scale, this script builds a small (~100-200 clip), full-128-shape, PLAIN
STATIC motion file -- no GlobalClipPool, no residency streaming, no curriculum-composition
confound -- so `mlp_wide.py` (unmodified, same architecture as the from-scratch stage2 run) can be
iterated on quickly and the "can this architecture clear 95%+ once GPU-hours-per-clip is no longer
scarce" question gets a fast, clean answer.

Two selection modes:

1. Difficulty-stratified (default, `--num-clips`): clips are NOT chosen randomly or as "the
   easiest N" -- they're stratified evenly across
   data/preprocessing/valid_ids_sorted_by_difficulty.txt (the same file GlobalClipPoolConfig uses
   to seed its priority scoreboard) so the subset spans the full easy->hard range in the same
   proportion as the full dataset, and a result on it says something about the real bottleneck
   rather than about an easy slice. This is how small150_128shape.pt itself was built.

2. Fixed clip-id list (`--clip-ids-file`): builds a subset out of an explicit, pre-selected list
   of clip_ids instead of sampling -- e.g. the frozen hard-clip intersection produced by
   tools/find_persistent_hard_clips.py (note/README.note.md §55 item 1). Use this to materialize
   a small, difficulty-frozen dataset for fast iteration against a specific known-hard tail,
   rather than a representative easy->hard spread.

Usage (run on the pod -- needs rclone + R2 credentials, see note/README.rclone.md):
    python tools/build_small_multishape_subset.py \\
        --num-clips 150 \\
        --output /workspace/motion_cache/small150_128shape.pt

    # or, from a frozen hard-clip list:
    python tools/build_small_multishape_subset.py \\
        --clip-ids-file results/analysis/hard_clips_discover_lineage.clip_ids.txt \\
        --output /workspace/motion_cache/hard_clips_discover_lineage.pt

Then train with the existing (unmodified) mlp_wide.py:
    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide.py \\
        --experiment-name hhi_wide_150motion_128shape \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --num-envs 4096 --batch-size 16384
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import torch

from protomotions.components.global_clip_pool import GlobalClipPool


def load_difficulty_sorted_clip_ids(difficulty_file: str) -> List[str]:
    """Returns clip_ids in the same order as the file -- already sorted easy(0.0) -> hard(1.0)."""
    clip_ids = []
    with open(difficulty_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            clip_id, _score = line.split(":", 1)
            clip_ids.append(clip_id.strip())
    return clip_ids


def stratified_sample(sorted_clip_ids: List[str], num_clips: int) -> List[str]:
    """Evenly-spaced picks across the difficulty-sorted list, so the subset spans the full
    easy->hard range in roughly the same proportion as the full dataset (not just the easiest
    N, and not a uniform-random sample that could clump)."""
    n = len(sorted_clip_ids)
    if num_clips >= n:
        return list(sorted_clip_ids)
    step = n / num_clips
    indices = sorted({int(i * step) for i in range(num_clips)})
    return [sorted_clip_ids[i] for i in indices]


def download_manifest(r2_source: str, manifest_name: str, cache_dir: Path) -> Dict[str, str]:
    manifest_dir = cache_dir / "_manifest_dl"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    remote_manifest = f"{r2_source.rstrip('/')}/{manifest_name}"
    local_manifest = manifest_dir / manifest_name

    if not local_manifest.exists():
        subprocess.run(
            [
                "rclone", "copy", remote_manifest, str(manifest_dir),
                "--s3-no-check-bucket", "--retries=10", "--retries-sleep=30s",
            ],
            check=True,
        )

    clip_id_to_remote_name = {}
    with open(local_manifest, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            clip_id_to_remote_name[entry["clip_id"]] = entry["file"]
    return clip_id_to_remote_name


def download_clips(r2_source: str, remote_names: List[str], cache_dir: Path) -> None:
    missing = [name for name in remote_names if not (cache_dir / name).exists()]
    if not missing:
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(missing))
        list_path = f.name
    try:
        subprocess.run(
            [
                "rclone", "copy", r2_source.rstrip("/"), str(cache_dir),
                f"--files-from={list_path}",
                "--transfers=8", "--checkers=8",
                "--retries=10", "--retries-sleep=30s", "--s3-no-check-bucket",
            ],
            check=True,
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


def load_fixed_clip_ids(clip_ids_file: str) -> List[str]:
    """One clip_id per line (e.g. the sidecar written by tools/find_persistent_hard_clips.py)."""
    with open(clip_ids_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-clips", type=int, default=150)
    parser.add_argument(
        "--difficulty-file", default="data/preprocessing/valid_ids_sorted_by_difficulty.txt"
    )
    parser.add_argument(
        "--clip-ids-file",
        default=None,
        help="If given, use this fixed clip_id list instead of difficulty-stratified sampling "
        "(--num-clips/--difficulty-file are ignored). One clip_id per line.",
    )
    parser.add_argument("--r2-source", default="r2:proto-data/hhi_stage2_per_clip/")
    parser.add_argument("--manifest-name", default="clip_manifest.jsonl")
    parser.add_argument("--cache-dir", default="/workspace/motion_cache/small_subset_raw")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.clip_ids_file is not None:
        selected = load_fixed_clip_ids(args.clip_ids_file)
        print(f"Loaded {len(selected)} fixed clip_ids from {args.clip_ids_file}")
    else:
        sorted_clip_ids = load_difficulty_sorted_clip_ids(args.difficulty_file)
        print(f"Loaded {len(sorted_clip_ids)} difficulty-sorted clip_ids from {args.difficulty_file}")

        selected = stratified_sample(sorted_clip_ids, args.num_clips)
        print(f"Selected {len(selected)} clips, stratified evenly across the difficulty range")

    clip_id_to_remote_name = download_manifest(args.r2_source, args.manifest_name, cache_dir)
    missing_from_manifest = [cid for cid in selected if cid not in clip_id_to_remote_name]
    if missing_from_manifest:
        raise RuntimeError(
            f"{len(missing_from_manifest)} selected clip_ids are not in the manifest, e.g. "
            f"{missing_from_manifest[:5]} -- difficulty file and manifest are out of sync."
        )

    remote_names = [clip_id_to_remote_name[cid] for cid in selected]
    print(f"Downloading {len(remote_names)} per-clip files from {args.r2_source} ...")
    download_clips(args.r2_source, remote_names, cache_dir)

    print("Loading and concatenating per-clip files ...")
    clip_dicts = []
    shape_counts = []
    for cid in selected:
        local_path = cache_dir / clip_id_to_remote_name[cid]
        clip_data = torch.load(local_path, map_location="cpu", weights_only=False)
        clip_dicts.append(clip_data)
        shape_counts.append(len(clip_data["motion_lengths"]))

    assembled = GlobalClipPool._concat_clip_dicts(clip_dicts)
    assembled["motion_weights"] = torch.ones(sum(shape_counts), dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(assembled, output_path)

    ids_sidecar = output_path.with_suffix(".clip_ids.txt")
    ids_sidecar.write_text("\n".join(selected) + "\n")

    total_motions = sum(shape_counts)
    print(f"\nSaved {output_path} ({total_motions:,} motions = {len(selected)} clips x ~128 shapes)")
    print(f"  shape_counts per clip: min={min(shape_counts)} max={max(shape_counts)}")
    print(f"  selected clip_ids written to {ids_sidecar}")


if __name__ == "__main__":
    main()
