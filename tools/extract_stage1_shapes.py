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
Filter Stage 2 (20,946 clips x 128 shapes) shards down to a small fixed set of body
shapes, reading from and writing to R2 directly. Run on a remote server with rclone
already configured for the `r2:` remote.

Keeps every clip, keeps only --asset-ids of each clip's 128 shape variants. Relies on
the Stage 2 shard guarantee (all 128 shape variants of a clip land in the same shard,
see note/README.stage2-data-pipeline.md), so this is a pure per-shard filter -- no
cross-shard bookkeeping needed. One shard (~3.4 GB) resident on local disk at a time.

This becomes the new "Stage 1" -- see note/README.note.md #19 for why (the old Stage 1,
`hhi_20946_neutral`, was single-shape and didn't meet expectations) and why
male_71fbbe41 + female_71fbbe41 (same beta_key, near population-median mass/height)
were picked over shape extremes.

Bandwidth note: downloading is NOT reduced by filtering -- each shard bundles all 128
shapes of its clips together, so getting all 20,946 clips means pulling every shard
regardless of how many shapes are kept (~1.1 TB either way). Only storage/upload/
everything downstream shrinks (N/128 the size, e.g. ~17 GB for N=2).

Resume: re-run the same command -- shards already written are recorded in
{workspace}/filter_log.txt and skipped automatically.

Not implemented: parallel/overlapping shard processing (download of shard N+1 while
filtering shard N). This is bandwidth-bound and processes shards strictly sequentially,
same as tools/prepare_stage2_data.py -- a reasonable place to speed things up later if
the ~335-shard sequential loop turns out to be too slow in practice.

Usage:
    python tools/extract_stage1_shapes.py \\
        --r2-source r2:proto-data/hhi_stage2/ \\
        --r2-dest r2:proto-data/hhi_stage1/ \\
        --workspace /workspace/stage1_prep \\
        --asset-ids male_71fbbe41 female_71fbbe41
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import torch

DEFAULT_ASSET_IDS = ["male_71fbbe41", "female_71fbbe41"]

FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "lrs", "contacts")
PER_MOTION_TENSOR_KEYS = (
    "motion_lengths", "motion_dt", "motion_num_frames", "motion_weights",
    "motion_betas", "motion_gender_ids",
)
PER_MOTION_TUPLE_KEYS = (
    "motion_genders", "motion_beta_keys", "motion_asset_ids",
    "motion_clip_ids", "motion_npz_files", "motion_files",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: List[str]) -> None:
    """Print and run a command; exit on failure."""
    log(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nCommand failed (exit {result.returncode}):\n  {' '.join(cmd)}")


def append_log(log_file: Path, line: str) -> None:
    with open(log_file, "a") as f:
        f.write(line + "\n")


def list_remote_shards(r2_source: str) -> List[str]:
    """rclone lsf, sorted for a deterministic, resume-friendly order."""
    result = subprocess.run(
        ["rclone", "lsf", r2_source, "--files-only"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"rclone lsf failed (exit {result.returncode}):\n{result.stderr}")
    return sorted(n.strip() for n in result.stdout.splitlines() if n.strip())


# ---------------------------------------------------------------------------
# Per-shard filtering
# ---------------------------------------------------------------------------

def filter_shard(data: dict, target_asset_ids: Set[str]) -> Optional[dict]:
    """Keep every clip in `data`, keep only `target_asset_ids` of each clip's shapes."""
    asset_ids = data["motion_asset_ids"]
    mask = [aid in target_asset_ids for aid in asset_ids]
    indices = [i for i, m in enumerate(mask) if m]
    if not indices:
        return None

    length_starts = data["length_starts"]
    motion_num_frames = data["motion_num_frames"]

    frame_chunks = {k: [] for k in FRAME_KEYS if k in data}
    meta_chunks = {k: [] for k in PER_MOTION_TENSOR_KEYS if k in data}
    tuple_lists = {k: [] for k in PER_MOTION_TUPLE_KEYS if k in data}

    for i in indices:
        start = length_starts[i].item()
        n = motion_num_frames[i].item()
        for k in frame_chunks:
            frame_chunks[k].append(data[k][start: start + n])
        for k in meta_chunks:
            meta_chunks[k].append(data[k][i: i + 1])
        for k in tuple_lists:
            tuple_lists[k].append(data[k][i])

    result = {}
    for k, chunks in frame_chunks.items():
        result[k] = torch.cat(chunks, dim=0)
    for k, vals in meta_chunks.items():
        result[k] = torch.cat(vals, dim=0)
    for k, vals in tuple_lists.items():
        result[k] = tuple(vals)

    num_frames = result["motion_num_frames"]
    starts = torch.zeros(len(num_frames), dtype=torch.long)
    starts[1:] = torch.cumsum(num_frames[:-1], dim=0)
    result["length_starts"] = starts

    # Fresh curriculum state -- these are freshly filtered training shards, not a
    # post-training artifact carrying over prior motion_weights.
    result["motion_weights"] = torch.ones(len(num_frames))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--r2-source", default="r2:proto-data/hhi_stage2/",
                         help="R2 path to the full 20,946x128 Stage 2 shards.")
    parser.add_argument("--r2-dest", default="r2:proto-data/hhi_stage1/",
                         help="R2 path to write filtered shards to.")
    parser.add_argument("--workspace", required=True, type=Path,
                         help="Local scratch dir. One shard resident at a time (~3.4 GB).")
    parser.add_argument("--asset-ids", nargs="+", default=DEFAULT_ASSET_IDS,
                         help=f"Shape asset_ids to keep. Default: {DEFAULT_ASSET_IDS}")
    parser.add_argument("--rclone-transfers", type=int, default=4)
    parser.add_argument("--rclone-retries", type=int, default=10)
    args = parser.parse_args()

    target_asset_ids = set(args.asset_ids)
    in_dir = args.workspace / "in"
    out_dir = args.workspace / "out"
    for d in [in_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = args.workspace / "filter_log.txt"

    log(f"\n{'=' * 64}")
    log(f"Stage 1 v2 shape filter -- {now()}")
    log(f"  Source        : {args.r2_source}")
    log(f"  Dest          : {args.r2_dest}")
    log(f"  Target shapes : {sorted(target_asset_ids)}")
    log(f"  Workspace     : {args.workspace}")
    log(f"{'=' * 64}\n")

    log("Listing remote shards ...")
    shard_names = list_remote_shards(args.r2_source)
    if not shard_names:
        sys.exit(f"No files found at {args.r2_source} -- check the path and rclone config.")
    log(f"Found {len(shard_names)} shards.\n")

    # -- Resume state --
    completed: Set[str] = set()
    if log_file.exists():
        for line in log_file.read_text().splitlines():
            if "DONE" in line:
                for p in line.split():
                    if p.endswith(".pt"):
                        completed.add(p)
                        break
    if completed:
        log(f"Resuming: {len(completed)}/{len(shard_names)} shards already complete.\n")

    total_motions_kept = 0
    total_clips_seen = 0
    empty_shards: List[str] = []

    for idx, shard_name in enumerate(shard_names):
        tag = f"[{idx + 1}/{len(shard_names)}] {shard_name}"

        if shard_name in completed:
            log(f"{tag}  already complete -- skipping")
            continue

        log(f"\n{tag}")

        # Step 1: download this shard only
        run([
            "rclone", "copy", f"{args.r2_source.rstrip('/')}/{shard_name}", str(in_dir),
            f"--transfers={args.rclone_transfers}",
            f"--retries={args.rclone_retries}", "--retries-sleep=30s",
        ])
        in_path = in_dir / shard_name

        # Step 2: filter locally
        data = torch.load(in_path, weights_only=False, map_location="cpu")
        n_clips = len(set(data["motion_clip_ids"]))
        filtered = filter_shard(data, target_asset_ids)
        del data
        in_path.unlink()

        if filtered is None:
            log(f"  WARNING: no motions matched {sorted(target_asset_ids)} in this "
                f"shard -- skipping")
            empty_shards.append(shard_name)
            append_log(log_file, f"{now()}  {shard_name}  DONE  0 motions (no match)")
            continue

        n_motions = len(filtered["motion_num_frames"])
        n_clips_kept = len(set(filtered["motion_clip_ids"]))
        expected = n_clips_kept * len(target_asset_ids)
        if n_motions != expected:
            log(f"  WARNING: expected {expected} motions ({n_clips_kept} clips x "
                f"{len(target_asset_ids)} shapes), got {n_motions} -- not every clip "
                f"in this shard has all target shapes.")

        # Step 3: save + upload, same filename so resume/tracking is trivial
        out_path = out_dir / shard_name
        torch.save(filtered, out_path)
        run([
            "rclone", "copy", str(out_path), args.r2_dest,
            f"--transfers={args.rclone_transfers}",
            f"--retries={args.rclone_retries}", "--retries-sleep=30s",
        ])
        out_path.unlink()

        total_motions_kept += n_motions
        total_clips_seen += n_clips
        log(f"  {n_clips} clips -> {n_motions} motions kept")
        append_log(log_file, f"{now()}  {shard_name}  DONE  {n_motions} motions")

    log(f"\n{'=' * 64}")
    log(f"Done. {total_clips_seen} clips seen, {total_motions_kept} motions kept "
        f"({len(target_asset_ids)} shapes each).")
    if empty_shards:
        log(f"WARNING: {len(empty_shards)} shards had no matching motions: {empty_shards}")
    log(f"{'=' * 64}")


if __name__ == "__main__":
    main()
