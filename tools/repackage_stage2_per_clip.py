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
Split Stage 2 R2 shards (328 files, 64 clips x 128 shapes = 8192 motions each) into
per-clip files (one .pt per clip_id, holding all 128 shape variants of that clip).

This is step 1 of note/README.stage2-global-clip-sampling-plan.md: a global,
clip-keyed weight vector needs a way to load/evict individual clips independent of
the 64-clip packaging batches they originally shipped in. Splitting is a pure
re-partition of already-computed data (no retargeting/physics re-run) -- each source
shard already distinguishes its 64 clips via the `motion_clip_ids` field written by
tools/convert_amass_to_motionlib_with_morphology.py.

Per-shard flow (bounded local disk: at most one shard's raw + split output on disk
at a time, ~3.4 GB source + ~3.4 GB split output):
  1. Download one shard from --r2-source
  2. Load it, group its 8192 motions by motion_clip_ids (typically 64 groups of 128)
  3. For each group, gather per-frame tensors (gts/grs/gvs/gavs/dvs/dps/contacts/lrs)
     by each motion's frame range and re-concatenate into a standalone per-clip file
     with freshly computed length_starts; slice per-motion tensors/tuples the same way
  4. Upload the shard's ~64 per-clip files to --r2-dest, append to the local clip
     manifest (clip_id -> per-clip filename), re-upload the manifest
  5. Delete local shard + per-clip files, move to the next shard

Resume: re-run the same command -- shards already processed are recorded in
{workspace}/repackage_log.txt and skipped.

Test on a couple shards before running the full 328:

    python tools/repackage_stage2_per_clip.py \\
        --r2-source r2:proto-data/hhi_stage2/ \\
        --r2-dest r2:proto-data/hhi_stage2_per_clip/ \\
        --workspace /workspace/repackage_prep \\
        --limit-shards 2

Full run (same command, no --limit-shards):

    python tools/repackage_stage2_per_clip.py \\
        --r2-source r2:proto-data/hhi_stage2/ \\
        --r2-dest r2:proto-data/hhi_stage2_per_clip/ \\
        --workspace /workspace/repackage_prep
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

SHAPES_PER_CLIP = 128

PER_FRAME_FIELDS = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs"]
PER_MOTION_TENSOR_FIELDS = [
    "motion_lengths",
    "motion_dt",
    "motion_num_frames",
    "motion_weights",
    "motion_betas",
    "motion_gender_ids",
]
PER_MOTION_TUPLE_FIELDS = [
    "motion_files",
    "motion_genders",
    "motion_beta_keys",
    "motion_asset_ids",
    "motion_clip_ids",
    "motion_npz_files",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: str) -> None:
    log(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"\nCommand failed (exit {result.returncode}):\n  {cmd}")


def list_remote_shards(r2_source: str) -> list[str]:
    result = subprocess.run(
        ["rclone", "lsf", "--files-only", r2_source],
        check=True,
        capture_output=True,
        text=True,
    )
    files = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not files:
        sys.exit(f"No files found at {r2_source} (rclone lsf returned nothing)")
    return files


def group_indices_by_clip(motion_clip_ids: tuple) -> dict:
    groups: dict = {}
    for i, clip_id in enumerate(motion_clip_ids):
        groups.setdefault(clip_id, []).append(i)
    return groups


def split_shard_into_clips(shard: dict, shard_name: str) -> dict:
    """Return {clip_id: per-clip packaged dict} for one loaded shard's contents."""
    if "motion_clip_ids" not in shard:
        sys.exit(f"Shard {shard_name} has no motion_clip_ids field -- cannot split by clip.")

    clip_ids = shard["motion_clip_ids"]
    length_starts = shard["length_starts"]
    motion_num_frames = shard["motion_num_frames"]

    groups = group_indices_by_clip(clip_ids)

    out = {}
    for clip_id, idxs in groups.items():
        if len(idxs) != SHAPES_PER_CLIP:
            log(
                f"  WARNING: clip {clip_id} in {shard_name} has {len(idxs)} motions "
                f"(expected {SHAPES_PER_CLIP}) -- keeping as-is."
            )
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        clip_data: dict = {}

        # Per-frame tensors: gather each motion's frame range, in group order.
        for field in PER_FRAME_FIELDS:
            src = shard.get(field)
            if src is None:
                continue
            pieces = [
                src[length_starts[i] : length_starts[i] + motion_num_frames[i]]
                for i in idxs
            ]
            clip_data[field] = torch.cat(pieces, dim=0)

        # Recompute length_starts for the new, standalone per-clip file.
        new_num_frames = motion_num_frames[idx_tensor]
        new_length_starts = torch.cat(
            [
                torch.zeros(1, dtype=new_num_frames.dtype),
                torch.cumsum(new_num_frames, dim=0)[:-1],
            ]
        )
        clip_data["length_starts"] = new_length_starts
        clip_data["motion_num_frames"] = new_num_frames

        # Per-motion tensors.
        for field in PER_MOTION_TENSOR_FIELDS:
            src = shard.get(field)
            if src is None or field == "motion_num_frames":
                continue
            clip_data[field] = src[idx_tensor]

        # Per-motion tuples.
        for field in PER_MOTION_TUPLE_FIELDS:
            src = shard.get(field)
            if src is None:
                continue
            clip_data[field] = tuple(src[i] for i in idxs)

        out[clip_id] = clip_data

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Stage 2 R2 shards into per-clip files (global clip-priority sampling prep).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--r2-source", required=True, help="e.g. r2:proto-data/hhi_stage2/")
    parser.add_argument("--r2-dest", required=True, help="e.g. r2:proto-data/hhi_stage2_per_clip/")
    parser.add_argument("--workspace", required=True, type=Path, help="Local scratch directory.")
    parser.add_argument(
        "--limit-shards", type=int, default=None,
        help="Only process the first N shards -- use this to validate before the full run.",
    )
    parser.add_argument("--transfers", type=int, default=4, help="rclone --transfers for uploads.")
    args = parser.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)
    shard_dir = args.workspace / "shard"
    clips_dir = args.workspace / "clips"
    log_file = args.workspace / "repackage_log.txt"
    manifest_file = args.workspace / "clip_manifest.jsonl"

    remote_files = list_remote_shards(args.r2_source)
    if args.limit_shards:
        remote_files = remote_files[: args.limit_shards]

    log(f"\n{'=' * 64}")
    log(f"Stage 2 per-clip repackaging -- {now()}")
    log(f"  Source shards : {len(remote_files)} at {args.r2_source}")
    log(f"  Dest          : {args.r2_dest}")
    log(f"  Workspace     : {args.workspace}")
    log(f"  Log file      : {log_file}")
    log(f"  Manifest      : {manifest_file}")
    log(f"{'=' * 64}\n")

    completed: set = set()
    if log_file.exists():
        for line in log_file.read_text().splitlines():
            if "DONE" in line:
                for token in line.split():
                    if token.endswith(".pt"):
                        completed.add(token)
                        break
    if completed:
        log(f"Resuming: {len(completed)}/{len(remote_files)} shards already complete.\n")

    for shard_idx, shard_name in enumerate(remote_files):
        if shard_name in completed:
            log(f"[{shard_name}] Already complete -- skipping.")
            continue

        log(f"\n{'-' * 64}")
        log(f"[{shard_name}]  ({shard_idx + 1}/{len(remote_files)})  {now()}")
        log(f"{'-' * 64}")

        for d in (shard_dir, clips_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        # Step 1: download this shard.
        run(
            f"rclone copy {args.r2_source.rstrip('/')}/{shard_name} {shard_dir}/ "
            f"--transfers=1 --multi-thread-streams=16 --multi-thread-chunk-size=128M "
            f"--retries=10 --retries-sleep=30s --s3-no-check-bucket --progress"
        )

        local_shard_path = shard_dir / shard_name
        log(f"  Loading {local_shard_path} ...")
        shard = torch.load(local_shard_path, map_location="cpu", weights_only=False)

        # Step 2-3: split into per-clip dicts.
        clip_dicts = split_shard_into_clips(shard, shard_name)
        del shard
        log(f"  Split into {len(clip_dicts)} clips.")

        manifest_lines = []
        for clip_id, clip_data in clip_dicts.items():
            out_name = f"{clip_id}.pt"
            torch.save(clip_data, clips_dir / out_name)
            manifest_lines.append(
                json.dumps({"clip_id": clip_id, "file": out_name, "source_shard": shard_name})
            )
        del clip_dicts
        shutil.rmtree(shard_dir)

        # Step 4: upload this shard's per-clip files, extend + re-upload the manifest.
        run(
            f"rclone copy {clips_dir}/ {args.r2_dest.rstrip('/')}/ "
            f"--transfers={args.transfers} --s3-upload-concurrency=4 --s3-chunk-size=64M "
            f"--retries=10 --retries-sleep=30s --s3-no-check-bucket --progress"
        )
        shutil.rmtree(clips_dir)

        with open(manifest_file, "a") as f:
            for line in manifest_lines:
                f.write(line + "\n")
        run(
            f"rclone copy {manifest_file} {args.r2_dest.rstrip('/')}/ "
            f"--s3-no-check-bucket --progress"
        )

        with open(log_file, "a") as f:
            f.write(f"{now()}  {shard_name}  DONE  {len(manifest_lines)} clips\n")

        log(f"  [{shard_name}] done, uploaded, local files cleaned up.")

    log(f"\nAll done. Manifest: {manifest_file} (uploaded to {args.r2_dest}).")


if __name__ == "__main__":
    main()
