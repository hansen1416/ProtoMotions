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
Extract persistently-failing clips from the shard dataset into a standalone .pt file.

Reads results/hhi_1024_motion/persistent_failures.txt (or a custom failures file),
selects clips above a configurable avg-betas-failed threshold, then slices the
corresponding motions (all 128 body-shape variants per clip) out of the local
shard files and writes a new self-contained .pt file in the same format.

Memory strategy: each shard is loaded once, the needed clips are extracted and
immediately saved to a small per-shard temp file, then the shard is freed.
At the end, all temp files are merged into the final output. This keeps peak
RAM usage to roughly (one shard) + (one clip group), rather than all shards
simultaneously.

The output file can be used directly as a --motion-file argument for training
or inference, e.g. to fine-tune on hard clips only.

Usage:
    python tools/extract_failed_motions.py \\
        --failures results/hhi_1024_motion/persistent_failures.txt \\
        --shard-dir /home/hlz/datasets/humos_proto/offset \\
        --shard-pattern "humos_131072_{idx:04d}_offset.pt" \\
        --out /home/hlz/datasets/humos_proto/failed_clips.pt \\
        --min-avg-betas 5.0

    # Tighter threshold (65 clips — clearly failing):
    python tools/extract_failed_motions.py ... --min-avg-betas 8.0

    # Explicit list of global clip indices:
    python tools/extract_failed_motions.py ... --clip-indices 172 338 426 59
"""

from __future__ import annotations

import argparse
import gc
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch


CLIPS_PER_SHARD = 64
BETAS_PER_CLIP = 128

FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs")
MOTION_TENSOR_KEYS = (
    "motion_lengths", "motion_dt", "motion_num_frames",
    "motion_weights", "motion_betas", "motion_gender_ids",
)
MOTION_TUPLE_KEYS = (
    "motion_files", "motion_genders", "motion_beta_keys",
    "motion_asset_ids", "motion_clip_ids", "motion_npz_files",
)


# ---------------------------------------------------------------------------
# Parsing the failures file
# ---------------------------------------------------------------------------

def parse_failures(failures_file: Path, min_avg_betas: float) -> List[int]:
    """Return sorted list of global_clip_idx values at or above the threshold."""
    selected = []
    line_re = re.compile(r"^\s*(\d+)\s+\S+\s+\d+/\d+\s+([\d.]+)β")
    with open(failures_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = line_re.match(line)
            if not m:
                continue
            gci = int(m.group(1))
            avg_betas = float(m.group(2))
            if avg_betas >= min_avg_betas:
                selected.append(gci)
    return sorted(set(selected))


# ---------------------------------------------------------------------------
# Single-shard extraction (called once per shard, result saved to temp file)
# ---------------------------------------------------------------------------

def _extract_from_shard(
    shard: dict,
    positions: List[int],
) -> dict:
    """Extract clips at the given within-shard positions (0-63) from a loaded shard."""
    length_starts: torch.Tensor = shard["length_starts"]
    motion_num_frames: torch.Tensor = shard["motion_num_frames"]

    frame_parts: Dict[str, list] = {k: [] for k in FRAME_KEYS}
    motion_tensor_parts: Dict[str, list] = {k: [] for k in MOTION_TENSOR_KEYS}
    motion_tuple_parts: Dict[str, list] = {k: [] for k in MOTION_TUPLE_KEYS}

    for pos in sorted(positions):
        m_start = pos * BETAS_PER_CLIP
        m_end = m_start + BETAS_PER_CLIP

        f_start = int(length_starts[m_start].item())
        last_start = int(length_starts[m_end - 1].item())
        last_len = int(motion_num_frames[m_end - 1].item())
        f_end = last_start + last_len

        clip_id = shard["motion_clip_ids"][m_start]
        print(
            f"      clip pos={pos:3d}  id={clip_id}  "
            f"motions[{m_start}:{m_end}]  frames[{f_start}:{f_end}]"
        )

        for k in FRAME_KEYS:
            if k in shard and shard[k] is not None:
                frame_parts[k].append(shard[k][f_start:f_end].cpu())

        for k in MOTION_TENSOR_KEYS:
            motion_tensor_parts[k].append(shard[k][m_start:m_end].cpu())

        for k in MOTION_TUPLE_KEYS:
            seq = shard[k]
            motion_tuple_parts[k].extend(seq[m_start:m_end])

    chunk: dict = {}
    for k in FRAME_KEYS:
        if frame_parts[k]:
            chunk[k] = torch.cat(frame_parts[k], dim=0)
    for k in MOTION_TENSOR_KEYS:
        chunk[k] = torch.cat(motion_tensor_parts[k], dim=0)
    for k in MOTION_TUPLE_KEYS:
        chunk[k] = tuple(motion_tuple_parts[k])
    return chunk


# ---------------------------------------------------------------------------
# Main extraction loop (streaming, one shard at a time)
# ---------------------------------------------------------------------------

def extract_clips(
    clip_indices: List[int],
    shard_dir: Path,
    shard_pattern: str,
    temp_dir: Optional[Path] = None,
) -> dict:
    """
    Extract all 128 beta variants for each requested global_clip_idx.

    Processes one shard at a time to keep peak RAM manageable.
    Returns a merged dict ready for torch.save.
    """
    # Group by shard
    shard_to_positions: Dict[int, List[int]] = {}
    for gci in clip_indices:
        shard_idx = gci // CLIPS_PER_SHARD
        pos = gci % CLIPS_PER_SHARD
        shard_to_positions.setdefault(shard_idx, []).append(pos)

    n_shards = len(shard_to_positions)
    temp_files: List[Path] = []

    use_temp = temp_dir is not None
    if use_temp:
        temp_dir.mkdir(parents=True, exist_ok=True)

    for i, shard_idx in enumerate(sorted(shard_to_positions)):
        shard_path = shard_dir / shard_pattern.format(idx=shard_idx)
        positions = shard_to_positions[shard_idx]
        print(
            f"  [{i+1}/{n_shards}] Shard {shard_idx}: {shard_path.name} "
            f"— extracting {len(positions)} clip(s)"
        )

        shard = torch.load(shard_path, weights_only=False, map_location="cpu")
        chunk = _extract_from_shard(shard, positions)
        del shard
        gc.collect()

        if use_temp:
            tmp_path = temp_dir / f"chunk_{shard_idx:04d}.pt"
            torch.save(chunk, tmp_path)
            temp_files.append(tmp_path)
            del chunk
            gc.collect()
        else:
            temp_files.append(chunk)  # keep in memory if no temp dir

    # Merge all chunks
    print("\n  Merging chunks...")
    all_frame: Dict[str, list] = {k: [] for k in FRAME_KEYS}
    all_motion_tensor: Dict[str, list] = {k: [] for k in MOTION_TENSOR_KEYS}
    all_motion_tuple: Dict[str, list] = {k: [] for k in MOTION_TUPLE_KEYS}

    for item in temp_files:
        chunk = torch.load(item, weights_only=False) if isinstance(item, Path) else item
        for k in FRAME_KEYS:
            if k in chunk:
                all_frame[k].append(chunk[k])
        for k in MOTION_TENSOR_KEYS:
            all_motion_tensor[k].append(chunk[k])
        for k in MOTION_TUPLE_KEYS:
            all_motion_tuple[k].extend(list(chunk[k]))
        if isinstance(item, Path):
            item.unlink()
        del chunk
        gc.collect()

    result: dict = {}
    for k in FRAME_KEYS:
        if all_frame[k]:
            result[k] = torch.cat(all_frame[k], dim=0)
    for k in MOTION_TENSOR_KEYS:
        result[k] = torch.cat(all_motion_tensor[k], dim=0)
    for k in MOTION_TUPLE_KEYS:
        result[k] = tuple(all_motion_tuple[k])

    # Recompute length_starts
    shifted = result["motion_num_frames"].roll(1)
    shifted[0] = 0
    result["length_starts"] = shifted.cumsum(0)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract persistently-failing clips into a standalone .pt file."
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/hhi_1024_motion/persistent_failures.txt"),
        help="Path to persistent_failures.txt",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("/home/hlz/datasets/humos_proto/offset"),
        help="Directory containing shard .pt files",
    )
    parser.add_argument(
        "--shard-pattern",
        default="humos_131072_{idx:04d}_offset.pt",
        help="Filename pattern for shard files (uses {idx} placeholder)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/hlz/datasets/humos_proto/failed_clips.pt"),
        help="Output .pt file path",
    )
    parser.add_argument(
        "--min-avg-betas",
        type=float,
        default=5.0,
        help=(
            "Minimum avg betas failed per epoch to include a clip "
            "(default 5.0; use 8.0 for the 65 most severe clips)"
        ),
    )
    parser.add_argument(
        "--clip-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override: explicit global_clip_idx values instead of reading failures file",
    )
    parser.add_argument(
        "--no-temp",
        action="store_true",
        help="Keep extracted chunks in RAM instead of spilling to temp files (faster but uses more memory)",
    )
    args = parser.parse_args()

    # Determine clip selection
    if args.clip_indices is not None:
        clip_indices = sorted(set(args.clip_indices))
        print(f"Using {len(clip_indices)} explicit clip indices.")
    else:
        print(f"Parsing {args.failures}")
        clip_indices = parse_failures(args.failures, args.min_avg_betas)
        print(
            f"Selected {len(clip_indices)} clips with avg_betas >= {args.min_avg_betas}"
        )

    if not clip_indices:
        print("No clips selected. Exiting.")
        return

    total_motions = len(clip_indices) * BETAS_PER_CLIP
    print(
        f"Extracting {len(clip_indices)} clips × {BETAS_PER_CLIP} betas "
        f"= {total_motions} motions\n"
    )

    if args.no_temp:
        temp_dir = None
    else:
        tmp_root = args.out.parent / ".extract_tmp"
        temp_dir = tmp_root

    result = extract_clips(clip_indices, args.shard_dir, args.shard_pattern, temp_dir)

    n_motions = result["motion_num_frames"].shape[0]
    total_frames = result["gts"].shape[0]
    unique_clips = len(set(result["motion_clip_ids"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {args.out} ...")
    torch.save(result, args.out)

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nDone.")
    print(f"  Unique clips:  {unique_clips}")
    print(f"  Total motions: {n_motions}")
    print(f"  Total frames:  {total_frames}")
    print(f"  File size:     {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
