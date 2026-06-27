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
Extract gravity-core clips (floor_contact + squat/crouch + kneel + sit/stand) from the
pilot 1024-clip offset shards and save as a new MotionLib .pt file for targeted evaluation.

Gravity-core clips are those requiring large COM displacement (crawling, kneeling, squatting,
sitting/standing transitions) — the primary failure class across all pilot training runs.

Usage:
    python tools/extract_gravity_core_clips.py \\
        --failures-file results/hhi_1024_motion/persistent_failures.txt \\
        --shard-dir /home/hlz/datasets/humos_proto/offset \\
        --output /home/hlz/datasets/humos_proto/gravity_core_offset.pt

The output file is a valid MotionLib .pt file with the same schema as the input shards.
motion_weights are reset to 1.0 (fresh curriculum state).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch


# ---------------------------------------------------------------------------
# Gravity-core keyword categories
# ---------------------------------------------------------------------------

GRAVITY_CORE_KWS = {
    "floor_contact": [
        "crawl", "all fours", "on hands", "on knees", "on the floor",
        "on all fours", "on his knees", "on her knees", "on the ground",
        "hands and knees",
    ],
    "squat_crouch": [
        "squat", "crouch", "crouches", "crouching", "crouched",
        "bent knees", "bends knees", "lower", "lowering", "lowers",
    ],
    "kneel": [
        "kneel", "kneels", "kneeling", "kneeled", "gets on knees",
    ],
    "sit_stand": [
        "sits down", "sit down", "sitting down", "gets up", "stand up",
        "stands up", "rising", "rises", "get up", "getting up",
        "lying", "lies down", "lie down", "lay down", "lays down",
    ],
}


def is_gravity_core(desc: str) -> bool:
    desc = desc.lower()
    return any(
        kw in desc
        for kws in GRAVITY_CORE_KWS.values()
        for kw in kws
    )


def parse_gravity_core_clip_ids(failures_file: Path) -> Dict[str, str]:
    """Return {clip_id: description} for all gravity-core clips."""
    result = {}
    with open(failures_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            clip_id = parts[1]
            desc = parts[4]
            if is_gravity_core(desc):
                result[clip_id] = desc
    return result


# ---------------------------------------------------------------------------
# Per-shard extraction
# ---------------------------------------------------------------------------

FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "lrs", "contacts")
PER_MOTION_TENSOR_KEYS = (
    "motion_lengths", "motion_dt", "motion_num_frames", "motion_weights",
    "motion_betas", "motion_gender_ids",
)
PER_MOTION_TUPLE_KEYS = (
    "motion_genders", "motion_beta_keys", "motion_asset_ids",
    "motion_clip_ids", "motion_npz_files", "motion_files",
)


def extract_from_shard(data: dict, target_clip_ids: Set[str]) -> Optional[dict]:
    """Extract all motions whose clip_id is in target_clip_ids from one shard."""
    clip_ids = data["motion_clip_ids"]
    mask = [cid in target_clip_ids for cid in clip_ids]
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
            frame_chunks[k].append(data[k][start : start + n])
        for k in meta_chunks:
            meta_chunks[k].append(data[k][i : i + 1])
        for k in tuple_lists:
            tuple_lists[k].append(data[k][i])

    result = {}
    for k, chunks in frame_chunks.items():
        result[k] = torch.cat(chunks, dim=0)
    for k, vals in meta_chunks.items():
        result[k] = torch.cat(vals, dim=0)
    for k, vals in tuple_lists.items():
        result[k] = tuple(vals)

    # length_starts recomputed at merge time
    result["_num_frames"] = result["motion_num_frames"]  # scratch field, removed later
    return result


# ---------------------------------------------------------------------------
# Merge extracts from multiple shards
# ---------------------------------------------------------------------------

def merge_extracts(extracts: List[dict]) -> dict:
    if len(extracts) == 1:
        merged = extracts[0]
    else:
        merged = {}
        for k in FRAME_KEYS:
            chunks = [e[k] for e in extracts if k in e]
            if chunks:
                merged[k] = torch.cat(chunks, dim=0)
        for k in PER_MOTION_TENSOR_KEYS:
            chunks = [e[k] for e in extracts if k in e]
            if chunks:
                merged[k] = torch.cat(chunks, dim=0)
        for k in PER_MOTION_TUPLE_KEYS:
            vals = []
            for e in extracts:
                if k in e:
                    vals.extend(list(e[k]))
            if vals:
                merged[k] = tuple(vals)

    # Recompute length_starts from scratch
    num_frames = merged["motion_num_frames"]
    starts = torch.zeros(len(num_frames), dtype=torch.long)
    starts[1:] = torch.cumsum(num_frames[:-1], dim=0)
    merged["length_starts"] = starts

    # Reset curriculum weights
    merged["motion_weights"] = torch.ones(len(num_frames))

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failures-file", default="results/hhi_1024_motion/persistent_failures.txt",
                        help="persistent_failures.txt from hhi_1024_motion training run")
    parser.add_argument("--shard-dir", default="/home/hlz/datasets/humos_proto/offset",
                        help="Directory containing humos_131072_NNNN_offset.pt shards")
    parser.add_argument("--output", default="/home/hlz/datasets/humos_proto/gravity_core_offset.pt",
                        help="Output .pt file path")
    args = parser.parse_args()

    failures_file = Path(args.failures_file)
    shard_dir = Path(args.shard_dir)
    output_path = Path(args.output)

    # 1. Parse gravity-core clip IDs
    gc_clips = parse_gravity_core_clip_ids(failures_file)
    target_ids = set(gc_clips.keys())
    print(f"Gravity-core clips: {len(target_ids)}")
    for cat, kws in GRAVITY_CORE_KWS.items():
        count = sum(1 for d in gc_clips.values()
                    if any(kw in d.lower() for kw in kws))
        print(f"  {cat}: {count}")

    # 2. Scan shards
    shards = sorted(shard_dir.glob("humos_131072_????_offset.pt"))
    if not shards:
        raise FileNotFoundError(f"No offset shards found in {shard_dir}")
    print(f"\nScanning {len(shards)} shards ...")

    extracts = []
    found_clip_ids: Set[str] = set()

    for shard_path in shards:
        print(f"  {shard_path.name} ... ", end="", flush=True)
        data = torch.load(shard_path, weights_only=False, map_location="cpu")

        # Quick check before full extraction
        shard_clips = set(data["motion_clip_ids"])
        hits = shard_clips & target_ids
        if not hits:
            print("skip (no gravity-core clips)")
            del data
            continue

        extract = extract_from_shard(data, target_ids)
        del data

        if extract is not None:
            n_motions = len(extract["motion_num_frames"])
            n_clips = len(set(extract["motion_clip_ids"]))
            found_clip_ids.update(extract["motion_clip_ids"])
            print(f"{n_clips} clips / {n_motions} motions")
            extracts.append(extract)
        else:
            print("skip")

    if not extracts:
        raise RuntimeError("No gravity-core motions found in any shard.")

    # 3. Merge
    print("\nMerging ...")
    merged = merge_extracts(extracts)

    n_motions = len(merged["motion_num_frames"])
    n_clips = len(set(merged["motion_clip_ids"]))
    n_frames = merged["gts"].shape[0]

    # 4. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    size_mb = output_path.stat().st_size / 1024**2

    print(f"\nSaved: {output_path}")
    print(f"  Clips:    {n_clips}  (expected ~{len(target_ids)})")
    print(f"  Motions:  {n_motions}  ({n_motions // 128} clips × 128 shapes)")
    print(f"  Frames:   {n_frames:,}")
    print(f"  Size:     {size_mb:.1f} MB")

    missing = target_ids - found_clip_ids
    if missing:
        print(f"\nWARNING: {len(missing)} clips not found in any shard:")
        for cid in sorted(missing):
            print(f"  {cid}  {gc_clips[cid][:70]}")


if __name__ == "__main__":
    main()
