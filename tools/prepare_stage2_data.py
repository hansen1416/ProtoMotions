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
Stage 2 data pipeline: humos_output (.pt files) → grounding-offset MotionLib shards.

Supports two output modes (mutually exclusive):
  --output-dir   Save shards locally (for local machine / R drive)
  --r2-dest      Upload shards to R2 then delete local copy (for remote server)

Each batch of --batch-clips clips goes through five steps, cleaning up
intermediates as it goes to keep peak disk usage bounded.

  Step 1  Gather .pt files (copy from --local-humos-cache, download rest from GDrive)
  Step 2  Export HUMOS .pt → AMASS-style NPZ          [delete .pt copies]
  Step 3  Convert NPZ → MotionLib shards              [delete NPZ]
          --motions-per-shard must be a multiple of 128 (shapes per clip).
          Default 8192 = 64 clips × 128 shapes ≈ 3.4 GB per shard.
          All 128 shape variants of every clip stay in the same shard.
  Step 4  Apply frame-0 grounding offset (IsaacGym)   [replace pre-offset files]
  Step 5  Move shards to --output-dir  OR  upload to --r2-dest then delete

Resume: re-run the same command — completed batches are recorded in
{workspace}/pipeline_log.txt and skipped automatically.

--- Local usage (R drive) ---

  conda activate smplsim
  python tools/prepare_stage2_data.py \\
      --valid-ids /path/to/valid_ids_sorted_by_difficulty.txt \\
      --workspace /media/hlz/R/stage2_prep \\
      --proto-root /home/hlz/repos/ProtoMotions \\
      --asset-root /home/hlz/repos/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \\
      --output-dir /media/hlz/R/stage2_data \\
      --local-humos-cache /media/hlz/R/humos_output \\
      --batch-clips 512 \\
      --isaacgym-env isaacgym

--- Remote server usage (R2 upload) ---

  conda activate smplsim
  python tools/prepare_stage2_data.py \\
      --valid-ids /workspace/valid_ids_sorted_by_difficulty.txt \\
      --workspace /workspace/stage2_prep \\
      --proto-root /workspace/ProtoMotions \\
      --asset-root /workspace/ProtoMotions/protomotions/data/assets/mjcf/smpl_mor \\
      --r2-dest r2:proto-data/20946_humos_offset \\
      --batch-clips 4096 \\
      --isaacgym-env isaacgym
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SHAPES_PER_CLIP = 128

# From pilot: 54 GB for 131,072 motions → 0.413 MB per motion
_MB_PER_MOTION = 54.0 * 1024 / 131_072


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: str) -> None:
    """Print and run a shell command; exit on failure."""
    log(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"\nCommand failed (exit {result.returncode}):\n  {cmd}")


def load_valid_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        token = line.strip()
        if token:
            ids.append(token.split()[0].rstrip(":"))
    return ids


def write_rclone_filter(ids: list[str], filter_file: Path) -> None:
    with open(filter_file, "w") as f:
        for kid in ids:
            f.write(f"+ {kid}.pt\n")
        f.write("- *\n")


def append_log(log_file: Path, line: str) -> None:
    with open(log_file, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 data pipeline: humos_output → offset MotionLib shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required paths
    parser.add_argument(
        "--valid-ids", required=True, type=Path,
        help="Text file with valid keyids sorted by difficulty, one per line.",
    )
    parser.add_argument(
        "--workspace", required=True, type=Path,
        help="Working directory for intermediates (R drive recommended).",
    )
    parser.add_argument(
        "--proto-root", required=True, type=Path,
        help="ProtoMotions repository root.",
    )
    parser.add_argument(
        "--asset-root", required=True, type=Path,
        help="smpl_mor MJCF directory (*.xml + assets.yaml).",
    )

    # Output mode — exactly one required
    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--output-dir", type=Path,
        help="Local directory to accumulate final offset shards (local mode).",
    )
    out_group.add_argument(
        "--r2-dest", type=str,
        help="rclone R2 destination, e.g. r2:proto-data/20946_humos_offset (remote mode).",
    )

    # Source options
    parser.add_argument(
        "--local-humos-cache", type=Path, default=None,
        help=(
            "Local directory of already-downloaded HUMOS .pt files. "
            "Files found here are copied directly, skipping GDrive download. "
            "e.g. /media/hlz/R/humos_output"
        ),
    )
    parser.add_argument(
        "--gdrive-remote", default="gdrive:humos_output",
        help="rclone remote for HUMOS .pt files. (default: gdrive:humos_output)",
    )

    # Processing options
    parser.add_argument(
        "--batch-clips", type=int, default=512,
        help="Clips per batch. (default: 512 for local; use 4096 for remote)",
    )
    parser.add_argument(
        "--motions-per-shard", type=int, default=8192,
        help=(
            "Motions per output shard; must be a multiple of 128. "
            "8192 = 64 clips × 128 shapes ≈ 3.4 GB. (default: 8192)"
        ),
    )
    parser.add_argument(
        "--isaacgym-env", default="isaacgym",
        help=(
            "Conda env with IsaacGym for the offset step. "
            "Pass '' to use current python directly. (default: isaacgym)"
        ),
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device for MotionLib NPZ conversion. (default: cuda)",
    )

    args = parser.parse_args()

    # ── Validate ────────────────────────────────────────────────────────────
    if args.motions_per_shard % SHAPES_PER_CLIP != 0:
        sys.exit(
            f"--motions-per-shard ({args.motions_per_shard}) must be a multiple of "
            f"{SHAPES_PER_CLIP}."
        )

    for label, p in [
        ("--valid-ids", args.valid_ids),
        ("--proto-root", args.proto_root),
        ("--asset-root", args.asset_root),
    ]:
        if not p.exists():
            sys.exit(f"{label} not found: {p}")

    if args.local_humos_cache and not args.local_humos_cache.exists():
        sys.exit(f"--local-humos-cache not found: {args.local_humos_cache}")

    tools = args.proto_root / "tools"
    export_script  = tools / "export_humos_to_amass_npz.py"
    convert_script = tools / "convert_amass_to_motionlib_with_morphology.py"
    offset_script  = tools / "compute_humos_frame0_offsets.py"

    for p in [export_script, convert_script, offset_script]:
        if not p.exists():
            sys.exit(f"Tool not found: {p}")

    offset_python = (
        f"conda run -n {args.isaacgym_env} --no-capture-output python"
        if args.isaacgym_env else "python"
    )
    local_mode = args.output_dir is not None

    # ── Setup ────────────────────────────────────────────────────────────────
    valid_ids = load_valid_ids(args.valid_ids)
    batches = [
        valid_ids[i : i + args.batch_clips]
        for i in range(0, len(valid_ids), args.batch_clips)
    ]

    clips_per_shard = args.motions_per_shard // SHAPES_PER_CLIP
    shard_gb = args.motions_per_shard * _MB_PER_MOTION / 1024

    args.workspace.mkdir(parents=True, exist_ok=True)
    if local_mode:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.workspace / "pipeline_log.txt"

    log(f"\n{'=' * 64}")
    log(f"Stage 2 pipeline — {now()}")
    log(f"  Valid clips     : {len(valid_ids)}")
    log(f"  Batches         : {len(batches)} × up to {args.batch_clips} clips")
    log(f"  Motions/shard   : {args.motions_per_shard}  ({clips_per_shard} clips × {SHAPES_PER_CLIP} shapes ≈ {shard_gb:.1f} GB)")
    log(f"  Output mode     : {'local → ' + str(args.output_dir) if local_mode else 'R2 → ' + args.r2_dest}")
    log(f"  Local cache     : {args.local_humos_cache or 'none'}")
    log(f"  Workspace       : {args.workspace}")
    log(f"  Log file        : {log_file}")
    log(f"{'=' * 64}\n")

    # ── Resume state ─────────────────────────────────────────────────────────
    completed: set[str] = set()
    if log_file.exists():
        for line in log_file.read_text().splitlines():
            if "DONE" in line:
                parts = line.split()
                # format: "YYYY-MM-DD HH:MM:SS  batch_NNNN  DONE  ..."
                for p in parts:
                    if p.startswith("batch_"):
                        completed.add(p)
                        break
    if completed:
        log(f"Resuming: {len(completed)}/{len(batches)} batches already complete.\n")

    # ── Batch loop ────────────────────────────────────────────────────────────
    for batch_idx, batch_ids in enumerate(batches):
        batch_name = f"batch_{batch_idx:04d}"

        if batch_name in completed:
            log(f"[{batch_name}] Already complete — skipping.")
            continue

        log(f"\n{'─' * 64}")
        log(f"[{batch_name}]  {len(batch_ids)} clips  ({batch_idx + 1}/{len(batches)})  {now()}")
        log(f"{'─' * 64}")
        append_log(log_file, f"{now()}  {batch_name}  STARTED  {len(batch_ids)} clips")

        raw_dir    = args.workspace / "raw"
        npz_dir    = args.workspace / "npz"
        proto_dir  = args.workspace / "proto"
        offset_dir = args.workspace / "offset"

        for d in [raw_dir, npz_dir, proto_dir, offset_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        # ── Step 1: Gather .pt files ─────────────────────────────────────────
        log(f"\n[{batch_name}] Step 1/5  Gather .pt files")

        # Copy from local cache first (hard-link if same filesystem, else copy)
        cached, missing = [], []
        for kid in batch_ids:
            if args.local_humos_cache:
                src = args.local_humos_cache / f"{kid}.pt"
                if src.exists():
                    dst = raw_dir / f"{kid}.pt"
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                    cached.append(kid)
                    continue
            missing.append(kid)

        log(f"  From local cache : {len(cached)} clips")
        log(f"  From GDrive      : {len(missing)} clips")

        if missing:
            filter_file = args.workspace / "rclone_filter.txt"
            write_rclone_filter(missing, filter_file)
            run(
                f"rclone copy {args.gdrive_remote}/ {raw_dir}/ "
                f"--filter-from {filter_file} --transfers=16 --progress"
            )

        downloaded = list(raw_dir.glob("*.pt"))
        log(f"  Total in raw_dir : {len(downloaded)} files")
        if not downloaded:
            sys.exit(f"[{batch_name}] No .pt files gathered — check cache and rclone config.")

        # ── Step 2: Export .pt → AMASS NPZ ──────────────────────────────────
        log(f"\n[{batch_name}] Step 2/5  Export HUMOS .pt → AMASS NPZ")
        yaml_name = f"{batch_name}.yaml"
        run(
            f"python {export_script} "
            f"--input-dir {raw_dir} "
            f"--out-root {npz_dir} "
            f"--yaml-name {yaml_name} "
            f"--fps 30.0 "
            f"--skip-existing"
        )
        shutil.rmtree(raw_dir)
        log(f"  raw_dir deleted.")

        # ── Step 3: Convert NPZ → MotionLib shards ───────────────────────────
        log(f"\n[{batch_name}] Step 3/5  Convert NPZ → MotionLib shards")
        yaml_path = npz_dir / yaml_name
        run(
            f"python {convert_script} "
            f"{npz_dir} {proto_dir} "
            f"--motion-config {yaml_path} "
            f"--humanoid-type smpl "
            f"--output-fps 30 "
            f"--device {args.device} "
            f"--batch-size {args.motions_per_shard}"
        )
        shutil.rmtree(npz_dir)
        log(f"  npz_dir deleted.")

        # ── Step 4: Frame-0 grounding offset (one shard at a time) ──────────
        log(f"\n[{batch_name}] Step 4/5  Apply frame-0 grounding offsets (IsaacGym)")
        chunks = sorted(proto_dir.glob("*.pt"))
        log(f"  {len(chunks)} shards")
        for i, chunk in enumerate(chunks):
            log(f"  [{i + 1}/{len(chunks)}] {chunk.name}")
            out_path = offset_dir / f"{chunk.stem}_offset.pt"
            run(
                f"{offset_python} {offset_script} "
                f"--motion-file {chunk} "
                f"--asset-root {args.asset_root} "
                f"--out-motion-file {out_path} "
                f"--limit -1 --overwrite"
            )
            chunk.unlink()  # free proto shard immediately after offset is written
        proto_dir.rmdir()
        log(f"  proto_dir deleted.")

        # ── Step 5: Save output ───────────────────────────────────────────────
        offset_shards = sorted(offset_dir.glob("*.pt"))
        log(f"\n[{batch_name}] Step 5/5  Save {len(offset_shards)} offset shards")

        if local_mode:
            for shard in offset_shards:
                shutil.move(str(shard), args.output_dir / shard.name)
            shutil.rmtree(offset_dir)
            log(f"  Moved to {args.output_dir}")
        else:
            run(
                f"rclone copy {offset_dir}/ {args.r2_dest}/ "
                f"--transfers=4 --s3-upload-concurrency=4 --s3-chunk-size=64M "
                f"--retries=10 --retries-sleep=30s --progress"
            )
            shutil.rmtree(offset_dir)
            log(f"  Uploaded to {args.r2_dest}")

        append_log(
            log_file,
            f"{now()}  {batch_name}  DONE  "
            f"{len(batch_ids)} clips  {len(offset_shards)} shards  "
            f"cached={len(cached)}  downloaded={len(missing)}"
        )
        log(f"\n[{batch_name}] Complete.  {now()}")

    log(f"\n{'=' * 64}")
    log(f"All {len(batches)} batches complete.  {now()}")
    if local_mode:
        log(f"Shards saved to: {args.output_dir}")
    else:
        log(f"Shards on R2:   {args.r2_dest}")
    log(f"Log:            {log_file}")
    log(f"{'=' * 64}")


if __name__ == "__main__":
    main()
