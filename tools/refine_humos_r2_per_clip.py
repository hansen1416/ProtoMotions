# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Stream, refine, and re-upload the Stage-2 per-clip HUMOS corpus.

The source and destination are rclone remotes. Only one clip is kept in the
workspace at a time:

1. Download one source ``.pt`` file.
2. Run ``tools/refine_humos_motion.py`` on all shape variants in that clip.
3. Upload the refined file under the same filename.
4. Verify the remote object size, then remove both local files.

The job is resumable. Existing destination ``.pt`` files are treated as
completed, and ``clip_manifest.jsonl`` is regenerated from verified destination
objects. The destination manifest is therefore partial while preprocessing is in
progress and becomes identical in coverage/order to the source manifest only when
the job completes. Do not train from the refined prefix until the script reports
``COMPLETE`` and uploads ``refinement_complete.json``.

Full run (CPU-only refinement, safe under nohup):

    nohup python3 -u tools/refine_humos_r2_per_clip.py \\
        --source r2:proto-data/hhi_stage2_per_clip/ \\
        --destination r2:proto-data/hhi_stage2_per_clip_refined/ \\
        --workspace /workspace/hhi_stage2_refinement \\
        --workers 4 --threads-per-worker 2 \\
        > /tmp/refine_hhi_stage2_per_clip.log 2>&1 &

Small end-to-end pilot (uploads the first two clips and can be resumed later):

    python3 -u tools/refine_humos_r2_per_clip.py \\
        --source r2:proto-data/hhi_stage2_per_clip/ \\
        --destination r2:proto-data/hhi_stage2_per_clip_refined/ \\
        --workspace /workspace/hhi_stage2_refinement \\
        --limit-clips 2 --report

Inspect selection/resume state without downloading, refining, or uploading:

    python3 tools/refine_humos_r2_per_clip.py --dry-run --limit-clips 10

Run only one instance per destination. A local workspace lock prevents accidental
duplicate processes on one machine; it cannot coordinate separate machines.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = "r2:proto-data/hhi_stage2_per_clip/"
DEFAULT_DESTINATION = "r2:proto-data/hhi_stage2_per_clip_refined/"
DEFAULT_WORKSPACE = Path("/workspace/hhi_stage2_refinement")
MANIFEST_NAME = "clip_manifest.jsonl"
COMPLETION_MARKER = "refinement_complete.json"

RCLONE_RETRY_ARGS = [
    "--retries=10",
    "--retries-sleep=30s",
    "--s3-no-check-bucket",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def remote_join(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


def validate_remote(value: str, label: str) -> str:
    if ":" not in value:
        raise ValueError(f"{label} must be an rclone remote path, got {value!r}")
    remote, path = value.split(":", 1)
    if not remote or not path.strip("/"):
        raise ValueError(f"{label} must name a non-root remote prefix, got {value!r}")
    return f"{remote}:{path.rstrip('/')}/"


def run_command(
    command: list[str],
    *,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log(f"$ {shlex.join(command)}")
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
        env=env,
    )


def acquire_workspace_lock(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)
    lock_path = workspace / ".refinement.lock"
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(
            f"Another refinement process holds the workspace lock: {lock_path}"
        ) from error
    lock_file.write(f"pid={os.getpid()} started={timestamp()}\n")
    lock_file.flush()
    return lock_file


def download_source_manifest(source: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "rclone",
            "copyto",
            remote_join(source, MANIFEST_NAME),
            str(local_path),
            *RCLONE_RETRY_ARGS,
        ]
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    clip_ids: set[str] = set()
    filenames: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        entry = json.loads(raw_line)
        clip_id = str(entry["clip_id"])
        filename = str(entry["file"])
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError(
                f"Unsafe/non-.pt filename at manifest line {line_number}: {filename!r}"
            )
        if clip_id in clip_ids:
            raise ValueError(
                f"Duplicate clip_id at manifest line {line_number}: {clip_id}"
            )
        if filename in filenames:
            raise ValueError(
                f"Duplicate filename at manifest line {line_number}: {filename}"
            )
        clip_ids.add(clip_id)
        filenames.add(filename)
        entries.append(entry)
    if not entries:
        raise ValueError(f"Source manifest is empty: {path}")
    return entries


def list_remote_pt_files(destination: str) -> set[str]:
    result = run_command(
        [
            "rclone",
            "lsf",
            "--files-only",
            "--include=*.pt",
            destination,
        ],
        capture_output=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def write_completed_manifest(
    entries: list[dict[str, Any]], completed: set[str], path: Path
) -> int:
    completed_entries = [entry for entry in entries if str(entry["file"]) in completed]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        for entry in completed_entries:
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
    tmp_path.replace(path)
    return len(completed_entries)


def upload_manifest(
    destination: str,
    entries: list[dict[str, Any]],
    completed: set[str],
    local_manifest: Path,
) -> None:
    count = write_completed_manifest(entries, completed, local_manifest)
    if count == 0:
        return
    run_command(
        [
            "rclone",
            "copyto",
            str(local_manifest),
            remote_join(destination, MANIFEST_NAME),
            *RCLONE_RETRY_ARGS,
        ]
    )
    log(f"Checkpointed destination manifest: {count}/{len(entries)} clips")


def download_clip(source: str, filename: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "rclone",
            "copyto",
            remote_join(source, filename),
            str(local_path),
            "--transfers=1",
            "--multi-thread-streams=8",
            "--multi-thread-chunk-size=64M",
            *RCLONE_RETRY_ARGS,
        ]
    )
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is missing or empty: {local_path}")


def refine_clip(
    python: str,
    refine_script: Path,
    input_path: Path,
    output_path: Path,
    asset_root: Path,
    device: str,
    report: bool,
    threads_per_worker: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-u",
        str(refine_script),
        "--motion-file",
        str(input_path),
        "--out-motion-file",
        str(output_path),
        "--asset-root",
        str(asset_root),
        "--device",
        device,
    ]
    if report:
        command.append("--report")
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        child_env[variable] = str(threads_per_worker)
    child_env["OMP_DYNAMIC"] = "FALSE"
    run_command(command, env=child_env)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Refinement did not create a valid output: {output_path}")


def upload_clip(destination: str, filename: str, local_path: Path) -> None:
    run_command(
        [
            "rclone",
            "copyto",
            str(local_path),
            remote_join(destination, filename),
            "--transfers=1",
            "--s3-upload-concurrency=4",
            "--s3-chunk-size=64M",
            *RCLONE_RETRY_ARGS,
        ]
    )


def verify_remote_size(destination: str, filename: str, expected_bytes: int) -> None:
    result = run_command(
        ["rclone", "size", "--json", remote_join(destination, filename)],
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    if payload.get("count") != 1 or payload.get("bytes") != expected_bytes:
        raise RuntimeError(
            f"Remote verification failed for {filename}: expected {expected_bytes} bytes, "
            f"got {payload}"
        )


def clean_local_clip(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def upload_completion_marker(
    destination: str,
    source: str,
    entries: list[dict[str, Any]],
    marker_path: Path,
) -> None:
    marker = {
        "status": "complete",
        "completed_at": timestamp(),
        "source": source,
        "destination": destination,
        "clips": len(entries),
        "refinement_tool": "tools/refine_humos_motion.py",
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    run_command(
        [
            "rclone",
            "copyto",
            str(marker_path),
            remote_join(destination, COMPLETION_MARKER),
            *RCLONE_RETRY_ARGS,
        ]
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=repo_root / "protomotions/data/assets/mjcf/smpl_mor",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of clips refined concurrently. Use 4 on an 8-vCPU pod.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help=(
            "CPU math threads per refinement process. 0 divides detected CPUs across "
            "--workers."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based source-manifest index at which to start selection.",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=-1,
        help="Select at most this many manifest entries after --start-index.",
    )
    parser.add_argument(
        "--manifest-upload-every",
        type=int,
        default=25,
        help="Upload the partial destination manifest after this many new clips.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed before/after metrics for every clip (very verbose).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect source/destination and selection without modifying either remote.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone was not found on PATH")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.limit_clips == 0 or args.limit_clips < -1:
        raise ValueError("--limit-clips must be -1 or a positive integer")
    if args.manifest_upload_every <= 0:
        raise ValueError("--manifest-upload-every must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.threads_per_worker < 0:
        raise ValueError("--threads-per-worker must be non-negative")

    detected_cpus = os.cpu_count() or 1
    threads_per_worker = args.threads_per_worker or max(
        1, detected_cpus // args.workers
    )
    if args.workers * threads_per_worker > detected_cpus:
        log(
            f"WARNING: workers*threads-per-worker={args.workers * threads_per_worker} "
            f"exceeds detected CPU count {detected_cpus}"
        )

    source = validate_remote(args.source, "--source")
    destination = validate_remote(args.destination, "--destination")
    if source == destination:
        raise ValueError("--source and --destination must be different")

    repo_root = Path(__file__).resolve().parents[1]
    refine_script = repo_root / "tools/refine_humos_motion.py"
    if not refine_script.is_file():
        raise FileNotFoundError(refine_script)
    asset_root = args.asset_root.resolve()
    if not asset_root.is_dir():
        raise FileNotFoundError(asset_root)

    workspace = args.workspace.resolve()
    lock_file = acquire_workspace_lock(workspace)
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    metadata_dir = workspace / "metadata"
    for directory in (input_dir, output_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_manifest_path = metadata_dir / "source_clip_manifest.jsonl"
    destination_manifest_path = metadata_dir / MANIFEST_NAME
    completion_marker_path = metadata_dir / COMPLETION_MARKER

    try:
        download_source_manifest(source, source_manifest_path)
        entries = load_manifest(source_manifest_path)
        manifest_filenames = {str(entry["file"]) for entry in entries}
        remote_files = list_remote_pt_files(destination)
        orphans = remote_files - manifest_filenames
        completed = remote_files & manifest_filenames

        if orphans:
            log(
                f"WARNING: destination has {len(orphans)} .pt files absent from the source "
                "manifest; they will be ignored, not deleted"
            )
        log(
            f"Source clips={len(entries)} destination-complete={len(completed)} "
            f"remaining={len(entries) - len(completed)}"
        )

        selected = entries[args.start_index :]
        if args.limit_clips > 0:
            selected = selected[: args.limit_clips]
        pending = [entry for entry in selected if str(entry["file"]) not in completed]
        log(
            f"Selection start={args.start_index} count={len(selected)} pending={len(pending)}"
        )
        log(
            f"CPU plan: workers={args.workers}, threads-per-worker={threads_per_worker}, "
            f"detected-cpus={detected_cpus}"
        )

        if args.dry_run:
            for entry in pending[:10]:
                log(f"DRY RUN pending: {entry['clip_id']} -> {entry['file']}")
            if len(pending) > 10:
                log(f"DRY RUN: ... and {len(pending) - 10} more pending clips")
            return

        if completed:
            upload_manifest(
                destination,
                entries,
                completed,
                destination_manifest_path,
            )

        new_since_manifest = 0
        worker_config = WorkerConfig(
            source=source,
            destination=destination,
            python=args.python,
            refine_script=refine_script,
            asset_root=asset_root,
            device=args.device,
            report=args.report,
            threads_per_worker=threads_per_worker,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        for filename, _output_size in process_pending_clips(
            pending, args.workers, worker_config
        ):
            completed.add(filename)
            new_since_manifest += 1

            if new_since_manifest >= args.manifest_upload_every:
                upload_manifest(
                    destination,
                    entries,
                    completed,
                    destination_manifest_path,
                )
                new_since_manifest = 0

        if completed:
            upload_manifest(
                destination,
                entries,
                completed,
                destination_manifest_path,
            )

        if len(completed) == len(entries):
            upload_completion_marker(
                destination,
                source,
                entries,
                completion_marker_path,
            )
            log(
                f"COMPLETE: refined and verified all {len(entries)} clips at {destination}"
            )
        else:
            log(
                f"PARTIAL: destination contains {len(completed)}/{len(entries)} clips. "
                "Re-run the same command to resume."
            )
    except BaseException:
        # A successful object upload remains the resume source of truth even if the
        # process fails before the next periodic manifest checkpoint.
        if "entries" in locals() and "completed" in locals() and completed:
            try:
                upload_manifest(
                    destination,
                    entries,
                    completed,
                    destination_manifest_path,
                )
            except Exception as manifest_error:
                log(f"WARNING: failed to checkpoint manifest during shutdown: {manifest_error}")
        raise
    finally:
        lock_file.close()


@dataclass(frozen=True)
class WorkerConfig:
    source: str
    destination: str
    python: str
    refine_script: Path
    asset_root: Path
    device: str
    report: bool
    threads_per_worker: int
    input_dir: Path
    output_dir: Path


def process_clip_entry(
    entry: dict[str, Any], position: int, total: int, config: WorkerConfig
) -> tuple[str, int]:
    clip_id = str(entry["clip_id"])
    filename = str(entry["file"])
    input_path = config.input_dir / filename
    output_path = config.output_dir / filename
    clean_local_clip(input_path, output_path)
    log(f"[{position}/{total}] Refining clip {clip_id} ({filename})")
    try:
        download_clip(config.source, filename, input_path)
        refine_clip(
            config.python,
            config.refine_script,
            input_path,
            output_path,
            config.asset_root,
            config.device,
            config.report,
            config.threads_per_worker,
        )
        output_size = output_path.stat().st_size
        upload_clip(config.destination, filename, output_path)
        verify_remote_size(config.destination, filename, output_size)
        log(
            f"Uploaded and verified {filename} ({output_size} bytes); "
            "removing local input/output"
        )
        return filename, output_size
    finally:
        clean_local_clip(input_path, output_path)


def process_pending_clips(
    pending: list[dict[str, Any]], workers: int, config: WorkerConfig
):
    """Yield completed filenames while keeping at most ``workers`` clips in flight."""
    if not pending:
        return

    next_index = 0
    active: dict[Future[tuple[str, int]], int] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while next_index < min(workers, len(pending)):
            future = executor.submit(
                process_clip_entry,
                pending[next_index],
                next_index + 1,
                len(pending),
                config,
            )
            active[future] = next_index
            next_index += 1

        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                yield future.result()
                if next_index < len(pending):
                    replacement = executor.submit(
                        process_clip_entry,
                        pending[next_index],
                        next_index + 1,
                        len(pending),
                        config,
                    )
                    active[replacement] = next_index
                    next_index += 1


if __name__ == "__main__":
    main()
