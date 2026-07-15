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
#
"""
Streaming MotionLib for Stage 2 (328 R2 shards, ~1.1 TB, too big to fit in RAM/VRAM at once).

Design doc: note/README.stage2-streaming-loader-plan.md. Each rank rotates through its own
disjoint slice of the remote shard list, one shard resident locally at a time plus one
background-prefetched shard, swapping in place (same MotionLib object, same convention
`MotionLib.load_from_file` already uses) on a schedule that is a pure function of
`current_epoch` -- no cursor file, resume just recomputes the target shard.
"""

import logging
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig

log = logging.getLogger(__name__)


@dataclass
class StreamingMotionLibConfig(MotionLibConfig):
    """Configuration for a streaming, shard-rotating MotionLib (Stage 2)."""

    _target_: str = "protomotions.components.motion_lib_pool.MotionLibPool"
    r2_source: str = field(
        default="r2:proto-data/hhi_stage2/",
        metadata={"help": "rclone remote directory of per-shard .pt motion files."},
    )
    local_cache_dir: str = field(
        default="/workspace/motion_cache",
        metadata={
            "help": "Local directory to cache downloaded shards. Bounded at 2 files per "
            "rank (current + prefetched) regardless of world size."
        },
    )
    epochs_per_shard: int = field(
        default=64,
        metadata={
            "help": "Number of training epochs to run on each shard before rotating to "
            "the next one in this rank's shuffled slice."
        },
    )
    shard_shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Seed for the deterministic shard shuffle/rank-partition. Must stay "
            "fixed across a run (including resume) for rotation to remain reproducible."
        },
    )


class FileDownloader:
    """Background-thread wrapper around `rclone copy <remote_file> <local_dir>`.

    Retries the whole `rclone copy` invocation a few times on top of rclone's own internal
    `--retries` -- covers the rclone process itself dying (network reset, transient auth
    failure) rather than a single transfer within it failing.
    """

    MAX_ATTEMPTS = 3
    RETRY_BACKOFF_SECONDS = 30.0

    def __init__(self, remote_path: str, local_dir: Path):
        self.remote_path = remote_path
        self.remote_name = Path(remote_path).name
        self.local_dir = Path(local_dir)
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def start(self) -> "FileDownloader":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                subprocess.run(
                    [
                        "rclone",
                        "copy",
                        self.remote_path,
                        str(self.local_dir),
                        "--retries=10",
                        "--retries-sleep=30s",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self._error = None
                return
            except BaseException as e:  # surfaced to the caller via wait()
                self._error = e
                log.warning(
                    f"[MotionLibPool] download attempt {attempt}/{self.MAX_ATTEMPTS} failed "
                    f"for {self.remote_path}: {e}"
                )
                if attempt < self.MAX_ATTEMPTS:
                    time.sleep(self.RETRY_BACKOFF_SECONDS)

    def is_ready(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise RuntimeError(
                f"Failed to download {self.remote_path}: {self._error}"
            ) from self._error


class MotionLibPool(MotionLib):
    """MotionLib that streams shard-by-shard from a remote (rclone) source.

    Each rank owns a disjoint, deterministically shuffled slice of the remote file list and
    rotates through it on an epoch schedule (`target_shard_idx = (current_epoch //
    epochs_per_shard) % len(rank_files)`). Rotation mutates this same object in place via the
    inherited `load_from_file`, so every other reference to this MotionLib (env,
    motion_manager, agent) keeps working unmodified.
    """

    def __init__(self, config: StreamingMotionLibConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._asset_id_to_motion_ids_cache = None
        self.get_motion_state_use_blend = config.get_motion_state_use_blend
        self.different_motion_files_across_ranks = True

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        remote_files = self._list_remote_shards(config.r2_source)
        rng = random.Random(config.shard_shuffle_seed)
        rng.shuffle(remote_files)
        self.rank_files: List[str] = remote_files[self.rank :: self.world_size]
        if not self.rank_files:
            raise RuntimeError(
                f"No shards assigned to rank {self.rank} (world_size={self.world_size}) "
                f"out of {len(remote_files)} files found at {config.r2_source}. "
                "world_size is larger than the number of remote shards."
            )
        log.info(
            f"[MotionLibPool] rank {self.rank}/{self.world_size}: "
            f"{len(self.rank_files)}/{len(remote_files)} shards assigned"
        )

        self.local_cache_dir = Path(config.local_cache_dir) / f"rank{self.rank}"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        self._current_shard_idx: Optional[int] = None
        self._current_local_path: Optional[Path] = None
        self._prefetcher: Optional[FileDownloader] = None

        # shard idx -> num_motions in that shard, for every shard idx this rank has ever
        # loaded. A dict (not a running counter) so a wraparound revisit of an already-seen
        # shard doesn't double count -- see `distinct_motions_seen`.
        self._visited_shard_motion_counts: dict = {}

        # Synchronous: env/simulator construction needs real motion data immediately.
        self._ensure_shard_loaded(0, force=True)

    @property
    def distinct_motions_seen(self) -> int:
        """Total distinct motions loaded by this rank so far (sums each visited shard's
        actual motion count once, regardless of how many times it's been revisited via
        wraparound)."""
        return sum(self._visited_shard_motion_counts.values())

    @staticmethod
    def _list_remote_shards(r2_source: str) -> List[str]:
        try:
            result = subprocess.run(
                ["rclone", "lsf", "--files-only", r2_source],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "`rclone` was not found on PATH -- required for StreamingMotionLibConfig."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"`rclone lsf {r2_source}` failed (rc={e.returncode}): {e.stderr}"
            ) from e

        files = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
        if not files:
            raise RuntimeError(f"No files found at {r2_source} (rclone lsf returned nothing)")
        return files

    def _remote_path(self, name: str) -> str:
        return f"{self.config.r2_source.rstrip('/')}/{name}"

    def _next_shard_idx(self, idx: int) -> int:
        return (idx + 1) % len(self.rank_files)

    def target_shard_idx(self, current_epoch: int) -> int:
        return (current_epoch // self.config.epochs_per_shard) % len(self.rank_files)

    def _start_prefetch(self, idx: int) -> None:
        remote_name = self.rank_files[idx]
        local_path = self.local_cache_dir / remote_name
        if local_path.exists():
            # Already local (e.g. small rank_files list wrapping around quickly) -- no-op.
            self._prefetcher = None
            return
        self._prefetcher = FileDownloader(
            self._remote_path(remote_name), self.local_cache_dir
        ).start()

    def _ensure_shard_loaded(self, idx: int, force: bool = False) -> bool:
        if not force and idx == self._current_shard_idx:
            return False

        remote_name = self.rank_files[idx]
        local_path = self.local_cache_dir / remote_name

        if self._prefetcher is not None and self._prefetcher.remote_name == remote_name:
            self._prefetcher.wait()
        elif not local_path.exists():
            FileDownloader(self._remote_path(remote_name), self.local_cache_dir).start().wait()

        log.info(f"[MotionLibPool] rank {self.rank}: loading shard {remote_name} (idx={idx})")
        self.load_from_file(str(local_path))
        self.motion_file = str(local_path)
        self._current_local_path = local_path
        self._current_shard_idx = idx
        self._prefetcher = None
        self._visited_shard_motion_counts[idx] = self.num_motions()
        # load_from_file() only overwrites the fields present in the saved packaged file
        # (motion_lib.py:754) -- it never touches this cache, so a stale shard-0 asset_id ->
        # motion_ids mapping would otherwise silently outlive every later rotation.
        self._asset_id_to_motion_ids_cache = None

        self._start_prefetch(self._next_shard_idx(idx))

        # Enforce "at most 2 local files per rank" even across a discontinuous jump (e.g. a
        # resume landing on a different shard than whatever was mid-prefetch) that would
        # otherwise leave a superseded download orphaned on disk.
        keep = {local_path.name}
        if self._prefetcher is not None:
            keep.add(self._prefetcher.remote_name)
        for f in self.local_cache_dir.glob("*.pt"):
            if f.name not in keep:
                f.unlink()

        return True

    def maybe_rotate(self, current_epoch: int) -> bool:
        """Rotate to the shard `current_epoch` maps to, if different from the loaded one."""
        return self._ensure_shard_loaded(self.target_shard_idx(current_epoch))

    def sync_to_epoch(self, current_epoch: int) -> None:
        """Unconditionally (re)load the shard `current_epoch` maps to. Used after resume."""
        self._ensure_shard_loaded(self.target_shard_idx(current_epoch), force=True)
