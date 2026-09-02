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
Global clip-priority resident pool for Stage 2 (~20,951 per-clip R2 files, too big to fit in
RAM/VRAM at once).

Design doc: note/README.stage2-global-clip-sampling-plan.md. Unlike `MotionLibPool` (shard-by-
shard rotation on a fixed epoch schedule, curriculum weights reset every rotation), this keeps a
persistent per-rank clip-priority scoreboard (`global_clip_weights` + `global_clip_visit_counts`)
that survives resident-set churn and resumes, and periodically swaps the resident set (K clips/
rank) to whichever clips currently look most important -- known-hard (per the evaluator's
success/failure signal) or still-unproven (per a UCB-style exploration bonus). Reuses the
existing full-block `MotionLib._load_motion_state_dict` swap-in-place path -- no incremental
insert/evict support is added to MotionLib's tensor indexing.
"""

import hashlib
import json
import logging
import math
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig

log = logging.getLogger(__name__)

# Batched-download retry/timeout knobs (mirrors motion_lib_pool.FileDownloader's constants, but
# applied to one rclone invocation covering all missing clips in a rebuild, not one per file --
# see GlobalClipPool._download_missing_clips for why that distinction matters).
DOWNLOAD_MAX_ATTEMPTS = 3
DOWNLOAD_RETRY_BACKOFF_SECONDS = 30.0
DOWNLOAD_BASE_TIMEOUT_SECONDS = 120.0
DOWNLOAD_PER_FILE_TIMEOUT_SECONDS = 8.0

# Field lists mirror tools/repackage_stage2_per_clip.py's per-clip packaging contract exactly.
PER_FRAME_FIELDS = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs"]
PER_MOTION_TENSOR_FIELDS = [
    "motion_lengths",
    "motion_dt",
    "motion_num_frames",
    "motion_weights",
    "motion_betas",
    "motion_gender_ids",
    "motion_source_id",
]
PER_MOTION_TUPLE_FIELDS = [
    "motion_files",
    "motion_genders",
    "motion_beta_keys",
    "motion_asset_ids",
    "motion_clip_ids",
    "motion_base_clip_ids",
    "motion_npz_files",
]


@dataclass
class GlobalClipPoolConfig(MotionLibConfig):
    """Configuration for the global clip-priority resident pool (Stage 2)."""

    _target_: str = "protomotions.components.global_clip_pool.GlobalClipPool"
    r2_source: str = field(
        default="r2:proto-data/hhi_stage2_per_clip/",
        metadata={"help": "rclone remote directory of per-clip .pt motion files + manifest."},
    )
    manifest_name: str = field(
        default="clip_manifest.jsonl",
        metadata={
            "help": "Training manifest path inside r2_source, one JSON line per clip. In "
            "legacy mode this remains the all-clips manifest."
        },
    )
    validation_manifest_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Explicit validation manifest path inside r2_source. Requires "
            "split_metadata_name and eval_holdout_size=0. There is intentionally no test "
            "manifest field in the training API."
        },
    )
    split_metadata_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Versioned split_metadata.json path inside r2_source. It binds the train "
            "and validation roles to exact files/hashes and exposes only test hash/count "
            "provenance; the test manifest itself is never downloaded."
        },
    )
    local_cache_dir: str = field(
        default="/workspace/motion_cache",
        metadata={"help": "Local directory to cache downloaded per-clip files."},
    )
    resident_pool_size: int = field(
        default=256,
        metadata={
            "help": "Number of clips resident per rank (K). Matches the validated "
            "hhi_1024_motion pilot's per-rank footprint (1024 clips / 4 ranks)."
        },
    )
    cache_size_multiplier: float = field(
        default=3.0,
        metadata={
            "help": "Local disk cache sized at this multiple of K, purely for download "
            "hysteresis near the resident-set boundary -- disk is cheap, VRAM isn't."
        },
    )
    clip_partition_shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Seed for the deterministic clip shuffle/rank-partition. Must stay fixed "
            "across a run (including resume) for the clip vocabulary to remain reproducible."
        },
    )
    pool_rebuild_every: int = field(
        default=64,
        metadata={
            "help": "Epochs between resident-pool rebuilds. Deliberately decoupled from the "
            "evaluator's eval_metrics_every: weight *updates* need real eval rollouts and stay "
            "on eval_metrics_every, but rebuilds (re-rank + swap using whatever weights "
            "currently exist) are cheap and should happen faster, so new/unproven clips get "
            "pulled in at least as often as today's shard rotation (epochs_per_shard=64)."
        },
    )
    exploration_bonus_coefficient: float = field(
        default=1.0,
        metadata={
            "help": "Scale of the UCB-style exploration bonus added to a clip's selection "
            "priority. 1.0 puts a never-visited clip's bonus on the same order of magnitude as "
            "the default initial weight (also 1.0)."
        },
    )
    download_transfers: int = field(
        default=8,
        metadata={
            "help": "rclone --transfers/--checkers for the single batched download call each "
            "rebuild issues. Bounded, internal-to-one-process concurrency -- NOT one OS "
            "subprocess per missing clip (that exhausts the open-file ulimit at K~256/rank)."
        },
    )
    selection_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature over priority *rank* (not raw value) used to sample "
            "the resident set. Higher = flatter/more exploratory; near-0 approaches "
            "deterministic top-K. Rank-based so it stays well-behaved regardless of the "
            "absolute scale of global_clip_weights."
        },
    )
    weight_ema_alpha: float = field(
        default=0.1,
        metadata={
            "help": "EMA blend factor for global_clip_weights updates: "
            "weight = alpha*target + (1-alpha)*weight, target=0 on success / 1 on failure. "
            "Replaces the old one-shot success/failure discount jump with a bounded, gradual "
            "update so a single eval tick can't move a whole cluster of clips across the "
            "selection cutoff at once."
        },
    )
    difficulty_scores_path: str = field(
        default="data/preprocessing/valid_ids_sorted_by_difficulty.txt",
        metadata={
            "help": "Path (repo-root-relative or absolute) to a 'clip_id: score' file, "
            "ascending easy->hard, used to seed global_clip_weights instead of a flat 1.0 -- "
            "gives the curriculum a real easy-first starting point before any clip has been "
            "evaluated. Not precise, just a general prior; missing ids fall back to 0.5."
        },
    )
    eval_holdout_size: int = field(
        default=0,
        metadata={
            "help": "Legacy mode only: number of clips/rank permanently excluded from the "
            "trainable vocabulary "
            "and reserved for a fixed generalization probe (see MimicEvaluator._evaluate_holdout). "
            "0 (default) disables the holdout entirely -- exactly today's behavior, no clips "
            "withheld. Holdout clips never enter global_clip_weights/_select_top_k's candidate "
            "space, so they can never be trained on and never influence the curriculum."
        },
    )
    weight_floor: float = field(
        default=0.0,
        metadata={
            "help": "Minimum value global_clip_weights can EMA-decay to on repeated success. "
            "0.0 (default) reproduces the old unbounded-decay behavior exactly. A small positive "
            "floor (e.g. 0.05) guarantees a 'solved' clip retains a nonzero selection priority "
            "forever instead of only being rediscoverable via the slow log-scale UCB bonus."
        },
    )
    random_fraction: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of each rebuild's K resident slots filled by pure uniform random "
            "selection (ignoring priority entirely), instead of the priority-rank softmax. 0.0 "
            "(default) reproduces the old fully-priority-based selection exactly. A nonzero "
            "fraction (e.g. 0.2) guarantees every clip gets periodic forced rehearsal regardless "
            "of how low its weight has decayed."
        },
    )
    # Populated from verified split metadata during GlobalClipPool construction. These init=False
    # fields remain part of dataclasses.asdict(), so train_agent stores them in
    # resolved_configs.pt/.yaml and W&B hyperparameters without mutable CLI overrides.
    split_version: Optional[str] = field(default=None, init=False)
    split_id: Optional[str] = field(default=None, init=False)
    source_manifest_sha256: Optional[str] = field(default=None, init=False)
    train_manifest_sha256: Optional[str] = field(default=None, init=False)
    validation_manifest_sha256: Optional[str] = field(default=None, init=False)
    test_manifest_sha256: Optional[str] = field(default=None, init=False)
    train_clip_count: int = field(default=0, init=False)
    validation_clip_count: int = field(default=0, init=False)
    test_clip_count: int = field(default=0, init=False)


class GlobalClipPool(MotionLib):
    """MotionLib backed by a persistent, globally-weighted resident pool of clips.

    Each rank owns a disjoint, deterministically shuffled slice of the clip manifest and keeps
    K=`resident_pool_size` of them resident at a time, chosen by `global_clip_weights +
    exploration_bonus` (a combinatorial-UCB priority). `global_clip_weights`/
    `global_clip_visit_counts` persist for the life of the run (checkpointed) and are never reset
    by a rebuild -- only the *resident set* changes; the curriculum itself doesn't forget.

    Rebuilds mutate this same object in place via the inherited `_load_motion_state_dict`, so
    every other reference (env, motion_manager, agent) keeps working unmodified -- same
    convention `MotionLibPool` already established for shard rotation.
    """

    def __init__(self, config: GlobalClipPoolConfig, device: str = "cpu"):
        assert config.max_motions is None, (
            "GlobalClipPoolConfig doesn't compose with max_motions -- resident_pool_size "
            "controls how much data is loaded instead."
        )

        self.config = config
        self.device = device
        self._asset_id_to_motion_ids_cache = None
        self._clip_id_to_motion_ids_cache = None
        self.get_motion_state_use_blend = config.get_motion_state_use_blend
        self.different_motion_files_across_ranks = True

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.local_cache_dir = Path(config.local_cache_dir) / f"rank{self.rank}"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        explicit_split = (
            config.validation_manifest_name is not None
            or config.split_metadata_name is not None
        )
        if explicit_split and (
            config.validation_manifest_name is None
            or config.split_metadata_name is None
        ):
            raise ValueError(
                "Explicit split mode requires both validation_manifest_name and "
                "split_metadata_name."
            )
        if explicit_split and config.eval_holdout_size != 0:
            raise ValueError(
                "eval_holdout_size is a legacy per-rank split and must be 0 when explicit "
                "train/validation manifests are configured."
            )

        if explicit_split:
            split_metadata = self._load_split_metadata(
                config.r2_source, config.split_metadata_name
            )
            self._validate_split_manifest_roles(split_metadata)
            clip_id_to_remote_name, train_manifest_sha256 = self._load_manifest(
                config.r2_source, config.manifest_name
            )
            validation_id_to_remote_name, validation_manifest_sha256 = self._load_manifest(
                config.r2_source, config.validation_manifest_name
            )
            overlap = set(clip_id_to_remote_name) & set(validation_id_to_remote_name)
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                raise RuntimeError(
                    f"Training and validation manifests overlap on {len(overlap)} clips, "
                    f"including {sample}."
                )
            self._validate_and_record_split_provenance(
                split_metadata,
                train_manifest_sha256,
                validation_manifest_sha256,
                len(clip_id_to_remote_name),
                len(validation_id_to_remote_name),
            )

            self.rank_clip_ids = self._partition_clip_ids(
                clip_id_to_remote_name,
                rank=self.rank,
                world_size=self.world_size,
                seed=config.clip_partition_shuffle_seed,
            )
            self.eval_holdout_clip_ids = self._partition_clip_ids(
                validation_id_to_remote_name,
                rank=self.rank,
                world_size=self.world_size,
                seed=config.clip_partition_shuffle_seed ^ 0x5EED,
            )
        else:
            clip_id_to_remote_name, train_manifest_sha256 = self._load_manifest(
                config.r2_source, config.manifest_name
            )
            config.train_manifest_sha256 = train_manifest_sha256
            config.train_clip_count = len(clip_id_to_remote_name)
            validation_id_to_remote_name = clip_id_to_remote_name

            # Legacy behavior: shuffle the all-clips manifest, partition it by rank, then carve
            # eval_holdout_size clips off each rank. New experiments should use explicit global
            # manifests because this legacy holdout membership changes with world_size.
            rank_clip_ids_full = self._partition_clip_ids(
                clip_id_to_remote_name,
                rank=self.rank,
                world_size=self.world_size,
                seed=config.clip_partition_shuffle_seed,
            )
            holdout_size = min(
                config.eval_holdout_size, max(len(rank_clip_ids_full) - 1, 0)
            )
            if holdout_size > 0:
                self.eval_holdout_clip_ids = rank_clip_ids_full[-holdout_size:]
                self.rank_clip_ids = rank_clip_ids_full[:-holdout_size]
            else:
                self.eval_holdout_clip_ids = []
                self.rank_clip_ids = rank_clip_ids_full

        if not self.rank_clip_ids:
            raise RuntimeError(
                f"No training clips assigned to rank {self.rank} "
                f"(world_size={self.world_size}) from {len(clip_id_to_remote_name)} clips."
            )
        if explicit_split and not self.eval_holdout_clip_ids:
            raise RuntimeError(
                f"No validation clips assigned to rank {self.rank} "
                f"(world_size={self.world_size}) from {len(validation_id_to_remote_name)} clips."
            )

        self._clip_remote_name: Dict[str, str] = {
            cid: clip_id_to_remote_name[cid] for cid in self.rank_clip_ids
        }
        self._eval_holdout_remote_name: Dict[str, str] = {
            cid: validation_id_to_remote_name[cid] for cid in self.eval_holdout_clip_ids
        }
        self.clip_id_to_local_index: Dict[str, int] = {
            cid: i for i, cid in enumerate(self.rank_clip_ids)
        }
        log.info(
            f"[GlobalClipPool] rank {self.rank}/{self.world_size}: "
            f"{len(self.rank_clip_ids)}/{len(clip_id_to_remote_name)} training clips assigned, "
            f"{len(self.eval_holdout_clip_ids)}/{len(validation_id_to_remote_name)} validation "
            f"clips assigned, resident_pool_size={config.resident_pool_size}, "
            f"split_version={config.split_version or 'legacy'}"
        )

        num_clips = len(self.rank_clip_ids)
        difficulty_scores = self._load_difficulty_scores(config.difficulty_scores_path)
        self.global_clip_weights = torch.tensor(
            [difficulty_scores.get(cid, 0.5) for cid in self.rank_clip_ids],
            dtype=torch.float32,
        )
        self.global_clip_visit_counts = torch.zeros(num_clips, dtype=torch.long)
        self.rebuild_count = 0

        self._resident_local_indices: Optional[torch.Tensor] = None
        self._resident_shape_counts: Optional[torch.Tensor] = None
        self._cache_last_used: Dict[str, int] = {}
        self._touch_count = 0

        # Stable for the life of the run: `Env.get_task_id()` derives the `env_<task_id>.ckpt`
        # checkpoint filename from this, and residency here depends on the checkpoint's own
        # contents (unlike `MotionLibPool`'s shard rotation, a pure function of `current_epoch`
        # computable before the checkpoint is even opened) -- if this string changed every
        # rebuild, resume could never find the right checkpoint file.
        self.motion_file = f"global_clip_pool_rank{self.rank}"

        # Synchronous: env/simulator construction needs real motion data immediately.
        self._rebuild_resident_pool(self._select_top_k(), force=True)

    # ---------------------------------------------------------------- manifest

    def _load_difficulty_scores(self, difficulty_scores_path: str) -> Dict[str, float]:
        """Parse a 'clip_id: score' file (one per line) into {clip_id: score}.

        Used to seed global_clip_weights with an easy->hard prior instead of a flat 1.0. Not
        precise by design -- see GlobalClipPoolConfig.difficulty_scores_path.
        """
        path = Path(difficulty_scores_path)
        scores: Dict[str, float] = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                clip_id, score = line.split(":", 1)
                scores[clip_id.strip()] = float(score.strip())
        return scores

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _partition_clip_ids(
        clip_id_to_remote_name: Dict[str, str],
        rank: int,
        world_size: int,
        seed: int,
    ) -> List[str]:
        """Partition one already-role-specific manifest without changing global membership."""
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(
                f"Invalid distributed partition rank={rank}, world_size={world_size}."
            )
        clip_ids = sorted(clip_id_to_remote_name)
        random.Random(seed).shuffle(clip_ids)
        return clip_ids[rank::world_size]

    @staticmethod
    def _normalize_remote_relative_path(relative_name: str) -> str:
        path = PurePosixPath(relative_name)
        if (
            not relative_name
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() == "."
        ):
            raise ValueError(
                f"Manifest paths must be non-empty, relative R2 paths without '..': "
                f"{relative_name!r}"
            )
        return path.as_posix()

    def _download_remote_file(self, r2_source: str, relative_name: str) -> Path:
        relative_name = self._normalize_remote_relative_path(relative_name)
        source_cache_key = hashlib.sha256(
            r2_source.rstrip("/").encode("utf-8")
        ).hexdigest()[:16]
        local_path = (
            self.local_cache_dir
            / "_manifest_dl"
            / source_cache_key
            / Path(*PurePosixPath(relative_name).parts)
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path = f"{r2_source.rstrip('/')}/{relative_name}"

        # Split artifacts are immutable and versioned. A source-specific local directory prevents
        # identically named manifests from different R2 prefixes sharing a stale cache entry.
        if not local_path.exists():
            try:
                subprocess.run(
                    [
                        "rclone",
                        "copyto",
                        remote_path,
                        str(local_path),
                        "--s3-no-check-bucket",
                        "--retries=10",
                        "--retries-sleep=30s",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    "`rclone` was not found on PATH -- required for GlobalClipPoolConfig."
                ) from e
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to download {remote_path} (rc={e.returncode}): {e.stderr}"
                ) from e

        if not local_path.exists():
            raise RuntimeError(f"Remote file not found after download: {local_path}")
        return local_path

    def _load_manifest(
        self, r2_source: str, manifest_name: str
    ) -> Tuple[Dict[str, str], str]:
        manifest_name = self._normalize_remote_relative_path(manifest_name)
        local_manifest = self._download_remote_file(r2_source, manifest_name)
        remote_manifest = f"{r2_source.rstrip('/')}/{manifest_name}"

        clip_id_to_remote_name: Dict[str, str] = {}
        with open(local_manifest, "r") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "clip_id" not in entry or "file" not in entry:
                    raise RuntimeError(
                        f"{remote_manifest}:{line_number} must contain 'clip_id' and 'file'."
                    )
                clip_id = str(entry["clip_id"])
                if clip_id in clip_id_to_remote_name:
                    raise RuntimeError(
                        f"Duplicate clip_id {clip_id!r} in manifest {remote_manifest}."
                    )
                clip_id_to_remote_name[clip_id] = str(entry["file"])

        if not clip_id_to_remote_name:
            raise RuntimeError(f"Manifest {remote_manifest} is empty.")
        return clip_id_to_remote_name, self._sha256_file(local_manifest)

    def _load_split_metadata(self, r2_source: str, metadata_name: str) -> dict:
        metadata_name = self._normalize_remote_relative_path(metadata_name)
        local_metadata = self._download_remote_file(r2_source, metadata_name)
        try:
            metadata = json.loads(local_metadata.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not parse split metadata {local_metadata}: {exc}") from exc
        if metadata.get("schema_version") != 1:
            raise RuntimeError(
                f"Unsupported split metadata schema_version={metadata.get('schema_version')!r}; "
                "expected 1."
            )
        return metadata

    def _split_role_remote_path(self, split_metadata: dict, role: str) -> str:
        try:
            role_file = split_metadata["manifests"][role]["file"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Split metadata is missing manifests.{role}.file.") from exc
        role_file = self._normalize_remote_relative_path(str(role_file))
        metadata_name = self._normalize_remote_relative_path(
            self.config.split_metadata_name
        )
        metadata_parent = PurePosixPath(metadata_name).parent
        return (metadata_parent / PurePosixPath(role_file)).as_posix()

    def _validate_split_manifest_roles(self, split_metadata: dict) -> None:
        expected_train = self._split_role_remote_path(split_metadata, "train")
        expected_validation = self._split_role_remote_path(split_metadata, "validation")
        configured_train = self._normalize_remote_relative_path(self.config.manifest_name)
        configured_validation = self._normalize_remote_relative_path(
            self.config.validation_manifest_name
        )
        if configured_train != expected_train:
            raise RuntimeError(
                f"Configured training manifest {configured_train!r} does not match the train "
                f"role bound by split metadata ({expected_train!r})."
            )
        if configured_validation != expected_validation:
            raise RuntimeError(
                f"Configured validation manifest {configured_validation!r} does not match the "
                f"validation role bound by split metadata ({expected_validation!r})."
            )
        # Test hash/count provenance is validated below. Its path is never resolved or downloaded;
        # GlobalClipPool intentionally has no test-manifest configuration or loader API.

    def _validate_and_record_split_provenance(
        self,
        split_metadata: dict,
        train_sha256: str,
        validation_sha256: str,
        train_count: int,
        validation_count: int,
    ) -> None:
        try:
            manifests = split_metadata["manifests"]
            train_spec = manifests["train"]
            validation_spec = manifests["validation"]
            test_spec = manifests["test"]
            source_spec = split_metadata["source_manifest"]
            split_version = str(split_metadata["split_version"])
            seed = int(split_metadata["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Malformed split metadata: {exc}") from exc

        checks = (
            ("train sha256", train_sha256, str(train_spec["sha256"])),
            (
                "validation sha256",
                validation_sha256,
                str(validation_spec["sha256"]),
            ),
            ("train clip count", train_count, int(train_spec["clip_count"])),
            (
                "validation clip count",
                validation_count,
                int(validation_spec["clip_count"]),
            ),
        )
        for label, actual, expected in checks:
            if actual != expected:
                raise RuntimeError(
                    f"Split metadata {label} mismatch: actual={actual!r}, expected={expected!r}."
                )

        split_identity = {
            "schema_version": int(split_metadata["schema_version"]),
            "split_version": split_version,
            "seed": seed,
            "source_manifest_sha256": str(source_spec["sha256"]),
            "manifest_sha256": {
                role: str(manifests[role]["sha256"])
                for role in ("train", "validation", "test")
            },
            "manifest_clip_count": {
                role: int(manifests[role]["clip_count"])
                for role in ("train", "validation", "test")
            },
        }
        computed_split_id = hashlib.sha256(
            json.dumps(
                split_identity, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        metadata_split_id = str(split_metadata.get("split_id", ""))
        if computed_split_id != metadata_split_id:
            raise RuntimeError(
                "split_metadata.json has an invalid split_id: "
                f"computed={computed_split_id}, recorded={metadata_split_id}."
            )

        self.config.split_version = split_version
        self.config.split_id = computed_split_id
        self.config.source_manifest_sha256 = str(source_spec["sha256"])
        self.config.train_manifest_sha256 = train_sha256
        self.config.validation_manifest_sha256 = validation_sha256
        self.config.test_manifest_sha256 = str(test_spec["sha256"])
        self.config.train_clip_count = train_count
        self.config.validation_clip_count = validation_count
        self.config.test_clip_count = int(test_spec["clip_count"])

    def split_provenance(self) -> dict:
        """Return the role hashes/counts saved with checkpoints and W&B config."""
        return {
            "split_version": self.config.split_version,
            "split_id": self.config.split_id,
            "source_manifest_sha256": self.config.source_manifest_sha256,
            "train_manifest_sha256": self.config.train_manifest_sha256,
            "validation_manifest_sha256": self.config.validation_manifest_sha256,
            "test_manifest_sha256": self.config.test_manifest_sha256,
            "train_clip_count": self.config.train_clip_count,
            "validation_clip_count": self.config.validation_clip_count,
            "test_clip_count": self.config.test_clip_count,
        }

    def _download_missing_clips(self, remote_names: List[str]) -> None:
        """Download all listed (not-yet-cached) per-clip files in ONE rclone invocation, using
        rclone's own bounded internal concurrency (`--transfers`) rather than one OS subprocess
        per file.

        The original implementation launched a separate `FileDownloader` (i.e. a separate
        `rclone copy` process) per missing clip, all concurrently. At K~256 clips/rank -- and
        worse, all 256 missing at once on the very first cold-start rebuild -- that's up to 256
        simultaneous OS processes per rank (more across multiple ranks on one node), each opening
        its own set of file descriptors (network sockets, config/credential files) on top of
        whatever else the training process has open. That exhausts the user's open-file ulimit
        (`EMFILE`, "Too many open files") in practice, unlike `MotionLibPool`'s shard prefetch
        which never runs more than 1-2 concurrent downloads. A single rclone process with
        `--files-from` keeps the OS-level process/FD footprint to one process regardless of K;
        parallelism across files still happens, just internally to that one process.
        """
        if not remote_names:
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(remote_names))
            list_path = f.name

        timeout = DOWNLOAD_BASE_TIMEOUT_SECONDS + DOWNLOAD_PER_FILE_TIMEOUT_SECONDS * len(
            remote_names
        )
        try:
            for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
                try:
                    subprocess.run(
                        [
                            "rclone",
                            "copy",
                            self.config.r2_source.rstrip("/"),
                            str(self.local_cache_dir),
                            f"--files-from={list_path}",
                            f"--transfers={self.config.download_transfers}",
                            f"--checkers={self.config.download_transfers}",
                            "--retries=10",
                            "--retries-sleep=30s",
                            "--s3-no-check-bucket",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    return
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    reason = (
                        f"timed out after {timeout:.0f}s"
                        if isinstance(e, subprocess.TimeoutExpired)
                        else str(e)
                    )
                    log.warning(
                        f"[GlobalClipPool] rank {self.rank}: batched download attempt "
                        f"{attempt}/{DOWNLOAD_MAX_ATTEMPTS} failed "
                        f"({len(remote_names)} files): {reason}"
                    )
                    if attempt == DOWNLOAD_MAX_ATTEMPTS:
                        raise RuntimeError(
                            f"Failed to download {len(remote_names)} clip files after "
                            f"{DOWNLOAD_MAX_ATTEMPTS} attempts: {reason}"
                        ) from e
                    time.sleep(DOWNLOAD_RETRY_BACKOFF_SECONDS)
        finally:
            Path(list_path).unlink(missing_ok=True)

    # ---------------------------------------------------------------- priority / selection

    def _select_top_k(self) -> torch.Tensor:
        """Combinatorial-UCB priority: known-hard clips OR still-unproven clips win a slot.

        `+1` smoothing on both the log term and the visit count avoids the divide-by-zero/
        log(0) edge case at the very first rebuild (`rebuild_count=0`, all `visit_counts=0`)
        while preserving the two properties the design doc requires: a never-visited clip has
        the highest possible bonus at any given rebuild_count, and that bonus keeps growing
        (slowly, log-scale) with `rebuild_count` even if the clip is never picked, so it's
        guaranteed to eventually outrank a stale already-known-hard clip.

        Selection itself samples proportional to priority *rank* (softmax over -rank/temperature)
        rather than taking the deterministic top-K. Deterministic top-K swaps a whole cluster of
        similarly-scored clips into (or out of) residency in lockstep the moment their shared
        score axis crosses the cutoff -- rank-based sampling still favors high-priority clips but
        avoids that correlated, all-at-once churn (cf. Prioritized Level Replay, Jiang et al. 2021).

        `config.random_fraction` reserves that fraction of the K slots for pure uniform-random
        selection (ignoring priority entirely), sampled BEFORE the priority pass and excluded from
        it. The slow log-scale exploration bonus above already guarantees eventual re-selection of
        a long-deprioritized clip, but only asymptotically -- a bounded random slice gives every
        clip a fixed, non-vanishing per-rebuild chance of forced rehearsal regardless of how low
        its weight has decayed. 0.0 (default) reproduces the old fully-priority-based behavior
        exactly (k_random=0).
        """
        device = self.global_clip_weights.device
        num_clips = self.global_clip_weights.numel()
        k = min(self.config.resident_pool_size, num_clips)
        k_random = min(round(self.config.random_fraction * k), k)
        k_priority = k - k_random

        if k_random > 0:
            random_idx = torch.randperm(num_clips, device=device)[:k_random]
        else:
            random_idx = torch.empty(0, dtype=torch.long, device=device)

        if k_priority == 0:
            return random_idx

        remaining_mask = torch.ones(num_clips, dtype=torch.bool, device=device)
        remaining_mask[random_idx] = False
        remaining_indices = torch.nonzero(remaining_mask).flatten()

        t = self.rebuild_count
        bonus = self.config.exploration_bonus_coefficient * torch.sqrt(
            2.0 * math.log(t + 1) / (self.global_clip_visit_counts[remaining_indices].float() + 1.0)
        )
        priority = self.global_clip_weights[remaining_indices] + bonus
        ranks = priority.argsort(descending=True).argsort()
        probs = torch.softmax(-ranks.float() / self.config.selection_temperature, dim=0)
        chosen_sub = torch.multinomial(probs, k_priority, replacement=False)
        priority_idx = remaining_indices[chosen_sub]

        return torch.cat([random_idx, priority_idx])

    # ---------------------------------------------------------------- rebuild

    def _rebuild_resident_pool(
        self, target_local_idx: torch.Tensor, force: bool = False
    ) -> bool:
        """Curriculum-driven rebuild: materializes `target_local_idx` as the resident set AND
        records it as a genuine rebuild decision (bumps `rebuild_count`/`visit_counts`). Use
        this for real "should the resident set change" decisions (init, `maybe_rebuild`) --
        NOT for reconstituting an already-decided state on resume (see `_materialize_resident_set`
        for that; double-counting a resume as a fresh decision would silently inflate the UCB
        clock and falsely mark resumed clips as newly-visited with no new eval evidence).
        """
        target_local_idx = target_local_idx.sort().values
        if (
            not force
            and self._resident_local_indices is not None
            and torch.equal(target_local_idx, self._resident_local_indices)
        ):
            return False

        self._materialize_resident_set(target_local_idx)
        self.rebuild_count += 1
        self.global_clip_visit_counts[target_local_idx] += 1
        return True

    def _load_clip_set(
        self,
        clip_ids: List[str],
        remote_name_map: Dict[str, str],
        motion_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Download (if needed), assemble, and load `clip_ids` as the live MotionLib data.

        Generic over the (clip_id -> remote_name) mapping so both the training resident set
        (`self._clip_remote_name`) and the fixed eval holdout (`self._eval_holdout_remote_name`)
        can share this one download/assemble/load path. Pure I/O + tensor-assembly -- does NOT
        touch `rebuild_count`/`visit_counts`/`_resident_local_indices`; callers decide what, if
        anything, to update around this (see `_materialize_resident_set` vs. `load_eval_holdout`).

        Returns the per-clip shape counts (needed by callers that track `_resident_shape_counts`).
        """
        # Download anything missing in ONE batched rclone call (bounded internal concurrency via
        # --transfers), not one OS subprocess per file. Unlike MotionLibPool's shard prefetch
        # (which downloads the *known* next shard during the *current* shard's many epochs of
        # residency), which clips get promoted here is only known once this rebuild's priority is
        # computed -- there's no way to know it in advance, so this download is unavoidably
        # synchronous from the training loop's perspective.
        missing_remote_names = [
            remote_name_map[clip_id]
            for clip_id in clip_ids
            if not (self.local_cache_dir / remote_name_map[clip_id]).exists()
        ]
        self._download_missing_clips(missing_remote_names)

        clip_dicts = []
        shape_counts = []
        for clip_id in clip_ids:
            local_path = self.local_cache_dir / remote_name_map[clip_id]
            clip_data = torch.load(local_path, map_location="cpu", weights_only=False)
            clip_dicts.append(clip_data)
            shape_counts.append(len(clip_data["motion_lengths"]))
            self._touch_count += 1
            self._cache_last_used[clip_id] = self._touch_count

        assembled = self._concat_clip_dicts(clip_dicts)

        # Overwrite the assembled per-motion weights with the caller-supplied per-clip value,
        # broadcast across its shape count -- the per-clip file's own `motion_weights` is just a
        # stale placeholder from packaging time.
        shape_counts_t = torch.tensor(shape_counts, dtype=torch.long)
        assembled["motion_weights"] = motion_weights.repeat_interleave(shape_counts_t)

        self._load_motion_state_dict(assembled)

        # `_load_motion_state_dict` only overwrites fields present in the loaded dict -- it never
        # touches these caches, so a stale mapping from whatever was previously loaded would
        # otherwise silently outlive this load (same reasoning as
        # `MotionLibPool._ensure_shard_loaded`).
        self._asset_id_to_motion_ids_cache = None
        self._clip_id_to_motion_ids_cache = None

        return shape_counts_t

    def _materialize_resident_set(self, target_local_idx: torch.Tensor) -> None:
        """Mechanically load `target_local_idx` as the resident MotionLib data. Pure I/O +
        tensor-assembly -- does NOT touch `rebuild_count`/`visit_counts` (see
        `_rebuild_resident_pool` for the curriculum-decision wrapper that does)."""
        target_local_idx = target_local_idx.sort().values
        target_clip_ids = [self.rank_clip_ids[i] for i in target_local_idx.tolist()]

        log.info(
            f"[GlobalClipPool] rank {self.rank}: rebuilding resident pool "
            f"({len(target_clip_ids)} clips, rebuild_count={self.rebuild_count})"
        )

        shape_counts_t = self._load_clip_set(
            target_clip_ids, self._clip_remote_name, self.global_clip_weights[target_local_idx]
        )

        self._resident_local_indices = target_local_idx.clone()
        self._resident_shape_counts = shape_counts_t

        self._evict_cache(keep_clip_ids=set(target_clip_ids))

    def has_eval_holdout(self) -> bool:
        return len(self.eval_holdout_clip_ids) > 0

    def load_eval_holdout(self) -> None:
        """Swap the live MotionLib data to the fixed, permanently-excluded eval holdout set.

        Does NOT touch `rebuild_count`/`visit_counts`/`_resident_local_indices` -- the training
        resident set's bookkeeping stays intact in memory while holdout data is loaded in its
        place, so `restore_training_resident_set` can reload it afterward. Callers must also
        trigger `env.motion_manager.on_motion_lib_reloaded()` since the underlying tensors change
        size/identity (see `MimicEvaluator._evaluate_holdout`).
        """
        log.info(
            f"[GlobalClipPool] rank {self.rank}: loading eval holdout "
            f"({len(self.eval_holdout_clip_ids)} clips)"
        )
        # Weight value is irrelevant here -- holdout clips are never selected via priority, this
        # just satisfies _load_clip_set's motion_weights broadcast contract.
        motion_weights = torch.ones(len(self.eval_holdout_clip_ids), dtype=torch.float32)
        self._load_clip_set(self.eval_holdout_clip_ids, self._eval_holdout_remote_name, motion_weights)

    def restore_training_resident_set(self) -> None:
        """Reload whatever was resident for training before `load_eval_holdout` was called."""
        assert self._resident_local_indices is not None, (
            "restore_training_resident_set called before any training resident set was loaded."
        )
        target_local_idx = self._resident_local_indices
        target_clip_ids = [self.rank_clip_ids[i] for i in target_local_idx.tolist()]
        self._load_clip_set(
            target_clip_ids, self._clip_remote_name, self.global_clip_weights[target_local_idx]
        )

    @staticmethod
    def _concat_clip_dicts(clip_dicts: List[dict]) -> dict:
        """Concatenate K per-clip packaged dicts into one, in the given order.

        Mirrors the field handling in `tools/repackage_stage2_per_clip.py`, just in reverse
        (merge instead of split). Fails loudly (rather than silently dropping or leaving stale
        data) if an optional field is present in some but not all of the K dicts -- that would
        indicate malformed/inconsistent per-clip data, since presence of these fields is a
        whole-dataset conversion-pipeline invariant, not a per-clip one.
        """
        assembled: dict = {}

        def gather(field_name: str):
            present = [cd.get(field_name) for cd in clip_dicts]
            n_present = sum(p is not None for p in present)
            if n_present == 0:
                return None
            if n_present != len(present):
                raise RuntimeError(
                    f"Inconsistent presence of '{field_name}' across resident clips "
                    f"({n_present}/{len(present)} have it) -- indicates malformed per-clip data."
                )
            return present

        for field_name in PER_FRAME_FIELDS:
            present = gather(field_name)
            if present is not None:
                assembled[field_name] = torch.cat(present, dim=0)

        motion_num_frames = torch.cat([cd["motion_num_frames"] for cd in clip_dicts], dim=0)
        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        assembled["length_starts"] = lengths_shifted.cumsum(0)
        assembled["motion_num_frames"] = motion_num_frames

        for field_name in PER_MOTION_TENSOR_FIELDS:
            if field_name == "motion_num_frames":
                continue
            present = gather(field_name)
            if present is not None:
                assembled[field_name] = torch.cat(present, dim=0)

        for field_name in PER_MOTION_TUPLE_FIELDS:
            present = gather(field_name)
            if present is not None:
                assembled[field_name] = tuple(x for tup in present for x in tup)

        return assembled

    def _evict_cache(self, keep_clip_ids: set) -> None:
        max_files = int(self.config.cache_size_multiplier * self.config.resident_pool_size)
        cached = {
            cid: name
            for cid, name in self._clip_remote_name.items()
            if (self.local_cache_dir / name).exists()
        }
        if len(cached) <= max_files:
            return

        evictable = [cid for cid in cached if cid not in keep_clip_ids]
        evictable.sort(key=lambda cid: self._cache_last_used.get(cid, -1))
        num_to_evict = len(cached) - max_files
        for cid in evictable[:num_to_evict]:
            (self.local_cache_dir / self._clip_remote_name[cid]).unlink(missing_ok=True)
            self._cache_last_used.pop(cid, None)

    def maybe_rebuild(self) -> bool:
        """Rebuild the resident pool if the current top-K differs from what's loaded.

        Called on `pool_rebuild_every`'s own epoch cadence (see
        `GlobalClipPoolRebuildCallback`), deliberately independent of the evaluator's
        `eval_metrics_every` -- see `GlobalClipPoolConfig.pool_rebuild_every`.
        """
        return self._rebuild_resident_pool(self._select_top_k())

    # ---------------------------------------------------------------- weight updates

    def update_global_clip_weights(
        self,
        failed_motion_ids: torch.Tensor,
        success_motion_ids: torch.Tensor,
    ) -> None:
        """Nudge the persistent scoreboard towards the evaluator's latest pass/fail signal.

        `failed_motion_ids`/`success_motion_ids` are global motion ids into the *currently
        resident* MotionLib, already expanded to all shape variants of each evaluated clip by the
        caller (`MimicEvaluator._expand_to_clip_variants`). Failure wins over success on overlap,
        matching the existing precedence in `MimicEvaluator._update_motion_sampling_weights`.

        Uses a bounded EMA (`weight = alpha*target + (1-alpha)*weight`, target=0/1) instead of a
        one-shot multiplicative jump -- a single eval tick can only move each clip's weight a
        little, not snap a whole cluster of similarly-scored clips across the selection cutoff
        at once (cf. Graves et al. 2017 on oscillatory curriculum-chasing from hard step updates).
        """
        clip_ids = self.motion_clip_ids
        alpha = self.config.weight_ema_alpha

        def to_local_indices(motion_ids: torch.Tensor) -> torch.Tensor:
            unique_clips = {clip_ids[i] for i in motion_ids.detach().cpu().tolist()}
            return torch.tensor(
                [self.clip_id_to_local_index[c] for c in unique_clips], dtype=torch.long
            )

        if success_motion_ids.numel() > 0:
            idx = to_local_indices(success_motion_ids)
            self.global_clip_weights[idx] = torch.clamp(
                (1 - alpha) * self.global_clip_weights[idx], min=self.config.weight_floor
            )

        if failed_motion_ids.numel() > 0:
            idx = to_local_indices(failed_motion_ids)
            self.global_clip_weights[idx] = (
                alpha + (1 - alpha) * self.global_clip_weights[idx]
            )

    def project_global_weights_to_resident_motion_weights(self) -> torch.Tensor:
        """Re-broadcast the current global weights onto the resident MotionLib's per-motion
        `motion_weights` shape, without waiting for the next rebuild."""
        weights = self.global_clip_weights[self._resident_local_indices].repeat_interleave(
            self._resident_shape_counts
        )
        return weights.to(self.device)

    # ---------------------------------------------------------------- checkpointing

    def get_global_clip_weights_state_dict(self) -> dict:
        return {
            "clip_ids": tuple(self.rank_clip_ids),
            "weights": self.global_clip_weights.clone(),
            "visit_counts": self.global_clip_visit_counts.clone(),
            "rebuild_count": self.rebuild_count,
            "split_provenance": self.split_provenance(),
        }

    def load_global_clip_weights_state_dict(self, state_dict: dict) -> None:
        checkpoint_provenance = state_dict.get("split_provenance")
        current_provenance = self.split_provenance()
        if checkpoint_provenance is not None and checkpoint_provenance != current_provenance:
            raise RuntimeError(
                "Checkpointed split provenance does not match the configured split. "
                f"checkpoint={checkpoint_provenance}, current={current_provenance}."
            )
        if self.config.split_id is not None and checkpoint_provenance is None:
            raise RuntimeError(
                "Checkpoint has no split provenance but the current run uses an explicit "
                "versioned split; refusing an unverifiable resume."
            )

        if tuple(state_dict["clip_ids"]) != tuple(self.rank_clip_ids):
            raise RuntimeError(
                "Checkpointed clip vocabulary doesn't match this rank's current partition -- "
                "did the manifest, world_size, or clip_partition_shuffle_seed change? Refusing "
                "to load global clip weights against a mismatched vocabulary."
            )
        self.global_clip_weights = state_dict["weights"].clone()
        self.global_clip_visit_counts = state_dict["visit_counts"].clone()
        self.rebuild_count = state_dict["rebuild_count"]
        # Reconstitutes a resident set consistent with the restored scoreboard -- not a new
        # curriculum decision, so this must NOT bump rebuild_count/visit_counts again (see
        # `_materialize_resident_set` vs. `_rebuild_resident_pool`). Note this isn't guaranteed to
        # be bit-identical to whatever was resident at save time: `_select_top_k()` uses
        # `rebuild_count` as UCB's time axis, and the restored value is one past whatever `t` the
        # save-time decision actually used (selection happens, then the counter increments) -- the
        # same kind of shift that naturally happens between any two consecutive live rebuilds
        # anyway. What must round-trip exactly is the scoreboard (weights/visit_counts/
        # rebuild_count) itself, not the derived resident set, which is always free to re-derive.
        self._materialize_resident_set(self._select_top_k())

    @property
    def distinct_motions_seen(self) -> int:
        """Total distinct clips visited by this rank so far (parity with `MotionLibPool`, read
        by `agent.py`'s logging for any motion_lib that exposes it)."""
        return int((self.global_clip_visit_counts > 0).sum().item())
