# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch

from protomotions.components.global_clip_pool import (
    GlobalClipPool,
    GlobalClipPoolConfig,
)
from tools.create_hhi_split_manifests import create_split_manifests, sha256_file


def _write_synthetic_inputs(root: Path):
    clip_ids = [f"clip_{index:02d}" for index in range(36)]
    clip_ids += [f"Mclip_{index:02d}" for index in range(4)]

    source_manifest = root / "clip_manifest.jsonl"
    with open(source_manifest, "w") as stream:
        for index, clip_id in enumerate(reversed(clip_ids)):
            stream.write(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "file": f"{clip_id}.pt",
                        "source_shard": f"batch_{index // 10:02d}.pt",
                    }
                )
                + "\n"
            )

    difficulty_file = root / "difficulty.txt"
    difficulty_file.write_text(
        "".join(f"{clip_id}: {index / 39:.8f}\n" for index, clip_id in enumerate(clip_ids))
    )
    forced_train = root / "forced_train.txt"
    forced_train.write_text("clip_00\nclip_20\n")
    return source_manifest, difficulty_file, forced_train


def _load_ids(path: Path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def test_split_generator_is_deterministic_exact_and_group_safe(tmp_path):
    source_manifest, difficulty_file, forced_train = _write_synthetic_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    kwargs = dict(
        source_manifest=source_manifest,
        difficulty_file=difficulty_file,
        force_train_clip_ids=forced_train,
        split_version="test_v1",
        seed=42,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
        num_difficulty_strata=8,
    )
    metadata = create_split_manifests(output_dir=first, **kwargs)
    repeated = create_split_manifests(output_dir=second, **kwargs)

    assert metadata == repeated
    for filename in (
        "train_manifest.jsonl",
        "validation_manifest.jsonl",
        "test_manifest.jsonl",
        "train_ids.txt",
        "validation_ids.txt",
        "test_ids.txt",
        "split_metadata.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    split_ids = {
        role: _load_ids(first / f"{role}_ids.txt")
        for role in ("train", "validation", "test")
    }
    assert {role: len(ids) for role, ids in split_ids.items()} == {
        "train": 32,
        "validation": 4,
        "test": 4,
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert {"clip_00", "Mclip_00", "clip_20"} <= split_ids["train"]
    for index in range(4):
        base = f"clip_{index:02d}"
        mirror = f"M{base}"
        assert any({base, mirror} <= ids for ids in split_ids.values())

    assert metadata["split_id"]
    for role in ("train", "validation", "test"):
        spec = metadata["manifests"][role]
        assert spec["sha256"] == sha256_file(first / spec["file"])


def test_explicit_manifest_partition_changes_rank_assignment_not_membership():
    manifest = {f"clip_{index:03d}": f"clip_{index:03d}.pt" for index in range(101)}

    for world_size in (1, 2, 3, 8):
        partitions = [
            GlobalClipPool._partition_clip_ids(manifest, rank, world_size, seed=42)
            for rank in range(world_size)
        ]
        assert set().union(*map(set, partitions)) == set(manifest)
        assert sum(len(partition) for partition in partitions) == len(manifest)
        for left in range(world_size):
            for right in range(left + 1, world_size):
                assert set(partitions[left]).isdisjoint(partitions[right])


def test_resume_restores_global_scoreboard_on_cpu():
    pool = object.__new__(GlobalClipPool)
    pool.config = GlobalClipPoolConfig()
    pool.rank_clip_ids = ["clip_a", "clip_b"]
    pool.rebuild_count = 0
    pool._select_top_k = lambda: torch.tensor([0, 1])
    pool._materialize_resident_set = lambda _: None

    pool.load_global_clip_weights_state_dict(
        {
            "clip_ids": ("clip_a", "clip_b"),
            "weights": torch.tensor([0.25, 0.75]),
            "visit_counts": torch.tensor([2, 3]),
            "rebuild_count": 4,
        }
    )

    assert pool.global_clip_weights.device.type == "cpu"
    assert pool.global_clip_visit_counts.device.type == "cpu"
    assert torch.equal(pool.global_clip_weights, torch.tensor([0.25, 0.75]))
    assert torch.equal(pool.global_clip_visit_counts, torch.tensor([2, 3]))


def test_split_metadata_binds_roles_and_records_test_without_loading_it(tmp_path):
    source_manifest, difficulty_file, forced_train = _write_synthetic_inputs(tmp_path)
    split_dir = tmp_path / "splits" / "test_v1"
    metadata = create_split_manifests(
        source_manifest=source_manifest,
        difficulty_file=difficulty_file,
        force_train_clip_ids=forced_train,
        output_dir=split_dir,
        split_version="test_v1",
        seed=42,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
        num_difficulty_strata=8,
    )

    config = GlobalClipPoolConfig(
        manifest_name="splits/test_v1/train_manifest.jsonl",
        validation_manifest_name="splits/test_v1/validation_manifest.jsonl",
        split_metadata_name="splits/test_v1/split_metadata.json",
    )
    assert not hasattr(config, "test_manifest_name")
    pool = object.__new__(GlobalClipPool)
    pool.config = config
    pool._validate_split_manifest_roles(metadata)
    pool._validate_and_record_split_provenance(
        metadata,
        train_sha256=sha256_file(split_dir / "train_manifest.jsonl"),
        validation_sha256=sha256_file(split_dir / "validation_manifest.jsonl"),
        train_count=32,
        validation_count=4,
    )
    assert pool.split_provenance()["split_id"] == metadata["split_id"]
    assert pool.split_provenance()["test_manifest_sha256"] == metadata["manifests"]["test"][
        "sha256"
    ]
    wrong_checkpoint_provenance = {**pool.split_provenance(), "split_id": "wrong"}
    with pytest.raises(RuntimeError, match="Checkpointed split provenance"):
        pool.load_global_clip_weights_state_dict(
            {"split_provenance": wrong_checkpoint_provenance}
        )

    config.manifest_name = "splits/test_v1/test_manifest.jsonl"
    with pytest.raises(RuntimeError, match="does not match the train role"):
        pool._validate_split_manifest_roles(metadata)

    with pytest.raises(RuntimeError, match="train sha256 mismatch"):
        pool._validate_and_record_split_provenance(
            metadata,
            train_sha256="wrong",
            validation_sha256=sha256_file(split_dir / "validation_manifest.jsonl"),
            train_count=32,
            validation_count=4,
        )


def test_load_eval_holdout_accepts_validated_subset():
    pool = object.__new__(GlobalClipPool)
    pool.rank = 0
    pool.eval_holdout_clip_ids = ["clip_a", "clip_b", "clip_c"]
    pool._eval_holdout_remote_name = {
        "clip_a": "clip_a.pt",
        "clip_b": "clip_b.pt",
        "clip_c": "clip_c.pt",
    }
    observed = []

    def load_clip_set(clip_ids, remote_name_map, motion_weights):
        observed.append((list(clip_ids), remote_name_map, motion_weights.clone()))

    pool._load_clip_set = load_clip_set
    pool.load_eval_holdout(["clip_c", "clip_a"])

    clip_ids, remote_name_map, weights = observed.pop()
    assert clip_ids == ["clip_c", "clip_a"]
    assert remote_name_map is pool._eval_holdout_remote_name
    assert torch.equal(weights, torch.ones(2))

    pool.load_eval_holdout()
    assert observed.pop()[0] == pool.eval_holdout_clip_ids

    with pytest.raises(ValueError, match="empty"):
        pool.load_eval_holdout([])
    with pytest.raises(ValueError, match="duplicate"):
        pool.load_eval_holdout(["clip_a", "clip_a"])
    with pytest.raises(ValueError, match="unknown"):
        pool.load_eval_holdout(["clip_missing"])
