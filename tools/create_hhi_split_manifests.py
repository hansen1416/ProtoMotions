# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Create deterministic, clip-level train/validation/test manifests for HHI Stage 2.

The split is made over base clip files (one file contains every body-shape variant), so no
shape variant of a clip can cross split boundaries. Mirrored/unmirrored clip pairs are also kept
together. A caller-provided development set can be pinned to train before the two holdouts are
selected. Holdouts are sampled evenly over difficulty strata and have exact requested sizes.

Only the train and validation manifests are consumed by GlobalClipPool during training. The test
manifest is an offline evaluation artifact and GlobalClipPool intentionally has no test-manifest
configuration field or loading path.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SCHEMA_VERSION = 1
DEFAULT_SPLIT_VERSION = "hhi_stage2_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, salt: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{value}".encode("utf-8")).hexdigest()


def load_source_manifest(path: Path) -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    with open(path, "r") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "clip_id" not in entry or "file" not in entry:
                raise ValueError(
                    f"{path}:{line_number} must contain both 'clip_id' and 'file'."
                )
            clip_id = str(entry["clip_id"])
            if clip_id in entries:
                raise ValueError(f"Duplicate clip_id {clip_id!r} in {path}.")
            entries[clip_id] = entry
    if not entries:
        raise ValueError(f"Source manifest is empty: {path}")
    return entries


def load_difficulty_scores(path: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with open(path, "r") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                clip_id, score = line.split(":", 1)
                scores[clip_id.strip()] = float(score.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Malformed difficulty row at {path}:{line_number}: {line!r}"
                ) from exc
    return scores


def load_clip_ids(path: Path) -> List[str]:
    clip_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError(f"Duplicate clip ids in {path}.")
    return clip_ids


def mirrored_group_id(clip_id: str, all_clip_ids: set) -> str:
    """Return one stable group id for Mxxxxxx/xxxxxx when both clips exist."""
    if clip_id.startswith("M") and clip_id[1:] in all_clip_ids:
        return clip_id[1:]
    if not clip_id.startswith("M") and f"M{clip_id}" in all_clip_ids:
        return clip_id
    return clip_id


def build_groups(
    clip_ids: Iterable[str], difficulty_scores: Dict[str, float]
) -> List[Tuple[str, Tuple[str, ...], float]]:
    all_clip_ids = set(clip_ids)
    grouped: Dict[str, List[str]] = {}
    for clip_id in all_clip_ids:
        grouped.setdefault(mirrored_group_id(clip_id, all_clip_ids), []).append(clip_id)

    groups = []
    for group_id, members in grouped.items():
        members_tuple = tuple(sorted(members))
        mean_difficulty = sum(difficulty_scores[cid] for cid in members_tuple) / len(
            members_tuple
        )
        groups.append((group_id, members_tuple, mean_difficulty))
    return groups


def interleave_difficulty_strata(
    groups: Sequence[Tuple[str, Tuple[str, ...], float]],
    seed: int,
    salt: str,
    num_strata: int,
) -> List[Tuple[str, Tuple[str, ...], float]]:
    """Produce a seeded order that visits every difficulty band before revisiting one."""
    sorted_groups = sorted(groups, key=lambda group: (group[2], group[0]))
    strata: List[List[Tuple[str, Tuple[str, ...], float]]] = [
        [] for _ in range(min(num_strata, len(sorted_groups)))
    ]
    for index, group in enumerate(sorted_groups):
        stratum = min(index * len(strata) // len(sorted_groups), len(strata) - 1)
        strata[stratum].append(group)

    for stratum_index, stratum in enumerate(strata):
        stratum.sort(
            key=lambda group: stable_key(
                seed, f"{salt}:stratum-{stratum_index}", group[0]
            )
        )

    ordered = []
    max_length = max(len(stratum) for stratum in strata)
    for offset in range(max_length):
        # Rotate the stratum visitation order on each pass so boundary effects do not always
        # favor the easiest band.
        start = int(stable_key(seed, f"{salt}:pass", str(offset))[:8], 16) % len(strata)
        for delta in range(len(strata)):
            stratum = strata[(start + delta) % len(strata)]
            if offset < len(stratum):
                ordered.append(stratum[offset])
    return ordered


def select_holdouts(
    candidate_groups: Sequence[Tuple[str, Tuple[str, ...], float]],
    validation_count: int,
    test_count: int,
    seed: int,
    num_strata: int,
) -> Tuple[set, set]:
    """Select exact-size, difficulty-stratified validation/test sets by whole groups."""
    ordered = interleave_difficulty_strata(
        candidate_groups, seed=seed, salt="holdout", num_strata=num_strata
    )
    remaining = {"validation": validation_count, "test": test_count}
    selected = {"validation": set(), "test": set()}

    for group_id, members, _ in ordered:
        if remaining["validation"] == 0 and remaining["test"] == 0:
            break
        group_size = len(members)

        available = [name for name in ("validation", "test") if remaining[name] >= group_size]
        if not available:
            continue
        if len(available) == 1:
            destination = available[0]
        else:
            validation_need = remaining["validation"] / max(validation_count, 1)
            test_need = remaining["test"] / max(test_count, 1)
            if validation_need == test_need:
                destination = (
                    "validation"
                    if int(stable_key(seed, "holdout-role", group_id), 16) % 2 == 0
                    else "test"
                )
            else:
                destination = "validation" if validation_need > test_need else "test"

        selected[destination].update(members)
        remaining[destination] -= group_size

    if remaining["validation"] or remaining["test"]:
        raise RuntimeError(
            "Could not satisfy exact holdout sizes while preserving mirrored groups: "
            f"validation short by {remaining['validation']}, test short by {remaining['test']}."
        )
    return selected["validation"], selected["test"]


def write_manifest(path: Path, entries: Dict[str, dict], clip_ids: Iterable[str]) -> None:
    with open(path, "w") as stream:
        for clip_id in sorted(clip_ids):
            stream.write(json.dumps(entries[clip_id], sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def write_ids(path: Path, clip_ids: Iterable[str]) -> None:
    path.write_text("".join(f"{clip_id}\n" for clip_id in sorted(clip_ids)))


def create_split_manifests(
    source_manifest: Path,
    difficulty_file: Path,
    force_train_clip_ids: Path,
    output_dir: Path,
    split_version: str = DEFAULT_SPLIT_VERSION,
    seed: int = 42,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    num_difficulty_strata: int = 100,
) -> dict:
    fractions = train_fraction + validation_fraction + test_fraction
    if abs(fractions - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}.")
    if min(train_fraction, validation_fraction, test_fraction) < 0:
        raise ValueError("Split fractions must be non-negative.")
    if num_difficulty_strata <= 0:
        raise ValueError("num_difficulty_strata must be positive.")

    entries = load_source_manifest(source_manifest)
    difficulty_scores = load_difficulty_scores(difficulty_file)
    clip_ids = set(entries)
    missing_difficulty = clip_ids - set(difficulty_scores)
    if missing_difficulty:
        sample = ", ".join(sorted(missing_difficulty)[:5])
        raise ValueError(
            f"Difficulty file is missing {len(missing_difficulty)} source clips, "
            f"including {sample}."
        )

    requested_force_train = set(load_clip_ids(force_train_clip_ids))
    missing_forced = requested_force_train - clip_ids
    if missing_forced:
        sample = ", ".join(sorted(missing_forced)[:5])
        raise ValueError(
            f"Forced-train list contains {len(missing_forced)} clips absent from the source "
            f"manifest, including {sample}."
        )

    groups = build_groups(clip_ids, difficulty_scores)
    forced_group_ids = {
        mirrored_group_id(clip_id, clip_ids) for clip_id in requested_force_train
    }
    forced_train = {
        clip_id
        for group_id, members, _ in groups
        if group_id in forced_group_ids
        for clip_id in members
    }
    candidates = [group for group in groups if group[0] not in forced_group_ids]

    total_count = len(clip_ids)
    validation_count = round(total_count * validation_fraction)
    test_count = round(total_count * test_fraction)
    train_count = total_count - validation_count - test_count
    if len(forced_train) > train_count:
        raise ValueError(
            f"Forced-train groups contain {len(forced_train)} clips, exceeding the target "
            f"training size of {train_count}."
        )

    validation_ids, test_ids = select_holdouts(
        candidates,
        validation_count=validation_count,
        test_count=test_count,
        seed=seed,
        num_strata=num_difficulty_strata,
    )
    train_ids = clip_ids - validation_ids - test_ids

    assert len(train_ids) == train_count
    assert len(validation_ids) == validation_count
    assert len(test_ids) == test_count
    assert forced_train <= train_ids
    assert not (train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids)
    for _, members, _ in groups:
        destinations = sum(
            bool(set(members) & split_ids)
            for split_ids in (train_ids, validation_ids, test_ids)
        )
        assert destinations == 1, f"Grouped clips crossed split boundaries: {members}"

    output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }
    manifest_metadata = {}
    id_list_metadata = {}
    for split_name, ids in split_ids.items():
        manifest_path = output_dir / f"{split_name}_manifest.jsonl"
        ids_path = output_dir / f"{split_name}_ids.txt"
        write_manifest(manifest_path, entries, ids)
        write_ids(ids_path, ids)
        manifest_metadata[split_name] = {
            "file": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "clip_count": len(ids),
        }
        id_list_metadata[split_name] = {
            "file": ids_path.name,
            "sha256": sha256_file(ids_path),
            "clip_count": len(ids),
        }

    split_identity = {
        "schema_version": SCHEMA_VERSION,
        "split_version": split_version,
        "seed": seed,
        "source_manifest_sha256": sha256_file(source_manifest),
        "manifest_sha256": {
            role: manifest_metadata[role]["sha256"]
            for role in ("train", "validation", "test")
        },
        "manifest_clip_count": {
            role: manifest_metadata[role]["clip_count"]
            for role in ("train", "validation", "test")
        },
    }
    split_id = hashlib.sha256(
        json.dumps(split_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "split_version": split_version,
        "split_id": split_id,
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "source_manifest": {
            "sha256": sha256_file(source_manifest),
            "clip_count": total_count,
        },
        "difficulty": {
            "sha256": sha256_file(difficulty_file),
            "num_strata": min(num_difficulty_strata, len(groups)),
        },
        "forced_train": {
            "requested_ids_sha256": sha256_file(force_train_clip_ids),
            "requested_clip_count": len(requested_force_train),
            "effective_clip_count_with_mirrored_partners": len(forced_train),
        },
        "grouping": {"mirrored_pair_rule": "M<clip_id> stays with <clip_id> when both exist"},
        "manifests": manifest_metadata,
        "id_lists": id_list_metadata,
    }
    metadata_path = output_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--difficulty-file",
        type=Path,
        default=Path("data/preprocessing/valid_ids_sorted_by_difficulty.txt"),
    )
    parser.add_argument(
        "--force-train-clip-ids",
        type=Path,
        default=Path("data_cache/small150_128shape.clip_ids.txt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/splits/hhi_stage2_v1")
    )
    parser.add_argument("--split-version", default=DEFAULT_SPLIT_VERSION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--num-difficulty-strata", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = create_split_manifests(
        source_manifest=args.source_manifest,
        difficulty_file=args.difficulty_file,
        force_train_clip_ids=args.force_train_clip_ids,
        output_dir=args.output_dir,
        split_version=args.split_version,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        num_difficulty_strata=args.num_difficulty_strata,
    )
    print(f"Created split {metadata['split_version']} ({metadata['split_id']})")
    for role in ("train", "validation", "test"):
        spec = metadata["manifests"][role]
        print(f"  {role}: {spec['clip_count']} clips, sha256={spec['sha256']}")


if __name__ == "__main__":
    main()
