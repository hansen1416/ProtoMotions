#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


GENDER_TO_ID = {
    "female": -1,
    "neutral": 0,
    "male": 1,
}


def read_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_motion_path(amass_root_dir: Path, motion_file: str) -> Path:
    motion_path = Path(motion_file)
    if motion_path.is_absolute():
        return motion_path
    return amass_root_dir / motion_path


def motion_path_to_npz_path(motion_path: Path) -> Path:
    if motion_path.suffix != ".motion":
        raise ValueError(f"Expected .motion file path, got: {motion_path}")
    return motion_path.with_suffix(".npz")


def read_npz_scalar_string(x) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        return str(x.tolist())
    return str(x)


def infer_beta_key_from_stem(stem: str, gender: str) -> str:
    # Expected HUMOS name: {source_stem}_v00_{gender}_{beta_key}
    marker = f"_{gender}_"
    if marker not in stem:
        raise ValueError(
            f"Cannot infer beta_key from filename stem='{stem}' with gender='{gender}'"
        )
    return stem.rsplit(marker, 1)[1]


def load_morphology_metadata_for_config(
    amass_root_dir: Path,
    motion_config: Path,
):
    config = read_yaml(motion_config)

    motions = config.get("motions", [])
    if len(motions) == 0:
        raise RuntimeError(f"No motions found in config: {motion_config}")

    motion_betas = []
    motion_gender_ids = []
    motion_genders = []
    motion_beta_keys = []
    motion_asset_ids = []
    motion_npz_files = []
    motion_clip_ids = []

    for motion in motions:
        motion_file = motion["file"]
        motion_path = resolve_motion_path(amass_root_dir, motion_file)
        npz_path = motion_path_to_npz_path(motion_path)

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Cannot find source .npz for motion:\n"
                f"  motion: {motion_path}\n"
                f"  npz:    {npz_path}"
            )

        data = np.load(npz_path, allow_pickle=True)

        if "betas" not in data:
            raise KeyError(f"'betas' not found in {npz_path}")

        if "gender" not in data:
            raise KeyError(f"'gender' not found in {npz_path}")

        betas = np.asarray(data["betas"], dtype=np.float32)
        if betas.ndim == 2:
            betas = betas[0]
        if betas.shape[0] != 10:
            raise ValueError(f"Expected betas shape [10], got {betas.shape} in {npz_path}")

        gender = read_npz_scalar_string(data["gender"])
        if gender not in GENDER_TO_ID:
            raise ValueError(f"Unknown gender '{gender}' in {npz_path}")

        beta_key = infer_beta_key_from_stem(npz_path.stem, gender)
        asset_id = f"{gender}_{beta_key}"

        if "clip_id" in data:
            clip_id = read_npz_scalar_string(data["clip_id"])
        else:
            clip_id = infer_clip_id_from_stem(npz_path.stem, gender)

        motion_betas.append(betas)
        motion_gender_ids.append(GENDER_TO_ID[gender])
        motion_genders.append(gender)
        motion_beta_keys.append(beta_key)
        motion_asset_ids.append(asset_id)
        motion_clip_ids.append(clip_id)
        motion_npz_files.append(str(npz_path))

    return {
        "motion_betas": torch.tensor(np.stack(motion_betas), dtype=torch.float32),
        "motion_gender_ids": torch.tensor(motion_gender_ids, dtype=torch.long),
        "motion_genders": tuple(motion_genders),
        "motion_beta_keys": tuple(motion_beta_keys),
        "motion_asset_ids": tuple(motion_asset_ids),
        "motion_clip_ids": tuple(motion_clip_ids),
        "motion_npz_files": tuple(motion_npz_files),
    }


def inject_morphology_metadata(output_file: Path, metadata: dict):
    motionlib = torch.load(output_file, map_location="cpu", weights_only=False)

    num_motions = len(motionlib["motion_lengths"])
    if metadata["motion_betas"].shape[0] != num_motions:
        raise ValueError(
            f"Metadata/motion count mismatch for {output_file}: "
            f"metadata={metadata['motion_betas'].shape[0]}, motionlib={num_motions}"
        )

    motionlib.update(metadata)
    torch.save(motionlib, output_file)


def infer_clip_id_from_stem(stem: str, gender: str) -> str:
    """
    Expected HUMOS variant name: {clip_id}_v00_{gender}_{beta_key}
    Example: 000005_v00_male_0e26b88d -> 000005
    """
    marker = f"_{gender}_"
    if marker not in stem:
        raise ValueError(
            f"Cannot infer clip_id from filename stem='{stem}' with gender='{gender}'"
        )
    prefix = stem.rsplit(marker, 1)[0]  # 000005_v00
    return prefix.rsplit("_v", 1)[0]    # 000005


def package_one_chunk(
    *,
    chunk_config: dict,
    amass_root_dir: Path,
    output_file: Path,
    motionlib_script: Path,
    device: str,
    project_root: Path,
    env: dict,
):
    """Package a single chunk config dict into a MotionLib .pt with morphology."""

    abs_config = {"motions": []}
    for motion in chunk_config.get("motions", []):
        m = dict(motion)
        m["file"] = str(resolve_motion_path(amass_root_dir, m["file"]))
        abs_config["motions"].append(m)

    tmp_yaml = output_file.with_suffix(".tmp.yaml")
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(abs_config, f, sort_keys=False)

    package_cmd = [
        sys.executable,
        str(motionlib_script),
        "--motion-path",
        str(tmp_yaml),
        "--output-file",
        str(output_file),
        "--device",
        device,
    ]

    result = subprocess.run(
        package_cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tmp_yaml.unlink(missing_ok=True)

    if result.returncode != 0:
        tqdm.write(result.stdout.decode())
        sys.exit(result.returncode)

    meta_yaml = output_file.with_suffix(".meta.yaml")
    original_config = {"motions": list(chunk_config.get("motions", []))}
    with open(meta_yaml, "w") as f:
        yaml.safe_dump(original_config, f, sort_keys=False)

    metadata = load_morphology_metadata_for_config(
        amass_root_dir=amass_root_dir,
        motion_config=meta_yaml,
    )
    meta_yaml.unlink(missing_ok=True)

    inject_morphology_metadata(output_file=output_file, metadata=metadata)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert AMASS/HUMOS .npz to MotionLib .pt with morphology metadata."
    )

    parser.add_argument(
        "amass_root_dir",
        type=Path,
        help="Root directory containing AMASS/HUMOS .npz and generated .motion files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for packaged .pt MotionLib chunk files.",
    )
    parser.add_argument(
        "--motion-config",
        type=Path,
        action="append",
        dest="motion_configs",
        required=True,
        help="YAML motion config. Can be specified multiple times.",
    )
    parser.add_argument(
        "--humanoid-type",
        type=str,
        default="smpl",
        choices=["smpl", "smplx"],
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--force-remake",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help=(
            "Number of motions per MotionLib packaging call. "
            "Increase for smaller datasets, decrease if packaging OOMs. "
            "Default: 4096."
        ),
    )

    args = parser.parse_args()

    if not args.amass_root_dir.exists():
        raise FileNotFoundError(f"AMASS root does not exist: {args.amass_root_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for config in args.motion_configs:
        if not config.exists():
            raise FileNotFoundError(f"Motion config does not exist: {config}")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Convert AMASS/HUMOS .npz files to .motion files
    # ──────────────────────────────────────────────────────────────────────────
    convert_script = project_root / "data" / "scripts" / "convert_amass_to_proto.py"

    convert_cmd = [
        sys.executable,
        str(convert_script),
        str(args.amass_root_dir),
        "--humanoid-type",
        args.humanoid_type,
        "--output-fps",
        str(args.output_fps),
    ]

    if args.force_remake:
        convert_cmd.append("--force-remake")

    for config in args.motion_configs:
        convert_cmd.extend(["--motion-config", str(config)])

    print("=" * 80)
    print("Step 1: Convert AMASS/HUMOS .npz to .motion")
    print("=" * 80)

    result = subprocess.run(convert_cmd, cwd=project_root, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Package each motion config into chunk .pt files with morphology
    # ──────────────────────────────────────────────────────────────────────────
    motionlib_script = project_root / "protomotions" / "components" / "motion_lib.py"

    print("\n" + "=" * 80)
    print("Step 2: Package MotionLib chunks and inject morphology metadata")
    print("=" * 80)

    for config_path in args.motion_configs:
        config_name = config_path.stem
        config = read_yaml(config_path)
        total_motions = len(config.get("motions", []))

        effective_batch = args.batch_size if args.batch_size > 0 else total_motions
        chunks = list(split_motion_config(config, effective_batch))
        n_chunks = len(chunks)

        print(f"\n{config_path.name}: {total_motions} motions → {n_chunks} chunk(s) of ≤{effective_batch}")

        with tqdm(total=n_chunks, desc="Packaging chunks", unit="chunk") as pbar:
            for chunk_idx, chunk_config in enumerate(chunks):
                chunk_file = args.output_dir / f"{config_name}_{chunk_idx:04d}.pt"

                package_one_chunk(
                    chunk_config=chunk_config,
                    amass_root_dir=args.amass_root_dir,
                    output_file=chunk_file,
                    motionlib_script=motionlib_script,
                    device=args.device,
                    project_root=project_root,
                    env=env,
                )
                pbar.update(1)

    print(f"\nDone. Output: {args.output_dir}")


def split_motion_config(config: dict, batch_size: int):
    """Yield successive chunk dicts of at most batch_size motions."""
    motions = config.get("motions", [])
    for i in range(0, len(motions), batch_size):
        yield {"motions": motions[i : i + batch_size]}


if __name__ == "__main__":
    main()
