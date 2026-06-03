# scripts/export_humos_to_amass_npz.py

from pathlib import Path
import argparse
import re
import torch
import numpy as np
import yaml
from tqdm import tqdm


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s))


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def make_amass_like_pose(item):
    root_orient = to_np(item["root_orient"]).astype(np.float32)  # [T, 3]
    pose_body = to_np(item["pose_body"]).astype(np.float32)  # [T, 23, 3] or [T, 69]
    trans = to_np(item["trans"]).astype(np.float32)  # [T, 3]

    T = root_orient.shape[0]

    if pose_body.ndim == 3:
        pose_body = pose_body.reshape(T, -1)

    # ProtoMotions SMPL path takes:
    # root_orient [3] + first 21 SMPL body joints [63] = 66 dims,
    # then pads the remaining 2 joints internally.
    pose_body_21 = pose_body[:, :63]
    poses = np.concatenate([root_orient, pose_body_21], axis=1).astype(np.float32)

    return poses, trans, T


def collect_variants(data, genders, num_per_clip):
    """Return list of (gender, beta_key) from a loaded .pt dict."""
    all_beta_keys = sorted(
        set().union(*[set(data[g].keys()) for g in genders if g in data]),
        key=natural_key,
    )

    variants = []
    # Interleaved: beta0-male, beta0-female, beta1-male, beta1-female, ...
    for beta_key in all_beta_keys:
        for gender in genders:
            if gender not in data or beta_key not in data[gender]:
                continue
            variants.append((gender, beta_key))
            if num_per_clip > 0 and len(variants) >= num_per_clip:
                break
        if num_per_clip > 0 and len(variants) >= num_per_clip:
            break

    return variants


def process_single_file(
    src,
    seq_dir,
    args,
    global_idx,
    yaml_motions,
    manifest,
):
    """Process one HUMOS .pt file, appending entries to yaml_motions and manifest."""
    data = torch.load(src, map_location="cpu")

    variants = collect_variants(data, args.genders, args.num_per_clip)

    if not variants:
        tqdm.write(f"[WARN] no variants found in {src.name}, skipping")
        return global_idx

    for local_idx, (gender, beta_key) in enumerate(variants):
        item = data[gender][beta_key]

        poses, trans, T = make_amass_like_pose(item)

        if args.apply_offset_height and "offset_height" in item:
            trans[:, 2] += float(to_np(item["offset_height"]))

        betas = to_np(item["betas"]).astype(np.float32)
        if betas.ndim == 2:
            betas = betas[0]

        out_name = (
            f"{safe_name(src.stem)}_v{local_idx:02d}_{gender}_{safe_name(beta_key)}"
        )
        npz_path = seq_dir / f"{out_name}.npz"

        if not (args.skip_existing and npz_path.exists()):
            np.savez(
                npz_path,
                poses=poses,
                trans=trans,
                betas=betas,
                gender=np.array(gender),
                clip_id=np.array(src.stem),  # clip id from humos, e.g. 000005
                mocap_framerate=np.array(args.fps, dtype=np.float32),
            )

        duration = T / args.fps
        motion_rel = f"HUMOS/{out_name}.motion"

        yaml_motions.append(
            {
                "file": motion_rel,
                "fps": float(args.fps),
                "weight": 1.0,
                "sub_motions": [
                    {"timings": {"start": 0.0, "end": float(duration)}}
                ],
            }
        )

        manifest.append(
            {
                "index": global_idx,
                "clip_id": src.stem,
                "gender": gender,
                "beta_key": str(beta_key),
                "npz": str(npz_path),
                "motion": motion_rel,
                "duration": float(duration),
                "betas": betas.tolist(),
            }
        )

        global_idx += 1

    return global_idx


def main():
    parser = argparse.ArgumentParser(
        description="Convert HUMOS .pt files to AMASS-style .npz files."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Single HUMOS .pt file (single-clip mode).",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of HUMOS .pt files (batch mode, processes all *.pt).",
    )

    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--num-per-clip",
        "--num",  # legacy alias
        dest="num_per_clip",
        type=int,
        default=0,
        help="Max variants (gender×beta) to take per clip. 0 = all (default).",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default=None,
        help="Override output YAML filename. Default: humos_{N}.yaml where N=total variants.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--genders", nargs="+", default=["male", "female"])
    parser.add_argument("--apply-offset-height", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip .npz files already on disk (safe to re-run after interruption).",
    )

    args = parser.parse_args()

    # Collect input files
    if args.input_dir is not None:
        input_files = sorted(
            args.input_dir.glob("*.pt"), key=lambda p: natural_key(p.stem)
        )
        if not input_files:
            raise RuntimeError(f"No .pt files found in {args.input_dir}")
    else:
        input_files = [args.input]

    out_root = Path(args.out_root)
    seq_dir = out_root / "HUMOS"
    seq_dir.mkdir(parents=True, exist_ok=True)

    yaml_motions = []
    manifest = []
    global_idx = 0

    for src in tqdm(input_files, desc="Exporting", unit="clip"):
        global_idx = process_single_file(
            src=src,
            seq_dir=seq_dir,
            args=args,
            global_idx=global_idx,
            yaml_motions=yaml_motions,
            manifest=manifest,
        )

    total = global_idx
    yaml_name = args.yaml_name if args.yaml_name else f"humos_{total}.yaml"

    yaml_path = out_root / yaml_name
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"motions": yaml_motions}, f, sort_keys=False)

    manifest_name = yaml_name.replace(".yaml", "_manifest.yaml")
    manifest_path = out_root / manifest_name
    with open(manifest_path, "w") as f:
        yaml.safe_dump({"variants": manifest}, f, sort_keys=False)

    print(f"\nSaved YAML:        {yaml_path}")
    print(f"Saved manifest:    {manifest_path}")
    print(f"Total exported:    {total} variants from {len(input_files)} clip(s)")


if __name__ == "__main__":
    main()
