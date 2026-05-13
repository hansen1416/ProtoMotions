# scripts/export_humos_variants_to_amass_npz.py

from pathlib import Path
import argparse
import re
import torch
import numpy as np
import yaml


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--genders", nargs="+", default=["male", "female"])
    parser.add_argument("--apply-offset-height", action="store_true")
    
    args = parser.parse_args()

    yaml_name = f"humos_{args.num}.yaml"

    src = Path(args.input)
    out_root = Path(args.out_root)
    seq_dir = out_root / "HUMOS"
    seq_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(src, map_location="cpu")

    variants = []

    # Interleaved selection:
    # beta0 male, beta0 female, beta1 male, beta1 female, ...
    all_beta_keys = sorted(
        set().union(*[set(data[g].keys()) for g in args.genders if g in data]),
        key=natural_key,
    )

    for beta_key in all_beta_keys:
        for gender in args.genders:
            if gender not in data:
                continue
            if beta_key not in data[gender]:
                continue
            variants.append((gender, beta_key))
            if len(variants) >= args.num:
                break
        if len(variants) >= args.num:
            break

    if len(variants) == 0:
        raise RuntimeError("No valid HUMOS variants found.")

    yaml_motions = []
    manifest = []

    for idx, (gender, beta_key) in enumerate(variants):
        item = data[gender][beta_key]

        poses, trans, T = make_amass_like_pose(item)

        if args.apply_offset_height and "offset_height" in item:
            trans[:, 2] += float(to_np(item["offset_height"]))

        betas = to_np(item["betas"]).astype(np.float32)
        if betas.ndim == 2:
            betas = betas[0]

        out_name = f"{safe_name(src.stem)}_v{idx:02d}_{gender}_{safe_name(beta_key)}"
        npz_path = seq_dir / f"{out_name}.npz"

        np.savez(
            npz_path,
            poses=poses,
            trans=trans,
            betas=betas,
            gender=np.array(gender),
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
                    {
                        "timings": {
                            "start": 0.0,
                            "end": float(duration),
                        }
                    }
                ],
            }
        )

        manifest.append(
            {
                "index": idx,
                "gender": gender,
                "beta_key": str(beta_key),
                "npz": str(npz_path),
                "motion": motion_rel,
                "duration": float(duration),
                "betas": betas.tolist(),
            }
        )

        print(f"[{idx}] saved {npz_path} | gender={gender} | beta_key={beta_key}")

    yaml_path = out_root / yaml_name
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"motions": yaml_motions}, f, sort_keys=False)

    manifest_path = out_root / f"humos_{args.num}_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump({"variants": manifest}, f, sort_keys=False)

    print(f"\nSaved YAML:     {yaml_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Exported:       {len(variants)} variants")


if __name__ == "__main__":
    main()

"""
python scripts/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto/ \
    --num 8

python data/scripts/convert_amass_to_motionlib.py \
    /home/hlz/datasets/humos_proto/ \
    /home/hlz/datasets/humos_proto_motionlib/ \
    --humanoid-type smpl \
    --output-fps 30 \
    --motion-config /home/hlz/datasets/humos_proto/humos_8.yaml \
    --force-remake \
    --device cuda

python data/scripts/convert_amass_to_motionlib_with_morphology.py \
  /home/hlz/datasets/humos_proto/ \
  /home/hlz/datasets/humos_proto_motionlib/ \
  --humanoid-type smpl \
  --output-fps 30 \
  --motion-config /home/hlz/datasets/humos_proto/humos_8.yaml \
  --force-remake \
  --device cuda

python examples/motion_libs_visualizer_mor.py \
    --motion_files /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
    --robot smpl \
    --simulator isaacgym

python examples/motion_libs_visualizer.py \
    --motion_files /home/hlz/datasets/humos_proto_motionlib/humos_128.pt \
    --robot smpl \
    --simulator isaacgym

python scripts/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto_single/ \
    --num 1

python data/scripts/convert_amass_to_motionlib_with_morphology.py \
    /home/hlz/datasets/humos_proto_single/ \
    /home/hlz/datasets/humos_proto_motionlib/ \
    --humanoid-type smpl \
    --output-fps 30 \
    --motion-config /home/hlz/datasets/humos_proto_single/humos_1.yaml \
    --force-remake \
    --device cuda

python examples/motion_libs_visualizer.py \
    --motion_files /home/hlz/datasets/humos_proto_motionlib/humos_1.pt \
    --robot hhi_smpl_single \
    --simulator isaacgym
    
"""
