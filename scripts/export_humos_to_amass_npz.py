# scripts/export_humos_to_amass_npz.py

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="One HUMOS output .pt file")
    parser.add_argument("--out-root", required=True, help="AMASS-like output root")
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--beta-key", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--apply-offset-height", action="store_true")
    args = parser.parse_args()

    src = Path(args.input)
    out_root = Path(args.out_root)
    seq_dir = out_root / "HUMOS"
    seq_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(src, map_location="cpu")

    gender_block = data[args.gender]
    beta_key = args.beta_key or next(iter(gender_block.keys()))
    item = gender_block[beta_key]

    root_orient = to_np(item["root_orient"]).astype(np.float32)  # [T, 3]
    pose_body = to_np(item["pose_body"]).astype(np.float32)  # [T, 23, 3] or [T, 69]
    trans = to_np(item["trans"]).astype(np.float32)  # [T, 3]

    T = root_orient.shape[0]

    if pose_body.ndim == 3:
        pose_body = pose_body.reshape(T, -1)

    # ProtoMotions SMPL converter uses root + first 21 body joints = 66 dims,
    # then pads the remaining 2 joints internally.
    pose_body_21 = pose_body[:, :63]
    poses = np.concatenate([root_orient, pose_body_21], axis=1).astype(
        np.float32
    )  # [T, 66]

    if args.apply_offset_height and "offset_height" in item:
        trans[:, 2] += float(to_np(item["offset_height"]))

    betas = to_np(item["betas"]).astype(np.float32)
    if betas.ndim == 2:
        betas = betas[0]

    out_name = f"{safe_name(src.stem)}_{args.gender}_{safe_name(beta_key)}"
    npz_path = seq_dir / f"{out_name}.npz"

    np.savez(
        npz_path,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=np.array(args.gender),
        mocap_framerate=np.array(args.fps, dtype=np.float32),
    )

    duration = T / args.fps

    yaml_path = out_root / "humos_one.yaml"
    motion_rel = f"HUMOS/{out_name}.motion"

    yaml_data = {
        "motions": [
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
        ]
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    print(f"Saved npz:  {npz_path}")
    print(f"Saved yaml: {yaml_path}")
    print(f"Duration:   {duration:.3f}s")


if __name__ == "__main__":
    main()

"""
python scripts/export_humos_to_amass_npz.py \
    --input /home/hlz/datasets/humos_output/000005.pt \
    --out-root /home/hlz/datasets/humos_proto \
    --gender male


python data/scripts/convert_amass_to_motionlib.py \
    /home/hlz/datasets/humos_proto/ \
    /home/hlz/datasets/humos_proto_motionlib/ \
    --humanoid-type smpl \
    --output-fps 30 \
    --motion-config /home/hlz/datasets/humos_proto/humos_one.yaml \
    --force-remake \
    --device cuda

python examples/motion_libs_visualizer.py \
    --motion_files /home/hlz/datasets/humos_proto_motionlib/humos_one.pt \
    --robot smpl \
    --simulator isaacgym
"""
