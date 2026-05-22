# scripts/read_humos_frame0_offsets.py

"""
python scripts/read_humos_frame0_offsets.py \
    --offset-file /home/hlz/datasets/humos_frame0_offsets.pt
"""

from pathlib import Path
import argparse
import torch


def load_offset_db(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offset-file",
        type=Path,
        default=Path("/home/hlz/datasets/humos_frame0_offsets.pt"),
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=10,
        help="Number of clips to print. Use -1 for all.",
    )
    parser.add_argument(
        "--limit-betas",
        type=int,
        default=5,
        help="Number of beta offsets to print per gender. Use -1 for all.",
    )

    args = parser.parse_args()

    offset_db = load_offset_db(args.offset_file)

    print(f"Loaded: {args.offset_file}")
    print(f"Number of clips: {len(offset_db)}")
    print()

    all_offsets = []

    clip_items = list(offset_db.items())
    if args.limit_clips > 0:
        clip_items = clip_items[: args.limit_clips]

    for clip_id, gender_dict in clip_items:
        print(f"clip_id = {clip_id}")

        total_shapes = 0

        for gender, beta_dict in gender_dict.items():
            print(f"  gender = {gender}, num_betas = {len(beta_dict)}")
            total_shapes += len(beta_dict)

            beta_items = list(beta_dict.items())
            if args.limit_betas > 0:
                beta_items = beta_items[: args.limit_betas]

            for beta_key, offset in beta_items:
                offset = float(offset)
                all_offsets.append(offset)
                print(f"    beta_key = {beta_key}, offset = {offset:.6f}")

            if args.limit_betas > 0 and len(beta_dict) > args.limit_betas:
                print("    ...")

            if args.limit_betas <= 0:
                all_offsets.extend(float(v) for v in beta_dict.values())

        print(f"  total_shapes = {total_shapes}")
        print("-" * 60)

    # Collect all offsets for global stats, not only printed ones.
    all_offsets = []
    for gender_dict in offset_db.values():
        for beta_dict in gender_dict.values():
            for offset in beta_dict.values():
                all_offsets.append(float(offset))

    if len(all_offsets) > 0:
        t = torch.tensor(all_offsets, dtype=torch.float32)

        print()
        print("[GLOBAL OFFSET STATS]")
        print(f"count = {t.numel()}")
        print(f"min   = {t.min().item():.6f}")
        print(f"max   = {t.max().item():.6f}")
        print(f"mean  = {t.mean().item():.6f}")
        print(f"std   = {t.std(unbiased=False).item():.6f}")

        large = torch.abs(t) > 0.5
        print(f"abs(offset) > 0.5 count = {large.sum().item()}")

    else:
        print("[WARN] offset db is empty")


if __name__ == "__main__":
    main()