from pathlib import Path
import argparse
import re
import torch
import yaml


FILENAME_RE = re.compile(r"^(male|female|neutral)_(.+)_smpl\.xml$")


def to_float_list(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().tolist()
    return torch.as_tensor(x, dtype=torch.float32).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        default="protomotions/data/assets",
        help="ProtoMotions asset root.",
    )
    parser.add_argument(
        "--asset-folder",
        default="mjcf/smpl_mor",
        help="Folder containing morphology XML files, relative to asset-root.",
    )
    parser.add_argument(
        "--betas-file",
        default="protomotions/data/assets/all_betas.pt",
        help="Path to all_betas.pt.",
    )
    parser.add_argument(
        "--out",
        default="protomotions/data/assets/mjcf/smpl_mor/assets.yaml",
        help="Output YAML path.",
    )
    parser.add_argument(
        "--default-root-height",
        type=float,
        default=0.95,
        help="Temporary default root height for all templates.",
    )
    args = parser.parse_args()

    asset_root = Path(args.asset_root)
    asset_dir = Path(asset_root / args.asset_folder)
    betas_path = Path(args.betas_file)
    out_path = Path(args.out)

    if not asset_dir.exists():
        raise FileNotFoundError(f"Asset folder does not exist: {asset_dir}")

    if not betas_path.exists():
        raise FileNotFoundError(f"Betas file does not exist: {betas_path}")

    all_betas = torch.load(betas_path, map_location="cpu")

    if not isinstance(all_betas, dict):
        raise TypeError(
            f"Expected all_betas.pt to contain a dict, got {type(all_betas)}"
        )

    xml_files = sorted(asset_dir.glob("*.xml"))

    assets = []
    missing_betas = []

    for xml_file in xml_files:
        match = FILENAME_RE.match(xml_file.name)
        if match is None:
            print(f"[skip] filename does not match pattern: {xml_file.name}")
            continue

        gender, beta_key = match.groups()

        if beta_key not in all_betas:
            missing_betas.append(beta_key)
            print(f"[skip] beta_key not found in all_betas.pt: {beta_key}")
            continue

        betas = to_float_list(all_betas[beta_key])

        assets.append(
            {
                "asset_id": f"{gender}_{beta_key}",
                "gender": gender,
                "beta_key": beta_key,
                "betas": betas,
                "root_height": args.default_root_height,
            }
        )

    if len(assets) == 0:
        raise RuntimeError(f"No valid morphology XML assets found in {asset_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        yaml.safe_dump(assets, f, sort_keys=False)

    print(f"Saved asset info YAML: {out_path}")
    print(f"Valid assets: {len(assets)}")

    if missing_betas:
        unique_missing = sorted(set(missing_betas))
        print(f"Missing beta keys: {len(unique_missing)}")
        for key in unique_missing[:20]:
            print(f"  {key}")


if __name__ == "__main__":
    main()


    """
    python scripts/generate_smpl_mor_asset_info.py \
    --asset-folder mjcf/smpl_mor \
    --betas-file protomotions/data/assets/all_betas.pt \
    --out protomotions/data/assets/mjcf/smpl_mor/assets.yaml
    """