
import torch

data = torch.load("/home/hlz/datasets/humos_proto/humos_8_offset.pt", map_location="cpu", weights_only=False)

asset_ids    = data["motion_asset_ids"]     # tuple of strings
genders      = data["motion_genders"]       # tuple of strings
beta_keys    = data["motion_beta_keys"]     # tuple of strings
gender_ids   = data["motion_gender_ids"]    # tensor, -1/1
betas        = data["motion_betas"]         # [N, 10]

gender_id_map = {"male": 1, "female": -1}

for i in range(len(asset_ids)):
    expected_asset_id = f"{genders[i]}_{beta_keys[i]}"
    expected_gender_id = gender_id_map[genders[i]]

    assert asset_ids[i] == expected_asset_id, f"[{i}] asset_id mismatch"
    assert gender_ids[i].item() == expected_gender_id, f"[{i}] gender_id mismatch"
    print(f"[{i}] {asset_ids[i]}  betas[:3]={betas[i,:3].tolist()}")
