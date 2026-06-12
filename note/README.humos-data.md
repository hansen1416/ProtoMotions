# HUMOS Data Used for Motion Generation

## Source Repo

`/home/hlz/repos/humos` — generates retargeted motions used as imitation learning data.

---

## AMASS Sub-datasets Used: 18 of ~23

**Not the entire AMASS dataset.** The 18 included sub-datasets are defined in `humos/prepare/raw_pose_processing_humos.ipynb`:

```
ACCAD, BioMotionLab_NTroje, BMLhandball, BMLmovi, CMU,
DFaust_67, EKUT, Eyes_Japan_Dataset, HumanEva, KIT,
MPI_HDM05, MPI_Limits, MPI_mosh, SFU, SSM_synced,
TCD_handMocap, TotalCapture, Transitions_mocap
```

The following 5 are **excluded** (folders present in `datasets/amass_data/` but not processed):

| Excluded sub-dataset |
|---|
| `DanceDB` |
| `GRAB` |
| `HUMAN4D` |
| `SOMA` |
| `WEIZMANN` |

---

## Additional Clip Filtering

`humos/prepare/clean_amass_data.py` removes specific clips after processing:

- **`BioMotionLab_NTroje`** — treadmill and "normal" walk clips removed (moved to `datasets/pose_data_backup/`)
- **`MPI_HDM05`** — inline skating clips (`HDM_dg_07-01*`) removed

---

## 128 Body Shapes (infer.py)

- `humos/all_betas.pt` — **64 unique beta vectors** (SMPL body shape parameters)
- Loop in `infer.py`: `for gender in [-1, 1]` — **2 genders** (female = -1, male = +1; neutral excluded)
- **64 betas × 2 genders = 128 total body shapes** per motion clip

Note: `infer_with_offset.py` uses 3 genders (`[-1, 0, 1]`) with 64 betas → 192 shapes per motion.

---

## Data Pipeline Summary

```
datasets/amass_data/   (raw AMASS .npz)
        ↓  raw_pose_processing_humos.ipynb  (18 sub-datasets, resampled to 20fps)
datasets/pose_data/    (processed .npy + .npz)
        ↓  clean_amass_data.py              (remove treadmill, normal walk, skating)
datasets/pose_data/    (cleaned)
        ↓  compute_3dfeats.py
datasets/humos3dfeats/ (3D features for training)
        ↓  infer.py                         (128 body shapes per motion)
output/                (retargeted .pt files → imitation learning)
```
