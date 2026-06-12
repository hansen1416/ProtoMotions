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

## ProtoMotions "99% Success on Entire AMASS" — Why It's Misleading

ProtoMotions claims 99% imitation learning success on the entire AMASS dataset. This is questionable for several reasons.

### 1. "Success" = didn't fall, not faithful reproduction

In `protomotions/agents/evaluators/base_evaluator.py`:
```python
success_rate = 1.0 - self._motion_failed.float().mean().item()
```
A motion is only marked as "failed" if the physics rollout terminates early (character falls over). A character that hovers in a half-crouch through a "sit in chair" clip without falling is counted as a **success**, even though no chair exists in the simulation.

### 2. They don't filter semantically invalid motions

ProtoMotions' `data/scripts/motion_filter.py` only applies physics-quality filters (body parts below ground, unrealistically high velocities). It does **not** filter motions that require objects the simulator never provides.

### 3. 1,508 motions in HumanML3D require external objects

From `/home/hlz/repos/hhi/data-processing/invalid_category_counts.txt` (predecessor project analysis):

| Category | Count |
|---|---|
| `seat_support` (chairs, benches) | 657 |
| `terrain_or_structure` | 198 |
| `other_person_or_animal` | 187 |
| `table_shelf_surface` | 174 |
| `wall_door_window` | 111 |
| `obstacle_or_gap` | 83 |
| `external_support` | 53 |
| `forceful_object_interaction` | 32 |
| `environment_medium` (water, etc.) | 13 |
| **Total** | **1,508 of 22,459 (6.7%)** |

These motions are physically impossible to reproduce faithfully in a plain flat-ground simulation. The character will approximate the pose without the object — and as long as it doesn't fall, ProtoMotions counts it as a success.

### Summary

| Claim | Reality |
|---|---|
| "Entire AMASS" | Likely HumanML3D subset (~22K motions); ProtoMotions also applies velocity/height quality filters |
| "99% success" | 99% of rollouts don't fall over — not 99% faithful motion reproduction |
| Object-interaction motions | Not filtered; character improvises and passes the fall-detection threshold |

The filtering in `/home/hlz/repos/hhi/data-processing/` (valid_motions.txt / invalid_motions.txt) is the more principled approach: it excludes motions that are semantically impossible to reproduce in a flat-ground, object-free simulation.

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
