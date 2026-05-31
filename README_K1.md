# README_K1 — Retargeting Human Motion to the K1 Humanoid with PHUMA

A self-contained guide to running the PHUMA pipeline on the **K1** humanoid using motion
captured with **GVHMR** (video → SMPL-X). Covers setup, the end-to-end pipeline, and the
K1-specific configuration.

---

## 0. Pipeline Overview

The full pipeline, end to end:

```powershell
conda activate phuma
$PROJECT_DIR = "D:\Code\PHUMA"

# 1. Put GVHMR .pt files in data/GVHMR/dataset_<name>/pt/, then convert to (N,69) .npy:
python src/curation/gvhmr_to_phuma.py `
    --input_folder  data/GVHMR/dataset_<name>/pt `
    --output_folder data/GVHMR/dataset_<name>/npy

# 2. Physics-aware curation (one chunk per clip):
python src/curation/preprocess_smplx_folder.py `
    --project_dir $PROJECT_DIR `
    --human_pose_folder data/GVHMR/dataset_<name>/npy `
    --foot_contact_threshold 0.05 --ground_threshold 0.05 `
    --chunk_size 999999 --chunk_overlap 0 --chunk_min_frames 1 --visualize 1

# 3. (one-time per robot) Shape adaptation — fit SMPL betas to K1:
python src/retarget/shape_adaptation.py --project_dir $PROJECT_DIR --robot_name k1

# 4. Physics-constrained retargeting (root_scale auto-resolved per dataset):
python src/retarget/motion_adaptation_multiprocess.py `
    --project_dir $PROJECT_DIR --robot_name k1 `
    --human_pose_folder data/human_pose_preprocessed/dataset_<name> `
    --grounding_weight 30 --skating_weight 0.05 --num_iter_dof 5001 `
    --gpu_ids 0 --processes_per_gpu 3 --visualize 1
```

Outputs: `data/humanoid_pose/k1/dataset_<name>/` and `data/video/humanoid_pose/k1/dataset_<name>/`.

---

## 1. Environment Setup

**Python 3.9 with Conda.**

```powershell
conda create -p D:\conda-envs\phuma python=3.9 -y
conda activate phuma
conda install -p D:\conda-envs\phuma -y "pip>=24,<26"   # Python 3.9 needs pip < 26
pip install -r requirements.txt
pip install -e .                                          # required: scripts import `utils.*`
```

- Verify `python --version` prints **3.9.x** before installing/running.
- `torch==2.6.0` may resolve to **CPU-only**. To run on CPU, pass `--device cpu` (single-file)
  or drop `--gpu_ids`/`--processes_per_gpu` and use `--num_workers -1` (multiprocess).

**SMPL-X models** (required, not in repo): download from
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and place
`SMPLX_NEUTRAL/MALE/FEMALE.{npz,pkl}` in `asset/human_model/smplx/`. Curation uses NEUTRAL.

---

## 2. The K1 Robot

| property | value |
|---|---|
| Hip height (`root_pos.z`) | 0.5519 m |
| Leg length (`robot_leg_length`) | 0.411 m |
| DOF | 22 (head ×2, each arm ×4, each leg ×6) |
| Spine / wrist DOF | none (rigid `Trunk`; arms end at `*_hand_link`) |

Model files in `asset/humanoid_model/k1/`: `k1.xml` (source), `custom.xml` (robot-only model
with heel/toe keypoints), `scene.xml` (scene that `<include>`s `custom.xml`), `config.yaml`,
`meshes/`, and `betas.npy` (written by shape adaptation).

---

## 3. Data Layout and Formats

Put every dataset under `data/GVHMR/dataset_<name>/` with `pt/` and `npy/`. The pipeline
**auto-mirrors** `dataset_<name>` through all later stages:

```
data/
├── GVHMR/dataset_<name>/{pt,npy}                  # raw .pt  → converted (N,69) .npy
├── human_pose_preprocessed/dataset_<name>/        # curated chunks (N,73)
├── humanoid_pose/k1/dataset_<name>/               # retargeted robot motion
└── video/{human_pose_preprocessed,humanoid_pose/k1}/dataset_<name>/   # preview / robot mp4
```

| file | shape | columns |
|---|---|---|
| GVHMR `.pt` | dict | `smpl_params_global` = {transl(N,3), global_orient(N,3), body_pose(N,63), betas(N,10)} |
| converted `.npy` | (N, 69) | `[transl(3), global_orient(3), body_pose(63)]` |
| curated chunk `.npy` | (N, 73) | `[transl(3), global_orient(3), body_pose(63), foot_contact(4)]` |
| retargeted `.npy` | dict | `{root_trans, root_ori (xyzw quat), dof_pos, fps}` |

GVHMR's 63-dim body_pose maps directly to PHUMA (no joint remapping); its global frame is Y-up.
Note `betas` exist only in the `.pt` files — the (N,69)/(N,73) formats drop them.

---

## 4. Pipeline Stages

### 4.1 Conversion (`gvhmr_to_phuma.py`)
Reads `smpl_params_global` from each `.pt`, writes (N,69) `.npy`.

### 4.2 Curation (`preprocess_smplx_folder.py`)
Runs the SMPL-X forward pass + 5 physics filters (foot-contact score, root jerk, min/max pelvis
height, pelvis/spine→base-of-support), splits into chunks, writes passing (N,73) chunks. CPU-only.

- Keep each clip whole: `--chunk_size 999999 --chunk_overlap 0 --chunk_min_frames 1`
  (default is 4 s / 120-frame windows with 0.5 s overlap, for RL reference clips).
- `--foot_contact_threshold` — drops chunks below this contact score (higher = stricter).
- `--ground_threshold` — width of the ground-contact band (see Section 7).

### 4.3 Shape adaptation (`shape_adaptation.py`, one-time per robot)
Fits SMPL `betas` to K1; writes `asset/humanoid_model/k1/betas.npy`, reused by all motion
adaptation. CPU, ~5000 iters.

### 4.4 Motion Adaptation (`motion_adaptation.py` / `_multiprocess.py`)
PhySINK optimization → robot motion + optional video. Single-file uses `--human_pose_file
dataset_<name>/<clip>_chunk_0000`; the multiprocess version takes `--human_pose_folder` and
parallelizes (`--gpu_ids`, `--processes_per_gpu`, or `--num_workers -1` for CPU). End-of-run
metrics: G-MPJPE, joint violation, foot-contact distance, skating velocity.

---

## 5. K1 `config.yaml` Reference

`asset/humanoid_model/k1/config.yaml`:

| field | meaning |
|---|---|
| `root_pos`, `root_ori` | T-pose base position / orientation (xyzw quat) |
| `dof_pos` | T-pose joint angles (22), arms in a horizontal T |
| `robot_leg_length` | K1 leg length; drives automatic `root_scale` (Section 6) |
| `root_scale` | fallback path-scale, used only if auto can't run (Section 6) |
| `bone_mapping` | 12 bones `[smpl_parent, smpl_child, robot_parent, robot_child]`; count must be ÷6 |
| `keypoints` | SMPL keypoint ↔ robot body, for position matching |
| `joint_body_names` | body each joint drives, index-aligned with `joint_names` (`''` = free base) |
| `body_names`, `joint_names`, `joint_axes`, `dof`, `joint_velocity_limits` | robot kinematic definitions |

SMPL→K1 body mapping: hip→`*_Hip_Roll`, knee→`*_Shank`, ankle→`*_Ankle_Cross`,
shoulder→`*_Arm_2`, elbow→`*_Arm_3`, wrist→`*_hand_link`; pelvis & torso both → `Trunk`.

> Custom robots: `setup_humanoid.py --input <xml/urdf> --humanoid_type <name>` generates
> `custom.xml`/`scene.xml`/`config.yaml`. `custom.xml` must contain **only** the robot (no scene
> texture/material/ground assets — those live in `scene.xml`). Review the generated
> `bone_mapping`, `keypoints`, and `joint_body_names` before retargeting.

---

## 6. Automatic Root Scaling

A smaller-than-human robot must follow a **scaled-down** root path or it over-strides and the
feet slide. PHUMA scales horizontal root travel by `root_scale`; z (height) is left unscaled.

`root_scale` is resolved automatically per dataset, in priority order:

1. `--root_scale <value>` flag (manual override)
2. **Auto**: `robot_leg_length / mean_human_leg`, where the human leg is averaged from the
   dataset's GVHMR `.pt` betas. Cached at `data/GVHMR/dataset_<name>/human_leg.json`.
3. `config.yaml` `root_scale` (fallback, if `.pt`/betas unavailable)
4. `1.0`

No per-dataset action needed — just retarget, and the resolved value is printed
(`[root_scale] auto: robot_leg 0.411 / mean_human_leg 0.989 ... = 0.4154`). Auto requires
`robot_leg_length` in the robot config and the dataset's `.pt` files present; otherwise it
falls back, so robots/datasets that don't follow the convention are unaffected.

---

## 7. Tuning Parameters

Parameters are grouped by the pipeline stage that consumes them.

### Curation — `preprocess_smplx_folder.py`

| parameter (default) | effect |
|---|---|
| `--ground_threshold` (0.05) | Width of the ground-contact band. Widen if stance frames are missed (feet float in the retargeted output); tighten if too many frames register as contact (penetration or no foot clearance). |
| `--foot_contact_threshold` (0.6) | Minimum contact score for a chunk to be retained. Higher values filter more aggressively. |

### Motion adaptation — `motion_adaptation.py` / `motion_adaptation_multiprocess.py`

| parameter (default) | effect |
|---|---|
| `--grounding_weight` (10) | Increase to 30–50 to reduce foot penetration or float during contact phases. |
| `--skating_weight` (0.002) | Increase to ~0.05 to reduce foot sliding during contact phases. |
| `--num_iter_dof` (3001) | Increase to ~5001 for improved convergence. |

Foot clearance and overall gait quality are bounded by the source capture: retargeting
reproduces the captured motion and cannot add movement the subject did not perform.
