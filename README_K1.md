# README_K1 — Retargeting Human Motion to the K1 Humanoid with PHUMA

This guide is a **complete, self-contained walkthrough** for running the PHUMA pipeline on the
**K1** custom humanoid robot, using motion captured with **GVHMR** (video → SMPL-X).

It covers environment setup, the end-to-end data pipeline, every K1-specific configuration
detail, and the non-obvious gotchas (with fixes) discovered while bringing K1 online. A new
person should be able to go from a fresh clone to retargeted K1 motion using only this file.

> **Scope note.** PHUMA was designed for human-sized Unitree robots (G1 ~1.3 m, H1-2 ~1.8 m).
> K1 is **sub-human-sized** (~0.55 m hip height, ~0.41 m legs). Several pipeline assumptions
> break for a robot this small/differently-named; the fixes are documented here and have been
> applied to the code/config. Where a change touches shared scripts, it is **backward-compatible**
> (G1/H1-2/example workflows are unaffected).

---

## 0. TL;DR — the whole pipeline

```powershell
# 0. Activate env (Python 3.9). See Section 1 if not yet created.
conda activate phuma
$PROJECT_DIR = "D:\Code\PHUMA"

# 1. Put GVHMR .pt files in:  data/GVHMR/dataset_<name>/pt/
#    Convert .pt -> PHUMA (N,69) .npy:
python src/curation/gvhmr_to_phuma.py `
    --input_folder  data/GVHMR/dataset_<name>/pt `
    --output_folder data/GVHMR/dataset_<name>/npy

# 2. Physics-aware curation (one chunk per clip; see Section 4 for chunking):
python src/curation/preprocess_smplx_folder.py `
    --project_dir $PROJECT_DIR `
    --human_pose_folder data/GVHMR/dataset_<name>/npy `
    --foot_contact_threshold 0.05 `
    --chunk_size 999999 --chunk_overlap 0 --chunk_min_frames 1 `
    --visualize 1

# 3. (one-time per robot) Shape adaptation — find SMPL betas matching K1:
python src/retarget/shape_adaptation.py --project_dir $PROJECT_DIR --robot_name k1

# 4. Physics-constrained retargeting (writes humanoid_pose + video, mirrors subfolder):
python src/retarget/motion_adaptation_multiprocess.py `
    --project_dir $PROJECT_DIR --robot_name k1 `
    --human_pose_folder data/human_pose_preprocessed/dataset_<name> `
    --grounding_weight 30 --skating_weight 0.05 --num_iter_dof 5001 `
    --gpu_ids 0 --processes_per_gpu 3 --visualize 1
```

Outputs land in `data/humanoid_pose/k1/dataset_<name>/` and
`data/video/humanoid_pose/k1/dataset_<name>/`.

---

## 1. Environment setup (Windows + Conda)

PHUMA requires **Python 3.9**. Two Windows/Conda gotchas bit us; both have known fixes.

### 1.1 Create the env

```powershell
conda create -p D:\conda-envs\phuma python=3.9 -y
conda activate D:\conda-envs\phuma   # or: conda activate phuma  (see 1.4)
pip install -r requirements.txt
pip install -e .                     # REQUIRED: scripts do `from utils.smpl import *`
```

`requirements.txt` pins: torch 2.6.0, numpy 1.26.4, scipy 1.13.1, smplx 0.1.28,
mujoco 3.3.0, pytorch-kinematics 0.7.5, pyrender 0.1.45, trimesh 4.6.4, opencv 4.11,
pandas 2.2.3, PyYAML, easydict, tqdm, PyOpenGL.

### 1.2 GOTCHA — broken pip on fresh 3.9 envs (`dataclass() got an unexpected keyword argument 'slots'`)

Fresh conda envs may ship **pip 26**, which requires Python 3.10+ and crashes on every
invocation under 3.9. Fix (conda doesn't depend on the broken pip):

```powershell
conda install -p D:\conda-envs\phuma -y "pip>=24,<26"
```

This downgrades pip to 25.x. **Do not** run `pip install -U pip` in this env or it re-breaks.
`python -m ensurepip --upgrade` does **not** fix it (it refuses to "downgrade" from 26).

### 1.3 GOTCHA — CPU vs GPU torch

`requirements.txt` installs `torch==2.6.0` which on this machine resolved to **CPU-only**
(`2.6.0+cpu`). Consequences:
- **Curation and shape/motion adaptation run fine on CPU** (just slower). Pass `--device cpu`.
- The multiprocess retarget script's `--gpu_ids 0` will try `cuda:0`. If you have no CUDA
  torch, either install a CUDA build of torch, or drop `--gpu_ids`/`--processes_per_gpu` and
  use `--num_workers -1` (all CPU cores). At 5001 iters × many clips, CPU is slow but works.

### 1.4 GOTCHA — base env shadowing the project env (Windows PATH)

Base miniconda (Python 3.13) can shadow an activated env, so `pip install` silently runs
under 3.13 and tries to build numpy/scipy from source (scipy then fails needing a Fortran
compiler). Durable fixes already applied to this machine:

```powershell
conda config --set auto_activate_base false       # base no longer auto-loads
conda config --prepend envs_dirs D:\conda-envs     # `conda activate phuma` works by name
```

Always verify before installing/running: `python --version` must print **3.9.x**. If a VSCode
terminal still picks the wrong interpreter, run **Python: Select Interpreter → phuma** once.

### 1.5 SMPL-X model files (required, not in repo)

Curation and retargeting need the SMPL-X body model. Download from
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) (free registration) and place in
`asset/human_model/smplx/`:

```
asset/human_model/smplx/SMPLX_NEUTRAL.npz   (+ .pkl)
                        SMPLX_MALE.npz       (+ .pkl)
                        SMPLX_FEMALE.npz     (+ .pkl)
```

Curation/retargeting use the **NEUTRAL** model. Without these, the very first SMPL call crashes.

---

## 2. The K1 robot at a glance

| property | value |
|---|---|
| Hip height (T-pose `root_pos.z`) | 0.5519 m |
| Leg length (hip→foot, summed) | ~0.411 m |
| DOF | 22 (head ×2, each arm ×4, each leg ×6) |
| Spine / waist DOF | **none** (rigid `Trunk`) |
| Wrist DOF | **none** (arm ends at elbow-yaw `*_hand_link`) |
| Size vs. captured humans | ~0.44× of an SMPL-neutral human |

K1 model files live in `asset/humanoid_model/k1/`:
- `k1.xml` — original MuJoCo model (input to `setup_humanoid.py`)
- `custom.xml` — robot model with heel/toe keypoints (robot-only; **no scene assets**, see 5.2)
- `scene.xml` — MuJoCo scene that `<include>`s `custom.xml`
- `config.yaml` — the retargeting config (the heart of the setup; see Section 6)
- `meshes/` — STL files
- `betas.npy` — SMPL shape fit (generated by shape adaptation)

---

## 3. Data layout & conventions

Organize **every** new dataset under `data/GVHMR/dataset_<name>/` with `pt/` and `npy/`
subfolders. The pipeline then **auto-mirrors** `dataset_<name>` through all later stages
(this mirroring was added for K1 — see Section 5.5):

```
data/
├── GVHMR/
│   └── dataset_<name>/
│       ├── pt/                         # raw GVHMR hmr4d_results (.pt)  ← you put these here
│       └── npy/                        # converted PHUMA (N,69) .npy    ← gvhmr_to_phuma.py
├── human_pose_preprocessed/
│   └── dataset_<name>/                 # curated chunks (N,73) .npy     ← preprocess_smplx_folder.py
├── humanoid_pose/
│   └── k1/
│       └── dataset_<name>/             # retargeted robot motion .npy   ← motion_adaptation*.py
└── video/
    ├── human_pose_preprocessed/dataset_<name>/   # curation preview mp4
    └── humanoid_pose/k1/dataset_<name>/          # retargeted robot mp4
```

### Data formats

| file | shape | columns |
|---|---|---|
| GVHMR `.pt` | dict | `smpl_params_global` = {transl (N,3), global_orient (N,3), body_pose (N,63), betas (N,10)} |
| converted `.npy` | (N, 69) | `[transl(3), global_orient(3), body_pose(63)]` |
| curated chunk `.npy` | (N, 73) | `[transl(3), global_orient(3), body_pose(63), foot_contact(4)]` |
| retargeted `.npy` | dict | `{root_trans, root_ori(xyzw quat), dof_pos, fps}` |

**Why GVHMR maps cleanly:** GVHMR already emits a **63-dim body_pose** (21 SMPL-X joints),
exactly PHUMA's expectation, so `3+3+63 = 69` with no joint remapping. GVHMR's global frame
is **Y-up** (vertical range ~cm vs. metres horizontally), matching PHUMA's curation assumption.

> **Note on betas:** the (N,69) format **drops betas**. K1's body size is therefore handled by
> shape adaptation (`betas.npy`) + per-link scales + `root_scale`, **not** by the per-clip data.

---

## 4. Stage 1 — Convert & curate

### 4.1 Convert GVHMR → PHUMA

`src/curation/gvhmr_to_phuma.py` reads `smpl_params_global` from each `.pt` and writes a
(N,69) `.npy`.

```powershell
python src/curation/gvhmr_to_phuma.py `
    --input_folder  data/GVHMR/dataset_<name>/pt `
    --output_folder data/GVHMR/dataset_<name>/npy
```

### 4.2 Physics-aware curation

`src/curation/preprocess_smplx_folder.py` runs the SMPL-X forward pass + physics filters on
**CPU** (no GPU needed), splits into chunks, and writes passing chunks (N,73) with foot-contact.

```powershell
python src/curation/preprocess_smplx_folder.py `
    --project_dir $PROJECT_DIR `
    --human_pose_folder data/GVHMR/dataset_<name>/npy `
    --foot_contact_threshold 0.05 `
    --chunk_size 999999 --chunk_overlap 0 --chunk_min_frames 1 `
    --visualize 1
```

**The 5 physics filters** (per chunk): foot-contact score, root jerk, min/max pelvis height,
pelvis→base-of-support distance, spine→base-of-support distance. Tune via the `--*_threshold`
args. Raise `--foot_contact_threshold` (e.g. 0.6–0.8) for stricter, walking-only filtering;
0.05 is permissive (keeps almost everything).

**Chunking — why you may want one chunk per clip.** Defaults split motion into
**4 s windows (120 frames @30 fps) with 0.5 s overlap**, intended for RL reference clips.
For short GVHMR captures, pass `--chunk_size 999999 --chunk_overlap 0 --chunk_min_frames 1`
to keep each clip whole (one `*_chunk_0000.npy`). Trade-off: the physics filters are **per
chunk**, so with one big chunk a single bad span can fail the whole clip.

> Output path is auto-derived as `data/human_pose_preprocessed/dataset_<name>/...`
> (see 5.5). With `--visualize 1`, preview mp4s mirror to
> `data/video/human_pose_preprocessed/dataset_<name>/`.

---

## 5. Stage 2 — Retargeting (PhySINK) & the K1 fixes

### 5.1 One-time custom-robot setup

If starting from scratch for a robot, `setup_humanoid.py` generates `custom.xml`,
`scene.xml`, and a **draft** `config.yaml`:

```powershell
python src/utils/setup_humanoid.py --input asset\humanoid_model\k1\k1.xml --humanoid_type k1
```

**Always review the generated `config.yaml`** — for K1 the generator could **not** infer the
mappings (different naming), so several fields were filled in by hand (Section 6). The README
note "verify the bone mappings and keypoints" is not optional for a non-Unitree robot.

### 5.2 GOTCHA — duplicate texture name (`repeated name 'texplane'`)

`setup_humanoid.py` put scene assets (`skybox`, `texplane`, `matplane`, a `ground` geom)
**inside `custom.xml`**. Since `scene.xml` `<include>`s `custom.xml` and also defines
`texplane`, MuJoCo errors on the duplicate. **Fix (applied):** `custom.xml` now contains
**only** the robot (meshes + bodies); all scene/env assets removed (matches how G1 does it).
Verify: `python -c "import mujoco; mujoco.MjModel.from_xml_path('asset/humanoid_model/k1/scene.xml')"`.

### 5.3 Shape adaptation (one-time)

Finds the SMPL `betas` whose body best matches K1; saves `asset/humanoid_model/k1/betas.npy`.
Reused by all motion adaptation. CPU, ~5000 iters, a couple of minutes.

```powershell
python src/retarget/shape_adaptation.py --project_dir $PROJECT_DIR --robot_name k1
```

**Expected K1 result & a known limitation:** Kinematic Angular Loss converges low (~0.016)
once the config is correct, but **~5 of 10 betas saturate at the ±5 clamp** and Scale Unit
Loss stays ~0.35. This is **fundamental, not a bug**: SMPL's shape space is *human*; K1's legs
(0.41 m) are smaller than any SMPL body can reach (bottoms out ~0.53 m, still ~29% too big),
and K1's arms are proportionally much shorter (per-link scale ~2.4). For **locomotion** this is
mostly cosmetic (arms barely matter for walking). If you want to reduce saturation, lower
`--kinematic_loss_weight` and/or raise `--keypoint_matching_weight`.

### 5.4 Motion adaptation

Single file:
```powershell
python src/retarget/motion_adaptation.py `
    --project_dir $PROJECT_DIR --robot_name k1 `
    --human_pose_file dataset_<name>/<clip>_chunk_0000 `
    --device cpu --visualize 1 `
    --grounding_weight 30 --skating_weight 0.05 --num_iter_dof 5001
```

Folder (recommended, multiprocessing):
```powershell
python src/retarget/motion_adaptation_multiprocess.py `
    --project_dir $PROJECT_DIR --robot_name k1 `
    --human_pose_folder data/human_pose_preprocessed/dataset_<name> `
    --grounding_weight 30 --skating_weight 0.05 --num_iter_dof 5001 `
    --gpu_ids 0 --processes_per_gpu 3 --visualize 1
```

### 5.5 K1-specific code changes (applied; backward-compatible)

Three changes were made to shared scripts to support K1. All are gated/auto-detected so
G1/H1-2/example workflows behave identically to before.

1. **`joint_body_names` mapping (fixes "T-pose sliding / frozen joints").**
   The retarget FK loop mapped each joint to a body by checking if the *joint name is a
   substring of a body name* (works for G1: `left_knee_joint`→`left_knee_link`). K1's names
   (`Left_Knee_Pitch` vs body `Left_Shank`) don't match, so **16 of 22 joints were silently
   frozen** — the root moved but limbs stayed in T-pose. Fix: `config.yaml` now provides an
   explicit `joint_body_names` list (index-aligned with `joint_names`), and both
   `motion_adaptation.py` and `motion_adaptation_multiprocess.py` use it when present, else
   fall back to the old substring heuristic. **Both files were patched** (the multiprocess
   script has its own copy of the loop — easy to miss).

2. **`root_scale` (fixes over-striding / foot sliding for small robots).**
   PHUMA copies the human root path **verbatim** (fine for human-sized G1/H1-2). K1 at ~0.44×
   the human is dragged along a full-size path → each step over-reaches → planted foot slides.
   Fix: scale the **horizontal** root travel by `root_scale = robot_leg / captured_human_leg`.
   Height (z) is left unscaled so feet stay grounded; limb geometry is unchanged. Read from
   `config.yaml` `root_scale`, overridable per-run with `--root_scale`. This is the GMR-style
   trajectory normalization PHUMA omits.

   > **`root_scale` is SUBJECT-dependent.** It depends on the *captured human's* leg length,
   > which is encoded in the `.pt` **betas** (not the (N,69) data). Measure it per subject
   > from the `.pt` files. Current config value `0.415` targets the **turning_joe_2** subject
   > (real leg 0.989 m). For a different subject, recompute and pass `--root_scale`.

3. **Automatic dataset-subfolder mirroring.**
   Curation and both retarget scripts now preserve the `dataset_<name>` subfolder through
   `human_pose_preprocessed/`, `humanoid_pose/k1/`, and both `video/` trees automatically.
   `gvhmr_to_phuma.py` default paths were also fixed to `dataset_turning_joe_2`.

---

## 6. K1 `config.yaml` reference

Located at `asset/humanoid_model/k1/config.yaml`. Fields and what's K1-specific:

| field | meaning | K1 note |
|---|---|---|
| `root_pos` | T-pose base position; `z` = standing hip height | 0.5519 m (feet exactly touch ground) |
| `root_ori` | T-pose base orientation (xyzw quat) | identity |
| `root_scale` | **horizontal root-travel scale** = robot_leg / human_leg | **0.415** (subject = turning_joe_2); see 5.5 |
| `dof_pos` | T-pose joint angles (22) | **arms hand-fixed to a real T-pose** (see below) |
| `bone_mapping` | 12 bones: `[smpl_parent, smpl_child, robot_parent, robot_child]` | filled by hand (legs+arms); count must be ÷6 for the symmetry loss |
| `keypoints` | SMPL keypoint ↔ robot body, for position matching | empty `body:''` entries were filled by hand |
| `joint_body_names` | **body each joint drives** (index-aligned w/ `joint_names`, `''`=free base) | **K1-added**; fixes frozen-joints bug (5.5) |
| `body_names` | robot body order | — |
| `joint_names` | robot joint order (`floating_base_joint` first) | — |
| `joint_axes`, `dof`, `joint_velocity_limits` | per-joint axis / DOF index / vel limits | — |

### 6.1 The T-pose `dof_pos` (GOTCHA — auto-generated arms were wrong)

`setup_humanoid.py` set shoulder-roll to ±1.57 (arms raised straight **overhead**), which made
K1's arm bones ~90–105° off SMPL's horizontal T-pose → huge angular loss + betas pinned. The
arm DOFs were re-solved to a true T-pose. Current arm values (rest = 0):

```
Left  shoulder_pitch=0.7582, shoulder_roll=-0.1248, elbow_pitch=-2.2700
Right shoulder_pitch=0.7582, shoulder_roll= 0.1248, elbow_pitch= 2.2700
```

After this, all 12 bones align to **2.5°–20°** (legs 3–17°, arms ~11° / 2–4°), down from ~90°.

### 6.2 K1 bone/keypoint mapping (SMPL → K1 body)

| SMPL joint | K1 body | K1 joint at that body |
|---|---|---|
| hip | `Left/Right_Hip_Roll` | Hip_Roll |
| knee | `Left/Right_Shank` | Knee_Pitch |
| ankle | `Left/Right_Ankle_Cross` | Ankle_Pitch |
| shoulder | `Left/Right_Arm_2` | Shoulder_Roll |
| elbow | `Left/Right_Arm_3` | Elbow_Pitch |
| wrist | `left/right_hand_link` | Elbow_Yaw (terminal; no wrist DOF) |
| pelvis & torso | `Trunk` | (no spine DOF — both map to Trunk) |

---

## 7. Tuning guide (artifacts → knobs)

Re-render one clip and eyeball it; tune in this order.

| symptom | most likely cause | knob |
|---|---|---|
| Robot in T-pose, root slides/spins, limbs frozen | joint→body mapping broken | ensure `joint_body_names` in config (5.5); both scripts patched |
| Over-striding, lots of sliding, "propelled" steps | root path not size-scaled | `root_scale` (config) or `--root_scale`; = robot_leg/subject_leg |
| Uniform mild sliding on every clip | `root_scale` slightly off for this subject | recompute from subject betas; pass `--root_scale` |
| Feet float/penetrate intermittently | grounding penalty too weak | `--grounding_weight` 10 → 30 (→50) |
| Feet slide during contact | skating penalty too weak | `--skating_weight` 0.002 → 0.05 |
| Constant ~1 cm float (all frames, uniform) | heel/toe keypoint Z geometry | adjust keypoint z in `custom.xml` (not a weight) |
| Duck-footed / feet angled out | hip-yaw weakly constrained vs oversized targets | partly the betas size gap; tune match weights |
| Stiff gait after raising grounding | grounding fighting keypoint match | back off `--grounding_weight` |

Key defaults (motion adaptation): `grounding_weight=10`, `skating_weight=0.002`,
`num_iter_dof=3001`, `local_match_weight=2.0`, `global_match_weight=0.1`,
`joint_feasibility_weight=1000`. Output metrics printed at the end: **G-MPJPE**,
joint violation, foot-contact distance, skating velocity.

---

## 8. Computing `root_scale` for a new subject

`root_scale = K1_leg / captured_human_leg`, with `K1_leg ≈ 0.411 m`.
The human leg length comes from the **betas in the `.pt`** (the (N,69) `.npy` has no betas, so
it always reports the SMPL-neutral 0.929 m — don't measure from there).

Sketch:
```python
import torch, smplx, numpy as np
from smplx.joint_names import JOINT_NAMES
sm = {n:i for i,n in enumerate(JOINT_NAMES)}
smpl = smplx.create("asset/human_model", model_type="smplx", num_pca_comps=45)
betas = torch.load("data/GVHMR/dataset_<name>/pt/<clip>.pt", map_location="cpu",
                   weights_only=False)["smpl_params_global"]["betas"].mean(0)   # (10,)
j = smpl(betas=betas[None].float(), body_pose=None).joints[0].detach().numpy()
leg = (np.linalg.norm(j[sm["left_hip"]]-j[sm["left_knee"]])
     + np.linalg.norm(j[sm["left_knee"]]-j[sm["left_ankle"]])
     + np.linalg.norm(j[sm["left_ankle"]]-j[sm["left_foot"]]))
print("root_scale =", 0.411/leg)
```
Reference: SMPL-neutral subject → 0.44; **turning_joe_2** (leg 0.989 m) → **0.415**.

> `root_scale` is first-principles; real stride isn't perfectly linear in leg length, so the
> visually-best value may land ~0.40–0.50. Sweep with `--root_scale` if needed.

---

## 9. Troubleshooting quick index

- **`dataclass() ... 'slots'`** → broken pip 26 on py3.9 → `conda install -p <env> "pip>=24,<26"` (1.2).
- **scipy build fails (Fortran/`ifort`)** → install ran under base py3.13, not the env → fix env priority (1.4); verify `python --version`.
- **`repeated name 'texplane'`** → scene assets inside `custom.xml` → already removed (5.2).
- **`KeyError: ''`** in shape adaptation → empty `keypoints[].body` in config → fill them (6).
- **Output is a sliding/spinning T-pose** → joints frozen by substring mapping → use `joint_body_names`; patch *both* retarget scripts (5.5).
- **`--argument expected one argument`** → unset PowerShell var (e.g. `$PROJECT_DIR`) in a new shell → set it again.
- **`size of tensor a (4) must match b (0)`** in motion adaptation → input `.npy` has no foot-contact columns (not curated, only 69 cols) → run it through curation first (must be 73 cols).
- **`--help` crashes with a Unicode/encode error** → pre-existing Windows cp1252 console issue in argparse; harmless, the args still parse.

---

## 10. What is K1-specific vs. upstream

**Upstream PHUMA (unchanged behavior for G1/H1-2/example):** curation filters, PhySINK losses,
shape/motion adaptation core, the (N,69)/(N,73) formats.

**Added/changed for K1 (backward-compatible):**
- `config.yaml`: full `bone_mapping` + `keypoints`, corrected T-pose `dof_pos`,
  new `joint_body_names`, new `root_scale`.
- `custom.xml`: scene assets removed (robot-only).
- `motion_adaptation.py` / `motion_adaptation_multiprocess.py`: honor `joint_body_names`,
  honor `root_scale` (+ `--root_scale` flag), mirror `dataset_<name>` subfolders.
- `preprocess_smplx_folder.py`: route outputs (and preview videos) into `dataset_<name>/`.
- `gvhmr_to_phuma.py`: default paths point at `dataset_turning_joe_2`.

**Known residual limitation:** SMPL cannot fully represent K1's body (betas saturate); this is
inherent to a sub-human-sized, non-human-proportioned robot and is mostly cosmetic for
locomotion. For visually faithful playback on very small robots, a GMR-style IK retargeter
(no SMPL shape fitting) is architecturally a better fit; PHUMA's value here is its
physics curation + constraints.
