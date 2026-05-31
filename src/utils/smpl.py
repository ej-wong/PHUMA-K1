import os
import glob
import json
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
import torch
from scipy.spatial import ConvexHull, Delaunay


def _butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def _unify_quaternion_signs(quats):
    for i in range(1, quats.shape[0]):
        if np.dot(quats[i], quats[i-1]) < 0.0:
            quats[i] = -quats[i]
    return quats

def low_pass_filter(motion_parms, fps):
    transl_filtered = _butter_lowpass_filter(data=motion_parms["transl"], cutoff=3.0, fs=fps)
    transl_filtered = np.ascontiguousarray(transl_filtered)
    motion_parms["transl"] = torch.from_numpy(transl_filtered).float()

    rot_obj = R.from_rotvec(motion_parms["global_orient"]) 
    quats = rot_obj.as_quat()
    quats = _unify_quaternion_signs(quats)
    quats_filtered = _butter_lowpass_filter(data=quats, cutoff=6.0, fs=fps)
    quats_filtered /= np.linalg.norm(quats_filtered, axis=1, keepdims=True)
    quats_filtered = np.ascontiguousarray(R.from_quat(quats_filtered).as_rotvec())
    motion_parms["global_orient"] = torch.from_numpy(quats_filtered).float()

    body_pose_filtered = _butter_lowpass_filter(data=motion_parms['body_pose'], cutoff=6.0, fs=fps)
    body_pose_filtered = np.ascontiguousarray(body_pose_filtered)
    motion_parms['body_pose'] = torch.from_numpy(body_pose_filtered).float()

    return motion_parms

def find_robust_ground(vertices, foot_contact_vertex_indices, ground_thresh=0.05, heel_offset=0.005):
    toe_contact_indices = list(foot_contact_vertex_indices["left_toe_indices"]) + list(foot_contact_vertex_indices["right_toe_indices"])
    heel_contact_indices = list(foot_contact_vertex_indices["left_heel_indices"]) + list(foot_contact_vertex_indices["right_heel_indices"])
    toe_contact_heights = np.min(vertices[:, toe_contact_indices, 1], axis=1)
    heel_contact_heights = np.min(vertices[:, heel_contact_indices, 1], axis=1) + heel_offset
    contact_heights = np.mean(np.stack([toe_contact_heights, heel_contact_heights], axis=1), axis=1)
    contact_heights.sort()

    max_count = 0
    best_h_stars = []
    
    j = 0
    for i in range(len(contact_heights)):
        while j < len(contact_heights) and contact_heights[j] <= contact_heights[i] + ground_thresh:
            j += 1
        current_count = j - i
        current_h_star = contact_heights[i] + ground_thresh / 2
        
        if current_count > max_count:
            max_count = current_count
            best_h_stars = [current_h_star]
        elif current_count == max_count:
            best_h_stars.append(current_h_star)

    return np.median(best_h_stars)

def get_foot_contact(vertices, foot_contact_vertex_indices, robust_ground_y, ground_threshold, heel_offset=0.005):
    num_frames = vertices.shape[0]
    foot_contacts = np.zeros((num_frames, len(foot_contact_vertex_indices)), dtype=np.float32)
    contact_range = [robust_ground_y - ground_threshold / 2.0, robust_ground_y + ground_threshold / 2.0]

    for i in range(num_frames):
        for j, (contact_type, indices) in enumerate(foot_contact_vertex_indices.items()):
            heights = vertices[i, indices, 1]
            offset = heel_offset if "heel" in contact_type else 0.0
            contact_count = np.sum((heights >= contact_range[0] + offset) & (heights <= contact_range[1] + offset))
            foot_contacts[i, j] = contact_count / len(indices)

    return foot_contacts

def load_motion_parms(smpl_pose_file, foot_contact: bool): 
    data = np.load(smpl_pose_file)
    transl = data[:, 0:0+3]
    global_orient = data[:, 3:3+3]
    body_pose = data[:, 6:6+63]
    motion_parms = {
        'body_pose': torch.from_numpy(body_pose).float(),
        'transl': torch.from_numpy(transl).float(),
        'global_orient': torch.from_numpy(global_orient).float(),
    }

    if data.shape[1] == 69+10:
        betas = data[:, 69:69+10]
        motion_parms["betas"] = torch.from_numpy(betas).float()

    if foot_contact:
        motion_parms["foot_contact"] = torch.from_numpy(data[:, 69:69+4]).float()

    assert data.shape[1] != 69+10 or not foot_contact

    return motion_parms

def _point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    
    dot_ab_ab = np.dot(ab, ab)
    if dot_ab_ab == 0:
        return np.linalg.norm(ap)

    t = np.dot(ap, ab) / dot_ab_ab
    
    if 0.0 <= t <= 1.0:
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    else:
        return min(np.linalg.norm(p - a), np.linalg.norm(p - b))

def calculate_bos_distance(joints, target_joint_id=0):
    num_frames = joints.shape[0]
    target_projection = joints[:, target_joint_id, [0, 2]]
    frame_distances = []
    
    for i in range(num_frames):
        # Get the 4 joint positions: left ankle (7), right ankle (8), left foot (10), right foot (11)
        foot_ankle_joints = joints[i, [7, 8, 10, 11], :]
        foot_ankle_points_xz = foot_ankle_joints[:, [0, 2]]
        pelvis_pt = target_projection[i]
        
        unique_points = np.unique(foot_ankle_points_xz, axis=0)
        
        hull = ConvexHull(unique_points)
        if Delaunay(hull.points[hull.vertices]).find_simplex(pelvis_pt) >= 0:
            dist = 0.0
        else:
            hull_vertices = hull.points[hull.vertices]
            min_dist_to_edge = float('inf')
            for j in range(len(hull_vertices)):
                p1 = hull_vertices[j]
                p2 = hull_vertices[(j + 1) % len(hull_vertices)]
                seg_dist = _point_to_segment_dist(pelvis_pt, p1, p2)
                if seg_dist < min_dist_to_edge:
                    min_dist_to_edge = seg_dist
            dist = min_dist_to_edge
                
        frame_distances.append(dist)

    return np.mean(frame_distances) if sum(frame_distances) > 0.0 else 0.0


# --- Automatic per-dataset root_scale -------------------------------------
# root_scale = robot_leg / captured_human_leg. The human leg length depends on
# SMPL *betas*, which only exist in the source GVHMR .pt files (the (N,69)/(N,73)
# pipeline drops them). These helpers compute the dataset's mean human leg from
# those .pt files (cached), so retargeting can size the root trajectory to any
# robot automatically instead of relying on a hand-set per-subject constant.

_LEG_CACHE_NAME = "human_leg.json"


def _smpl_leg_length_from_betas(betas, human_model_dir):
    """Sum hip->knee->ankle->foot segment lengths for an SMPL-X body (metres)."""
    import smplx
    from smplx.joint_names import JOINT_NAMES
    sm = {n: i for i, n in enumerate(JOINT_NAMES)}
    model = smplx.create(human_model_dir, model_type="smplx", num_pca_comps=45)
    b = torch.as_tensor(betas, dtype=torch.float32).reshape(1, -1)
    j = model(betas=b, body_pose=None).joints[0].detach().cpu().numpy()
    return float(
        np.linalg.norm(j[sm["left_hip"]] - j[sm["left_knee"]])
        + np.linalg.norm(j[sm["left_knee"]] - j[sm["left_ankle"]])
        + np.linalg.norm(j[sm["left_ankle"]] - j[sm["left_foot"]])
    )


def dataset_mean_human_leg(pt_folder, human_model_dir, use_cache=True):
    """Mean human leg length (m) over all GVHMR .pt files in pt_folder, or None.

    Result is cached as <pt_folder>/../human_leg.json so it is computed once per
    dataset. Returns None if the folder/.pt files/betas are unavailable.
    """
    if not pt_folder or not os.path.isdir(pt_folder):
        return None
    cache_path = os.path.join(os.path.dirname(os.path.normpath(pt_folder)), _LEG_CACHE_NAME)
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return float(json.load(f)["mean_human_leg"])
        except Exception:
            pass  # corrupt cache -> recompute

    pts = sorted(glob.glob(os.path.join(pt_folder, "*.pt")))
    if not pts:
        return None
    legs = []
    for p in pts:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            betas = d["smpl_params_global"]["betas"]
            betas = torch.as_tensor(betas).float().mean(0)  # (10,)
            legs.append(_smpl_leg_length_from_betas(betas, human_model_dir))
        except Exception:
            continue
    if not legs:
        return None
    mean_leg = float(np.mean(legs))
    try:
        with open(cache_path, "w") as f:
            json.dump({"mean_human_leg": mean_leg, "n_clips": len(legs)}, f, indent=2)
    except Exception:
        pass  # caching is best-effort
    return mean_leg


def resolve_root_scale(cli_root_scale, robot_config, project_dir, dataset_name,
                       human_model_dir, verbose=True):
    """Resolve the root_scale to use, in priority order:

    1. cli_root_scale (explicit --root_scale)        -> use as given
    2. auto: robot_leg_length / dataset_mean_human_leg  (needs both available)
    3. robot_config['root_scale']                    -> hand-set fallback
    4. 1.0                                            -> no scaling

    Auto-scale activates only when the robot config has 'robot_leg_length' AND
    the dataset's GVHMR .pt files are present, so robots/datasets that don't
    follow the convention keep the previous behaviour.
    """
    if cli_root_scale is not None:
        if verbose:
            print(f"[root_scale] using --root_scale override: {cli_root_scale:.4f}")
        return float(cli_root_scale)

    robot_leg = robot_config.get("robot_leg_length")
    if robot_leg and dataset_name:
        pt_folder = os.path.join(project_dir, "data", "GVHMR", dataset_name, "pt")
        human_leg = dataset_mean_human_leg(pt_folder, human_model_dir)
        if human_leg:
            scale = float(robot_leg) / human_leg
            if verbose:
                print(f"[root_scale] auto: robot_leg {robot_leg:.3f} / "
                      f"mean_human_leg {human_leg:.3f} (dataset '{dataset_name}') = {scale:.4f}")
            return scale
        elif verbose:
            print(f"[root_scale] auto unavailable (no .pt/betas at {pt_folder}); "
                  f"falling back to config value")

    fallback = robot_config.get("root_scale", 1.0)
    if verbose:
        print(f"[root_scale] using config root_scale: {fallback}")
    return float(fallback)


def dataset_name_from_path(path):
    """Best-effort dataset folder name from a retarget --human_pose_folder/-file.

    Folder form: 'data/human_pose_preprocessed/dataset_x' -> 'dataset_x'.
    File form:   'dataset_x/clip_chunk_0000'              -> 'dataset_x'.
    Returns the basename for a folder, else the first path component.
    """
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    # A folder reference ends in the dataset dir; a file reference includes a clip name.
    # Heuristic: if there's a parent dir in the relative path, the dataset is the first
    # component (file form); otherwise it's the basename (folder form).
    parts = [p for p in norm.split(os.sep) if p not in ("", ".", "..")]
    if not parts:
        return None
    # 'human_pose_preprocessed' anchors the folder form -> dataset is the component after it
    if "human_pose_preprocessed" in parts:
        idx = parts.index("human_pose_preprocessed")
        return parts[idx + 1] if idx + 1 < len(parts) else None
    # File form (relative, e.g. 'dataset_x/clip'): first component is the dataset
    if len(parts) >= 2:
        return parts[0]
    return base