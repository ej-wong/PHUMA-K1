$PROJECT_DIR = "D:\Code\PHUMA"
$human_pose_file = "D:\Code\PHUMA\data\human_pose\locomotionGVHMR_npy\turning_CW_slow"

python src/curation/gvhmr_to_phuma.py

python src/curation/preprocess_smplx_folder.py `
    --project_dir $PROJECT_DIR `
    --human_pose_folder data/GVHMR/dataset_locomotion/npy `
    --foot_contact_threshold 0.05 `
    --ground_threshold 0.12 `
    --chunk_size 999999 `
    --chunk_overlap 0 `
    --chunk_min_frames 1 `
    --visualize 1

python src/utils/setup_humanoid.py `
    --input asset\humanoid_model\k1\k1.xml `
    --humanoid_type k1

python src/retarget/shape_adaptation.py `
    --project_dir $PROJECT_DIR `
    --robot_name k1

$human_pose_preprocessed_file = "locomotionGVHMR_npy\turning_CC_slow_chunk_0000"

python src/retarget/motion_adaptation.py `
    --project_dir $PROJECT_DIR `
    --robot_name k1 `
    --human_pose_file $human_pose_preprocessed_file `
    --device cuda:0 `
    --visualize 1

$PROJECT_DIR = "D:\Code\PHUMA"
$human_pose_preprocessed_folder = "data\human_pose_preprocessed\dataset_locomotion"
python src/retarget/motion_adaptation_multiprocess.py `
    --project_dir $PROJECT_DIR `
    --robot_name k1 `
    --human_pose_folder $human_pose_preprocessed_folder `
    --grounding_weight 30 --skating_weight 0.05 --num_iter_dof 5001 `
    --gpu_ids 0 `
    --processes_per_gpu 4 `
    --visualize 1

$PROJECT_DIR = "D:\Code\PHUMA"
    python src/retarget/motion_adaptation.py --project_dir $PROJECT_DIR --robot_name k1 `
    --human_pose_file ..\GVHMR\turning_joe_2\npy\turn_fast_CC --device cpu --visualize 1 `
    --grounding_weight 30 --skating_weight 0.05 --num_iter_dof 5001
