import numpy as np
from utils.visualize import render_robot_pose, write_video

ROBOT = 'asset/humanoid_model/k1'
src = 'data/humanoid_pose/k1/dataset_turning_joe_3/turning_joe_3_mirrored.npy'
out = 'data/video/humanoid_pose/k1/dataset_turning_joe_3/turning_joe_3_mirrored.mp4'

d = np.load(src, allow_pickle=True).item()
frames = render_robot_pose(ROBOT, d['dof_pos'], d['root_trans'], d['root_ori'])
write_video(out, frames, fps=int(d.get('fps', 30)), reverse_rgb=True)
print('wrote', out, '|', len(frames), 'frames')
