"""Reground the floating intro in a raw GVHMR .npy by removing the low-frequency
vertical drift. Computes the planted-foot height per frame (rolling-min envelope of
the lowest foot vertex, above the mode-based floor), subtracts a smoothed version so
the planted foot returns to the floor through the drifted intro, and tapers to zero
once the clip settles (real foot lifts and the good part are preserved).

Dry run by default; pass --apply to write (backs up to .bak first).
"""
import numpy as np, torch, yaml, smplx, shutil, sys
from scipy.ndimage import minimum_filter1d, gaussian_filter1d
from utils.smpl import find_robust_ground

APPLY = '--apply' in sys.argv
gv = 'data/GVHMR/dataset_turning_joe_3/npy/turning_joe_3.npy'

cfg = yaml.safe_load(open('asset/human_model/config.yaml'))
fcv = {k: list(cfg[k]) for k in ['left_toe_indices', 'left_heel_indices', 'right_toe_indices', 'right_heel_indices']}
sm = smplx.create('asset/human_model', model_type='smplx', num_pca_comps=45)

# reduced foot-vertex buffer (full mesh would be ~0.4 GB)
allidx = sorted(set(sum(fcv.values(), [])))
remap = {v: i for i, v in enumerate(allidx)}
fcv_r = {k: [remap[x] for x in fcv[k]] for k in fcv}

a = np.load(gv); N = len(a); dt = a.dtype
transl = a[:, 0:3].astype(np.float64); go = a[:, 3:6]; bp = a[:, 6:69]
Vred = np.zeros((N, len(allidx), 3)); lowfoot = np.zeros(N)
with torch.no_grad():
    for i in range(N):
        v = sm(transl=torch.tensor(transl[i:i+1]).float(),
               global_orient=torch.tensor(go[i:i+1]).float(),
               body_pose=torch.tensor(bp[i:i+1]).float()).vertices[0].numpy()
        Vred[i] = v[allidx]; lowfoot[i] = v[allidx, 1].min()

floor = find_robust_ground(Vred, fcv_r)                              # pipeline's mode-based floor
env = minimum_filter1d(lowfoot - floor, size=61, mode='nearest')     # planted-foot height above floor
above = np.where(env > 0.02)[0]
end = int(above.max()) if len(above) else 0                         # last frame planted foot is >2cm up
o = np.clip(gaussian_filter1d(np.clip(env, 0, None), sigma=10), 0, None)
mask = np.ones(N)
if end + 90 < N:
    mask[end:end+90] = np.linspace(1, 0, 90); mask[end+90:] = 0
o = o * mask

after = lowfoot - o
print('raw floor (mode) = %.3f m   intro ends ~frame %d (%.1fs)   max drift = %.1f cm' % (floor, end, end/30.0, o.max()*100))
fr = [0, 30, 60, 150, 300, 450, 586, 800, 1200, 1959, 2500]
print('frame:         ' + '  '.join('%5d' % f for f in fr if f < N))
print('lowfoot BEFORE:' + '  '.join('%5.1f' % ((lowfoot[f]-floor)*100) for f in fr if f < N) + '   (cm above floor)')
print('lowfoot AFTER :' + '  '.join('%5.1f' % ((after[f]-floor)*100) for f in fr if f < N) + '   (cm above floor)')
print('good-part disturbance (max drift after frame %d): %.2f cm  (want ~0)' % (end+90, o[end+90:].max()*100 if end+90 < N else 0))

if APPLY:
    shutil.copy(gv, gv + '.bak')
    out = a.copy(); out[:, 1] = (transl[:, 1] - o).astype(dt)
    np.save(gv, out)
    print('APPLIED -> %s   (backup: %s.bak)' % (gv, gv))
else:
    print('DRY RUN — nothing written. Re-run with --apply to save.')
