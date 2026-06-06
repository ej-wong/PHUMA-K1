"""Create a sagittal-mirror (left<->right) copy of a retargeted K1 clip.

A mirror across the world X-Z plane (negate Y) reverses turn direction (CCW<->CW).
For the body we swap left/right joints and apply a per-joint sign. The sign map is
NOT assumed from axis labels (arm joints have rotated local frames) -- it is derived
empirically with forward kinematics and verified: FK(mirror(pose)) must equal the
world reflection of FK(pose) with left/right bodies swapped.
"""
import numpy as np, yaml, mujoco, glob, os

ROBOT = 'asset/humanoid_model/k1'
SRC = max(glob.glob('data/humanoid_pose/k1/dataset_turning_joe_3/**/*.npy', recursive=True), key=os.path.getmtime)
DST = os.path.join(os.path.dirname(SRC), 'turning_joe_3_mirrored.npy')

cfg = yaml.safe_load(open(os.path.join(ROBOT, 'config.yaml')))
jnames = cfg['joint_names'][1:]          # 22, dof_pos column order
bnames = cfg['body_names']               # 29
m = mujoco.MjModel.from_xml_path(os.path.join(ROBOT, 'custom.xml'))

def swap(n):
    for A, B in (('Left', 'Right'), ('left', 'right')):
        if A in n: return n.replace(A, B)
        if B in n: return n.replace(B, A)
    return n

partner = np.array([jnames.index(swap(n)) for n in jnames])     # dof column -> mirror column
bsig = [m.body(n).id for n in bnames]                            # body id per config body
bswap = [m.body(swap(n)).id for n in bnames]                    # mirrored body id
M = np.array([1.0, -1.0, 1.0])                                   # reflect across X-Z plane (negate Y)

def fk_xpos(root_pos, root_xyzw, dof):
    d = mujoco.MjData(m)
    q = np.zeros(m.nq)
    q[0:3] = root_pos
    q[3:7] = root_xyzw[[3, 0, 1, 2]]      # xyzw -> wxyz
    q[7:7+22] = dof
    d.qpos = q; mujoco.mj_forward(m, d)
    return d.xpos.copy()

ident = np.array([0.0, 0.0, 0.0, 1.0])

# ---- derive per-joint sign empirically (perturb one joint, check the mirror) ----
sign = np.zeros(22)
for k in range(22):
    th = 0.4
    d0 = np.zeros(22); d0[k] = th
    xo = fk_xpos(np.zeros(3), ident, d0)
    best, berr = 1.0, 1e9
    for s in (1.0, -1.0):
        dm = np.zeros(22); dm[partner[k]] = s * th
        xm = fk_xpos(np.zeros(3), ident, dm)
        err = np.max(np.abs(xm[bswap] - M * xo[bsig]))
        if err < berr: berr, best = err, s
    sign[k] = best

print('per-joint mirror sign (dof column order):')
for k in range(22):
    print('  %-24s -> %-24s  sign %+d' % (jnames[k], jnames[partner[k]], int(sign[k])))

# ---- verify on random full poses ----
lo = np.array([m.jnt_range[m.joint(n).id][0] for n in jnames])
hi = np.array([m.jnt_range[m.joint(n).id][1] for n in jnames])
rng = np.random.default_rng(0); worst = 0.0
for _ in range(200):
    dof = rng.uniform(lo, hi)
    rp = rng.uniform(-1, 1, 3)
    rq = rng.normal(size=4); rq /= np.linalg.norm(rq)
    xo = fk_xpos(rp, rq, dof)
    dofm = sign * dof[partner]
    rpm = M * rp
    rqm = np.array([-rq[0], rq[1], -rq[2], rq[3]])
    xm = fk_xpos(rpm, rqm, dofm)
    worst = max(worst, np.max(np.abs(xm[bswap] - M * xo[bsig])))
print('\nrandom-pose mirror verification: max body-position error = %.2e m' % worst)
assert worst < 1e-3, 'sign map FAILED verification'

# ---- apply to the trajectory ----
d = np.load(SRC, allow_pickle=True).item()
rt, ro, dp = d['root_trans'], d['root_ori'], d['dof_pos']
rt_m = rt * M
ro_m = ro * np.array([-1.0, 1.0, -1.0, 1.0])    # xyzw: (-x, y, -z, w)
dp_m = sign[None, :] * dp[:, partner]

# verify the whole trajectory mirrors exactly
worst_traj = 0.0
for i in range(0, len(dp), 7):
    xo = fk_xpos(rt[i], ro[i], dp[i])
    xm = fk_xpos(rt_m[i], ro_m[i], dp_m[i])
    worst_traj = max(worst_traj, np.max(np.abs(xm[bswap] - M * xo[bsig])))
print('trajectory mirror verification: max body-position error = %.2e m' % worst_traj)
assert worst_traj < 1e-3

np.save(DST, {'root_trans': rt_m.astype(rt.dtype), 'root_ori': ro_m.astype(ro.dtype),
              'dof_pos': dp_m.astype(dp.dtype), 'fps': d.get('fps', 30)})
print('\nsaved mirrored clip -> %s  (%d frames)' % (DST, len(dp)))
