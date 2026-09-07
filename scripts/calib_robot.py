# %% Common packages, set-up
import os

if os.path.exists('./startup_local.py'):
    import startup_local  # noqa # pylint: disable=unused-import

data_base = os.environ['DATA_BASE']

import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2

import agimus_cardboard.crbtools as crb
import geometry.basic as gb

opt = crb.Opt()

plt.ion()
plt.close('all')

img_file = data_base + 'calib/calib_robot_1.png'
calib_file = data_base + 'calib/calib_cam_1.yaml'

# img_file = 'data/calib/calib_robot_2.png'
# calib_file = 'data/calib/calib_cam_2.yaml'

# points in the robot coordinate frame
x = np.array([[0.45, 0.45, 0.70, 0.70],
              [0.25, -0.25, -0.25, 0.25],
              [0.095, 0.095, 0.095, 0.095]])


# %% camera calibration
with open(calib_file, 'r') as fh:
    calib_data = yaml.load(fh, Loader=yaml.SafeLoader)

calib = crb.Calib.from_dict(calib_data)
calib_u = calib.get_undistorted()
# %% points

plt.figure(1)
plt.clf()
img = cv2.imread(img_file)
plt.imshow(img)
ax = plt.gca()

# points in the raw camera image
pts_file = img_file +'_points.yaml'

if not os.path.exists(pts_file):
    xl = ax.get_xlim()
    yl = ax.get_ylim()

    u = []
    for i in range(x.shape[1]):
        msg = f'{i}: Click on point [{x[0, i]}, {x[1, i]}]'
        print(msg)
        ax.set_title(msg)
        plt.pause(0.01)
        xin = plt.ginput(1)[0]
        print(xin)
        ax.set_xlim(xin[0]-50, xin[0]+50)
        ax.set_ylim(xin[1]-50, xin[1]+50)
        plt.pause(0.01)
        ax.set_title('Refine...')
        xin = plt.ginput(1)[0]
        print(xin)

        u += [xin[0], xin[1]]

        ax.set_xlim(xl)
        ax.set_ylim(yl)

    u = [int(x) for x in u]
    with open(pts_file, 'w') as fh:
        yaml.safe_dump(u, fh)

with open(pts_file, 'r') as fh:
    u = yaml.safe_load(fh)

u = np.array(u).reshape(-1, 2).T

plt.plot(u[0], u[1], 'rx', markersize=10)

# %%
plt.figure(2)
plt.clf()

img_u = calib.transform_image_to(calib_u, img)
u_u = calib.transform_points_to(calib_u, u.astype(float).T).T
plt.imshow(img_u)

plt.plot(u_u[0], u_u[1], 'rx', markersize=10)

# calibrate rigid motion with scale
R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

rot, trn, scale = crb.abs_ori(R0@gb.e2p(u_u), x, True)
rot = rot@R0

x1 = scale*rot@gb.e2p(u_u)+trn
err = np.sqrt(((x-x1)**2).sum(axis=0))
print(err)
print(f'RMS: {np.sqrt((err**2).mean())}')
aa = gb.r2aa(rot)
calib_robot = {
    'rot_vec': [float(x) for x in aa[0]*aa[1]],
    'trn': [float(x) for x in trn.flatten()],
    'scale': float(scale),
}

with open('calib_robot.yaml', 'w') as fh:
    yaml.safe_dump(calib_robot, fh)
