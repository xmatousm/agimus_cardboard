# %% Common packages, set-up

import os

if os.path.exists('./startup_local.py'):
    import startup_local  # noqa # pylint: disable=unused-import

data_base = os.environ['DATA_BASE']

import matplotlib.pyplot as plt
import cv2
import xmmlib.graph.fig as fig
import yaml
import agimus_cardboard.crbtools as crb
import numpy as np
import geometry.basic as gb
import agimus_cardboard.draw as draw

plt.ion()
plt.close('all')

tmpl_file = '../templates/template_Z.yml'

# Workplace 2, camera on a ceiling above the robot
# img_dir, file0 = data_base + 'calib/2025-11-28/', '02.png'
# img_dir, file0 = 'data/calib/2025-11-28/', 'zz.png'

# Workplace 1, camera on a bar joined to the robot table [2026-05-25]
# img_dir, file0 = 'data/calib/2026-05-25/', '00.png'

# Workplace 1, side-camera on a bar joined to the robot table [2026-08-25]
img_dir, file0 = data_base + 'calib/2026-08-25/', '00.png'

# %% prepare board

fig.mxfig((2, 2), 1, 1, 'board')
plt.clf()
fig.figname(plt.gcf())

board, board_points = crb.prepare_aruco_board(plot=True)

# %% detect markers
img_files = sorted([img_dir + f for f in os.listdir(img_dir)])

img0 = None
dp0 = None

object_points = []
detected_points = []

n = -1
for ip in img_files:
    img = cv2.imread(ip)
    if img is None:
        print(f'Could not read {ip}, skipping')
        continue
    n += 1
    print(n, ip)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    op, dp = crb.detect_markers(gray, board, board_points, plot=False)

    object_points.append(op)
    detected_points.append(dp)

    if ip.endswith(file0):
        inx0 = n
        img0 = img
        dp0 = dp
        op0 = op

h, w = img0.shape[:2]

fig.mxfig((2, 2), 1, 2, f'detect{n}')
plt.clf()
fig.figname(plt.gcf())
plt.imshow(img0)
plt.plot(dp0[:, 0], dp0[:, 1], 'rx')

# %% calibrate camera

flags = (0
         + cv2.CALIB_FIX_ASPECT_RATIO
         + cv2.CALIB_ZERO_TANGENT_DIST
         + cv2.CALIB_FIX_K3
         + cv2.CALIB_FIX_K2
         )

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=object_points,
    imagePoints=detected_points,
    imageSize=(w, h),
    cameraMatrix=None,
    distCoeffs=None,
    flags=flags,
    #    criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-9)
)

im_sz = [img0.shape[0], img0.shape[1]]
print(f"Image size: {im_sz[0]} x {im_sz[1]}")

print(f"RMS: {ret}")
print(
    f"fx: {mtx[0, 0]}, fy: {mtx[1, 1]}, dx: {mtx[0, 2]}, dy: {mtx[1, 2]}, q: {mtx[0, 1]}")
print(f"rd: {dist}")
print(f"rot: {rvecs[inx0]}")
print(
    f"center difference: {mtx[0, 2] - (im_sz[1] / 2)}, {mtx[1, 2] - (im_sz[0] / 2)}")

plt.plot(mtx[0, 2], mtx[1, 2], 'gx', markersize=20, markeredgewidth=2)

# %% write to yaml
calib = crb.Calib(w=w, h=h)
calib.mat_k = mtx
calib.dist = dist
calib.r_vec = rvecs[inx0]
calib.t_vec = tvecs[inx0]

with open('calib.yml', 'w') as fh:
    calib_data = calib.to_dict()
    yaml.safe_dump(calib_data, fh)

calib_orig = calib
# %% undistort
with open('calib.yml', 'r') as fh:
    calib_data = yaml.load(fh, Loader=yaml.SafeLoader)

calib = crb.Calib.from_dict(calib_data)
calib_u = calib.get_undistorted()

img0_u = calib.transform_image_to(calib_u, img0)

plt.imsave('calib_undistort.png', img0_u, cmap='gray')

dp0_u = calib.transform_points_to(calib_u, dp0)
op0_u = calib_u.transform_3d_points_to(op0)

new_c = np.array([[calib_u.mat_k[0, 2]], [calib_u.mat_k[1, 2]]])
if calib_u.mat_h is not None:
    new_c = gb.p2e(calib_u.mat_h @ gb.e2p(new_c))

fig.mxfig((2, 2), 2, 2, 'undistort')
plt.clf()
fig.figname(plt.gcf())
plt.imshow(img0_u)
plt.plot(new_c[0], new_c[1], 'gx', markersize=20, markeredgewidth=2)

plt.plot(dp0_u[:, 0], dp0_u[:, 1], 'rx', markersize=20, markeredgewidth=2)
plt.plot(op0_u[:, 0], op0_u[:, 1], 'o', markersize=10,
         markerfacecolor='none', markeredgecolor=(0, 1, 0), markeredgewidth=2)

# %%

with open(tmpl_file, 'r') as fh:
    tmpl_dict = yaml.load(fh, Loader=yaml.SafeLoader)
tmpl_m = crb.TemplateMetric.from_dict(tmpl_dict)

tmpl = crb.Template.from_metric(calib_u, crb.Opt(), tmpl_m, fix_origin=False)

draw.template(tmpl, plt.gca())

# %%
