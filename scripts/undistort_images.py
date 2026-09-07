#!/usr/bin/env python3
"""Batch-undistort images using the calibration produced by calib.py."""

import os.path
import sys

if os.path.exists('./startup_local.py'):
    import startup_local  # noqa # pylint: disable=unused-import

data_base = os.environ['DATA_BASE']

import matplotlib.pyplot as plt
import cv2
import yaml
import agimus_cardboard.crbtools as crb

plt.ion()
plt.close('all')

img_dir = data_base + 'pieces/2026-08-26/side/'
calib_file = data_base + 'calib/calib_cam_1_side.yaml'

# img_dir = data_base + 'pieces/2026-08-26/top/'
# calib_file = data_base + 'calib/calib_cam_1.yaml'

with open(calib_file, 'r') as fh:
    calib_data = yaml.load(fh, Loader=yaml.SafeLoader)

calib = crb.Calib.from_dict(calib_data)
calib_u = calib.get_undistorted()
calib_u.mat_h = None

img_files = sorted([img_dir + f for f in os.listdir(img_dir)])

for ip in img_files:
    img = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'Could not read {ip}, skipping')
        continue
    print(ip)

    h, w = img.shape[:2]
    if (w, h) != (calib.w, calib.h):
        print(f'size mismatch, skipping')
        continue

    img_u = calib.transform_image_to(calib_u, img)
    print(img.shape)
    print(img_u.shape)


    op = 'work/' + ip[len(data_base):-4] + '_undistort.png'
    p, _= os.path.split(op)
    os.makedirs(p, exist_ok=True)
    if not cv2.imwrite(op, img_u):
        print(f"could not write: {op}")
        sys.exit(255)

with open(p + '/calib_u.yml', 'w') as fh:
    calib_data = calib_u.to_dict()
    yaml.safe_dump(calib_data, fh)
