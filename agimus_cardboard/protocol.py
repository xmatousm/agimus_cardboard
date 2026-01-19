# %% Common packages, set-up
import os

if os.path.exists('./startup_local.py'):
    import startup_local  # noqa # pylint: disable=unused-import

import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import xmmlib.fig as fig
import crbtools as crb
import draw

from xmmlib.tools import tic, toc

opt = crb.Opt()

plt.ion()
plt.close('all')
img_dir = 'data/cardboard/2025-11-28/'
img_files = sorted([img_dir + f for f in os.listdir(img_dir)])

with open('calib.yml', 'r') as fh:
    calib_data = yaml.load(fh, Loader=yaml.SafeLoader)

calib = crb.Calib.from_dict(calib_data)
calib_u = calib.get_undistorted()

# %% detect segments on the template
img_t = cv2.imread('template.png')
img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)

img_u = img_t.copy()

hole_t = []

for i in range(20):
    sel = (img_u >= (10 + 10 * i)) * (img_u < (20 + 10 * i))
    x = np.nonzero(sel)
    if len(x[0]) > 0:
        hole_t.append([x[1], x[0]])

img_u[img_u > 10] = 255

fig.mxfig((3, 5), 1, 1, 'template')
plt.clf()
plt.imshow(img_u, cmap='gray')
for h in hole_t:
    plt.plot(h[0], h[1], '.')

seg_template = crb.detect_all_segments(img_u, opt)[0]

draw.segments(seg_template)

print(f'Template segments: {len(seg_template)}')

template_pairs = crb.segment_pairs(seg_template, opt)
print(f'Template Pairs: {len(template_pairs)}')


# %%
if False:
    for i in range(len(img_files)):
        f = img_files[i]
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_u = calib.transform_image_to(calib_u, img)
        img_u = cv2.warpPerspective(img_u, mat_h, (new_w, new_h))

        r = (i + 1) // 5 + 1
        c = (i + 1) % 5 + 1
        fig.mxfig((3, 5), r, c, f'detect{i}')
        plt.clf()
        plt.imshow(img_u, cmap='gray')
        ax_detect = plt.gca()

        tic('total')
        seg_u = crb.detect_all_segments(img_u, opt, ax_detect=ax_detect)
        toc('total')

# %%
f = img_files[1]  # 12
img = cv2.imread(f)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_u = calib.transform_image_to(calib_u, img)

fd = fig.mxfig((3, 5), 1, 2, f'detect', row_span=3,col_span=4)
plt.clf()
plt.imshow(img_u, cmap='gray')
ax_detect = plt.gca()

tic('segments')

seg_u, _, u_e = crb.detect_all_segments(img_u, opt, ax_detect=ax_detect)
draw.segments(seg_u, ax_detect)

toc('segments')
print(f'Segments: {len(seg_u)}')

# %%
tic('pairs')
pairs = crb.segment_pairs(seg_u, opt)
toc('pairs')
print(f'Pairs: {len(pairs)}')

# %%
tic('match')
rot, t = crb.match_pairs(pairs, template_pairs, seg_u, seg_template, opt)
toc('match')

draw.segments(seg_template, rot=rot, t=t, linewidth=2, color='w')

# %%
fig.mxfig((3, 5), 2, 1, 'warped_image')
warp_mat = np.hstack((rot.T, -rot.T @ t))
img_w = cv2.warpAffine(img_u, warp_mat, (img_t.shape[1], img_t.shape[0]))
plt.imshow(img_w, cmap='gray')
for h in hole_t:
    plt.plot(h[0], h[1], '.')

# %%
tic('icp')
u_ok, u_ref = crb.icp_points_lines(seg_template, u_e, rot, t)
toc('icp')

ax_detect.plot(
    np.vstack((u_ok[0], u_ref[0])),
    np.vstack((u_ok[1], u_ref[1])),'y')

#ax_detect.plot(u_ref[0], u_ref[1],'y.')


# %%

fd.savefig("detections.png")
plt.imsave("warp.png", img_w, cmap='gray')

