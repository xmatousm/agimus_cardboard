# %% Common packages, set-up
import os

from scripts.calib_robot import img_file

if os.path.exists('./startup_local.py'):
    import startup_local  # noqa # pylint: disable=unused-import

data_base = os.environ['DATA_BASE']

import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import xmmlib.graph.fig as fig
import agimus_cardboard.crbtools as crb
import agimus_cardboard.draw as draw
import beartype

from xmmlib.tools import tic, toc

opt = crb.Opt()

plt.ion()
plt.close('all')

calib_file = data_base + 'calib/calib_cam_2.yaml'
mask_file = data_base + 'calib/mask_2.png'

img_dir = data_base + 'cardboard/2026-07-28/10/'
tmpl_file = '../templates/template_10.yml'

img_files = sorted([img_dir+f for f in os.listdir(img_dir)])

with open(calib_file, 'r') as fh:
    calib_data = yaml.load(fh, Loader=yaml.SafeLoader)


calib = crb.Calib.from_dict(calib_data)
calib_u = calib.get_undistorted()

mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
assert mask.shape == (calib.h, calib.w)
mask = (mask > 0).astype(np.uint8)


def prepare_fig(*args, **kwargs) -> plt.Figure:
    f = fig.mxfig(*args, **kwargs)
    fig.figname(plt.gcf())
    plt.clf()
    return f

# %% load the template
prepare_fig((3, 5), 3, 2, 'template_metric')

with open(tmpl_file, 'r') as fh:
    tmpl_dict = yaml.load(fh, Loader=yaml.SafeLoader)
tmpl_m = crb.TemplateMetric.from_dict(tmpl_dict)

draw.template_metric(tmpl_m, plt.gca())

prepare_fig((3, 5), 3, 1, 'template')

tmpl = crb.Template.from_metric(calib_u, opt, tmpl_m)

crb.opt_update(opt, tmpl)
draw.template(tmpl, plt.gca())

# %%
fd = prepare_fig((2, 5), 1, 3, 'detect', row_span=2, col_span=3)

for img_file in img_files:
    print(img_file)
# %%
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img * mask

    img_u = calib.transform_image_to(calib_u, img)

    fd.clf()
    plt.figure(fd)
    plt.imshow(img_u, cmap='gray')
    ax_detect = plt.gca()
    plt.tight_layout()

    tic('segments')
    seg_u, img_e, u_e = crb.detect_all_segments(img_u, opt, ax_detect=ax_detect)
    draw.segments(seg_u, ax_detect, linewidth=1, color='c')
    toc('segments')
    print(f'Segments: {len(seg_u)}')

    tic('pairs')
    pairs = crb.segment_pairs(seg_u, opt)
    toc('pairs')
    print(f'Pairs: {len(pairs)}')

    tic('match')
    rot, t, n_inl = crb.match_pairs(pairs, tmpl.pairs, seg_u, tmpl.seg, opt)
    toc('match')

    dq = crb.nearest_points_lines(tmpl.seg, u_e, rot, t, opt.icp['thr'])[2]
    print(f'Inl: {len(dq)}, rms: {np.sqrt(dq.mean())}')

    tic('icp')
    rot1, t1 = crb.icp_points_lines(tmpl.seg, u_e, rot, t, opt)
    toc('icp')

    u_ref, u_ok, dq = crb.nearest_points_lines(tmpl.seg, u_e, rot1, t1,
                                               opt.icp['thr'])

    print(f'Inl: {len(dq)}, rms: {np.sqrt(dq.mean())}')

    x_lim, y_lim = draw.template(tmpl, rot=rot1, t=t1, points=False)

    u_ref = rot1@u_ref+t1

    rng = np.percentile(img_u, [opt.diff['perc_min'], opt.diff['perc_max']])
    lines, ids, filled = tmpl.check_holes(img_u, rot1, t1, opt, rng)

    for l, f in zip(lines, filled):
        ax_detect.plot(l[0], l[1], '--',
                 color=[0.2, 1.0, 0.0] if f else [1.0, 0.0, 0.5 ], linewidth=2)

    ax_detect.set_xlim((x_lim[0] - 10, x_lim[1] + 10))
    ax_detect.set_ylim((y_lim[0] - 10, y_lim[1] + 10))

    outfile = 'work/' + img_file[len(data_base):-4] + '_detections.png'

    p, _ = os.path.split(outfile)
    os.makedirs(p, exist_ok=True)
    fd.savefig(outfile)


# %%

