# %% Common packages, set-up
import os

if os.path.exists('./startup_local.py'):
    import startup_local  # noqa # pylint: disable=unused-import

data_base = os.environ['DATA_BASE']


import beartype   # noqa # pylint: disable=unused-import

import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import xmmlib.graph.fig as fig
import agimus_cardboard.crbtools as crb
import agimus_cardboard.draw as draw

from xmmlib.tools import tic, toc

opt = crb.Opt()

plt.ion()
plt.close('all')

# img_dir = 'data/cardboard/2026-01-06/'
# img_dir = 'data/cardboard/2026-01-06b/'
# img_dir = 'data/cardboard/2026-01-08c/'
# tmpl_file = 'templates/template_2.yml'

# img_dir = 'data/cardboard/2026-05-25/'
# img_dir = 'data/cardboard/2026-05-25-slant/'

# tmpl_file = 'templates/template_2.yml'
# img_file = 'data/cardboard/2026-07-28/2/3.png'

#tmpl_file = 'templates/template_3.yml'
#img_file = 'data/cardboard/2026-07-28/3/3.png'

#tmpl_file = 'templates/template_4.yml'
#img_file = 'data/cardboard/2026-07-28/4/3.png'

tmpl_file = '../templates/template_9.yml'
img_file = data_base + 'cardboard/2026-07-28/9/0.png'


calib_file = data_base + 'calib/calib_cam_2.yaml'
mask_file = data_base + 'calib/mask_2.png'


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

do_cont = True

# %% generate the template
if False:
    img = cv2.imread('templates/template_3.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tmpl0 = crb.Template.from_image(img, opt)

    prepare_fig((3, 5), 1, 1, 'template_img')
    plt.clf()
    plt.imshow(img, cmap='gray')

    draw.template(tmpl0, plt.gca())

    print(f'Template segments: {len(tmpl0.seg)}')
    print(f'Template Pairs: {len(tmpl0.pairs)}')

    tmpl_m = tmpl0.to_metric(calib_u)
    tmpl_dict = tmpl_m.to_dict()

    with open('template_new.yml', 'w') as fh:
        yaml.safe_dump(tmpl_dict, fh)

# %% load the template
if True:
    prepare_fig((3, 5), 3, 2, 'template_metric')

    with open(tmpl_file, 'r') as fh:
        tmpl_dict = yaml.load(fh, Loader=yaml.SafeLoader)
    tmpl_m = crb.TemplateMetric.from_dict(tmpl_dict)

    draw.template_metric(tmpl_m, plt.gca())

    prepare_fig((3, 5), 3, 1, 'template')

    tmpl = crb.Template.from_metric(calib_u, opt, tmpl_m)
    crb.opt_update(opt, tmpl)

    draw.template(tmpl, plt.gca())


# do_cont = False

# %%
if True and do_cont:
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img * mask

    img_u = calib.transform_image_to(calib_u, img)

    fd = prepare_fig((2, 5), 1, 3, 'detect', row_span=2, col_span=3)

    plt.imshow(img_u, cmap='gray')
    ax_detect = plt.gca()
    plt.tight_layout()
    tic('segments')

    seg_u, img_e, u_e = crb.detect_all_segments(img_u, opt, ax_detect=ax_detect)
    draw.segments(seg_u, ax_detect)

    toc('segments')
    print(f'Segments: {len(seg_u)}')
# %%
if True and do_cont:
    tic('pairs')
    pairs = crb.segment_pairs(seg_u, opt)
    toc('pairs')
    print(f'Pairs: {len(pairs)}')

# %%
if True and do_cont:
    tic('match')
    rot, t, n_inl = crb.match_pairs(pairs, tmpl.pairs, seg_u, tmpl.seg, opt)
    toc('match')

    draw.segments(tmpl.seg, rot=rot, t=t, linewidth=2, color='m')

    dq = crb.nearest_points_lines(tmpl.seg, u_e, rot, t, opt.icp['thr'])[2]

    print(f'Inl: {len(dq)}, rms: {np.sqrt(dq.mean())}')

    prepare_fig((3, 5), 1, 2, 'hist1')

    d = np.sqrt(dq)
    plt.hist(d)

# %%
if True and do_cont:
    tic('icp')
    rot1, t1 = crb.icp_points_lines(tmpl.seg, u_e, rot, t, opt)
    toc('icp')

    u_ref, u_ok, dq = crb.nearest_points_lines(tmpl.seg, u_e, rot1, t1,
                                               opt.icp['thr'])

    print(f'Inl: {len(dq)}, rms: {np.sqrt(dq.mean())}')

    plt.sca(ax_detect)

    u_ref = rot1@u_ref+t1

    draw.template(tmpl, rot=rot1, t=t1, points=0)
    # ax_detect.plot(
    #    np.vstack((u_ok[0], u_ref[0])),
    #    np.vstack((u_ok[1], u_ref[1])),'y')

    prepare_fig((3, 5), 2, 2, 'hist2')
    d = np.sqrt(dq)
    plt.hist(d)

# %%
if True and do_cont:
    prepare_fig((3, 5), 2, 1, 'warped_image')

    warp_mat = np.hstack((rot1.T, -rot1.T@t1))
    img_w = cv2.warpAffine(img_u, warp_mat,
                           (tmpl.img_shape[1], tmpl.img_shape[0]))
    plt.imshow(img_w, cmap='gray')
    for h in tmpl.hole_pt:
        plt.plot(h[0], h[1], '.')

    plt.figure(fd)
    warp_mat = np.hstack((rot1, t1))
    if False:
        for i in range(len(tmpl.hole_pt)):
            h = (rot1@tmpl.hole_pt[i]+t1).astype(int)
            plt.plot(h[0], h[1], '.')
            h = (rot1@tmpl.nb_hole_pt[i]+t1).astype(int)
            plt.plot(h[0], h[1], '.')

    plt.imsave("warp.png", img_w, cmap='gray')

# %%
if True and do_cont:
    rng = np.percentile(img_u, [opt.diff['perc_min'], opt.diff['perc_max']])
    lines, ids, filled = tmpl.check_holes(img_u, rot1, t1, opt, rng)

    plt.figure(fd)
    for l, f in zip(lines, filled):
        plt.plot(l[0], l[1], '--',
                 color=[0.2, 1.0, 0.0] if f else [1.0, 0.0, 0.5 ], linewidth=2)

# %%

