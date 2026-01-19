import time
from dataclasses import dataclass
from typing import Optional, Any

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np

import geometry.all as g
from mathlib.optimize import ransac
from mathlib.types import VectorI, ColVector3, RowVector, Matrix33


@dataclass()
class Opt:
    template = {
        # width of the hole neighborhood area
        'neigh': 11,
    }

    canny = {
        # 'threshold1': 1580,
        # 'threshold2': 1580,
        # 'apertureSize': 5,
        'threshold1': 150,
        'threshold2': 110,
        'apertureSize': 3,
    }

    use_hough_p = True

    hough_p = {
        'rho': 0.2,
        'theta': np.pi / 360,
        'threshold': 10,
        'minLineLength': 100,
        'maxLineGap': 20,
    }

    hough = {
        'rho': 0.5,
        'theta': np.pi / 360,
        'threshold': 50,
    }

    segment = {
        'thr0': 10,
        'thr': 1.5,
        'log_prob': 4,
        'max_gap': 10,
        'min_length': 150,
        # maximum count of line segments
        'max_segments': 30,
    }

    match = {
        # minimum angle forming the 'intersection' type pair
        'min_join_angle': 30.0 / 180.0 * np.pi,
        # maximum angle forming the 'parallel' type pair
        'max_parallel_angle': 3.0 / 180.0 * np.pi,

        # max difference of angles for pairs to be matched
        'max_join_angle_diff': 3.0 / 180.0 * np.pi,

        # maximum point to segment distance
        'max_seg_distance': 10.0,

        # minimum matched segments to immediately accept
        'max_match': 5,

        # minimum matched segments to accept
        'min_match': 3,

    }

    icp = {
        # threshold for point to belongs to the nearest line
        'thr': 15,

        # fixed number of ICP iterations
        'iter': 5,
    }

    diff = {
        # percentiles for the actual intensity range
        'perc_min': 10,
        'perc_max': 90,

        # percentiles for the inside and the outside of a hole
        'perc_in': 50,
        'perc_out': 70,

        # threshold for inside/outside difference to be a hole
        'thr': 0.2,
    }


class LineSegment:
    def __init__(self, u1, u2):
        self.u1 = u1
        self.u2 = u2

        # line
        self.l = g.u2l_norm(u1, u2)

        # nearest point transformation
        # u_nearest = (I - n @ n.T ) @ u - n * d
        self.pt_mat = np.eye(2) - self.l[:2] @ self.l[:2].T
        self.pt_vec = - self.l[:2] * self.l[2, 0]

        # direction vector and length
        self.v = g.vnz(u2 - u1)
        self.len = g.vlen(u2 - u1)

        # t-range along the segment
        self.t1 = (self.v.T @ u1)[0, 0]
        self.t2 = (self.v.T @ u2)[0, 0]


class Calib():
    mat_k: Matrix33

    def __init__(self, w=None, h=None):
        self.mat_k = np.eye(3)
        self.dist: Optional[RowVector] = None
        self.r_vec: Optional[ColVector3] = None
        self.t_vec: Optional[ColVector3] = None
        self.w: int = w
        self.h: int = h
        self.mat_h: Optional[Matrix33] = None  # homography to the object plane
        self.w_h: Optional[int] = None  # width of the image after homography
        self.h_h: Optional[int] = None  # height of the image after homography

    def to_dict(self):
        """Serialize the instance to a dict using simple types. This is
         suitable, e.g., for saving to a YAML file."""

        assert self.mat_k.shape == (3, 3)

        data = {
            'fx': float(self.mat_k[0, 0]),
            'fy': float(self.mat_k[1, 1]),
            'cx': float(self.mat_k[0, 2]),
            'cy': float(self.mat_k[1, 2]),
            'q': float(self.mat_k[0, 1]),
        }

        if self.w is not None:
            assert self.h is not None
            data['w'] = int(self.w)
            data['h'] = int(self.h)

        if self.dist is not None:
            assert self.dist.shape == (1, 5)
            data['k1'] = float(self.dist[0, 0])
            data['k2'] = float(self.dist[0, 1])
            data['k3'] = float(self.dist[0, 4])
            data['p1'] = float(self.dist[0, 2])
            data['p2'] = float(self.dist[0, 3])

        if self.r_vec is not None:
            assert self.r_vec.shape == (3, 1)
            data['r_vec'] = [float(x) for x in self.r_vec.flatten()]

        if self.t_vec is not None:
            assert self.t_vec.shape == (3, 1)
            data['t_vec'] = [float(x) for x in self.t_vec.flatten()]

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Deserialize a new Calib instance from a dict."""

        def _req(*key, dtype: type = float):
            """Check the presence and types of required keys."""
            for k in key:
                assert k in data, f"missing {k} in the dict"
                assert type(data[k]) is dtype, \
                    f"type of {k} should be {dtype} not {type(data[k])}"

        calib = cls()

        if 'w' in data:
            _req('w', 'h', dtype=int)
            calib.w = data['w']
            calib.w = int(data['w'])
            calib.h = int(data['h'])

        _req('fx', 'fy', 'cx', 'cy', 'q')
        calib.mat_k = np.array([[data['fx'], data['q'], data['cx']],
                                [0., data['fy'], data['cx']],
                                [0., 0., 1.]])

        if 'k1' in data:
            _req('k1', 'k2', 'k3', 'p1', 'p2')

            calib.dist = np.array([[
                data['k1'], data['k2'], data['p1'], data['p2'], data['k3']]])

        if 'r_vec' in data:
            _req('r_vec', 't_vec', dtype=list)
            assert len(data['r_vec']) == 3, "length of r_vec should be 3"
            assert len(data['t_vec']) == 3, "length of t_vec should be 3"

            calib.r_vec = np.array(data['r_vec']).reshape((3, 1))
            calib.t_vec = np.array(data['t_vec']).reshape((3, 1))

        return calib

    def update_homography(self) -> None:
        """Compute homography to the object plane using the internal
         and the external calibration."""
        assert self.r_vec is not None
        rot = g.a2r(self.r_vec.flatten())
        mat_h = self.mat_k @ rot.T @ np.linalg.inv(self.mat_k)
        self.mat_h, self.w_h, self.h_h = im_fit_h(mat_h, self.w, self.h)

    def transform_image_to(self, calib_to: 'Calib', img):
        if calib_to.dist is not None:
            raise NotImplementedError()

        dst = cv2.undistort(img, self.mat_k, self.dist,
                            None, calib_to.mat_k)
        if calib_to.mat_h is not None:
            dst = cv2.warpPerspective(dst, calib_to.mat_h,
                                      (calib_to.w_h, calib_to.h_h))

        return dst

    def transform_points_to(self, calib_to: 'Calib', u):
        if calib_to.dist is not None:
            raise NotImplementedError()

        dst = cv2.undistortImagePoints(u, self.mat_k, self.dist)

        dst = g.e2p(dst[:, 0, :].T)
        dst = calib_to.mat_k @ np.linalg.inv(self.mat_k) @ dst

        if calib_to.mat_h is not None:
            dst = calib_to.mat_h @ dst

        return g.p2e(dst).T

    def get_undistorted(self) -> 'Calib':
        mat_k, roi = cv2.getOptimalNewCameraMatrix(
            self.mat_k, self.dist, (self.w, self.h), 1, (self.w, self.h))
        calib = Calib(w=self.w, h=self.h)
        calib.mat_k = mat_k
        calib.r_vec = self.r_vec
        calib.t_vec = self.t_vec
        calib.update_homography()
        return calib


class Template:
    def __init__(self, image_file: str, opt: Opt) -> None:
        # load a template bitmap
        img = cv2.imread(image_file)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # extract coordinates of the template holes segmented based on intensity
        self.hole_pt = []
        self.nb_hole_pt = []
        self.hole_line = []

        nb_kernel = np.ones((opt.template['neigh'], opt.template['neigh']),
                            np.uint8)

        step = 10
        for i in range(20, 240, step):
            sel = (self.img >= i) * (self.img < (i + step))
            x = np.nonzero(sel)
            if len(x[0]) > 0:
                self.hole_pt.append(np.vstack((x[1], x[0])))

                # line points
                v1 = self.img[sel].min()
                v2 = self.img[sel].max()

                x1 = np.nonzero(self.img == v1)
                x2 = np.nonzero(self.img == v2)

                line = np.array([[x1[1][0], x2[1][0]],
                                           [x1[0][0], x2[0][0]]])
                self.hole_line.append(line)

                # neighbourhood
                sel1 = cv2.dilate(sel.astype(np.uint8), nb_kernel)

                sel1 = sel1.astype(bool)
                sel1 = sel1 * np.bitwise_not(sel)
                x = np.nonzero(sel1)

                self.nb_hole_pt.append(np.vstack((x[1], x[0])))

        # template without holes
        self.img[self.img > 10] = 255

        # template segments and pairs
        self.seg = detect_all_segments(self.img, opt)[0]
        self.pairs = segment_pairs(self.seg, opt)

    def match(self, seg, pairs, u, opt):
        rot, t, n_inl = match_pairs(pairs, self.pairs, seg, self.seg, opt)

        if rot is None:
            return None, None, None, None

        rot1, t1 = icp_points_lines(self.seg, u, rot, t, opt)

        dq = nearest_points_lines(self.seg, u, rot1, t1, opt.icp['thr'])[2]

        return rot1, t1, n_inl, dq

    def check_holes(self, img, rot, t, opt, rng):
        lines = []
        # TODO maybe bettere to compute area of dark pixels in hole and
        #  neighbourhood and compare to area of template hole

        for i in range(len(self.hole_pt)):
            # histogram of values inside and outside the hole
            h = (rot @ self.nb_hole_pt[i] + t).astype(int)
            # TODO check if we are inside
            vals_out = img[h[1], h[0]]

            h = (rot @ self.hole_pt[i] + t).astype(int)
            vals_in = img[h[1], h[0]]

            val_in = np.percentile(vals_in, opt.diff['perc_in'])
            val_out = np.percentile(vals_out, opt.diff['perc_out'])

            val_in = (val_in - rng[0]) / (rng[1] - rng[0])
            val_out = (val_out - rng[0]) / (rng[1] - rng[0])

            val_in = min(1, max(0, val_in))
            val_out = min(1, max(0, val_out))

            df = val_out - val_in

            if df < opt.diff['thr']:
                lines += [rot @ self.hole_line[i] + t]

        return lines

def im_fit_h(mat_h, w, h):
    """Update homography and image sizes."""
    c1 = np.array([[0, w, w, 0], [0, 0, h, h]])
    c2 = g.p2e(mat_h @ g.e2p(c1))

    dx = -c2[0].min()
    dy = -c2[1].min()

    w = int(c2[0].max() - c2[0].min() + 0.5)
    h = int(c2[1].max() - c2[1].min() + 0.5)

    mat_h = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ mat_h
    return mat_h, w, h


def prepare_aruco_board(plot=False):
    """Prepare a checkerboard with aruco markers (the one we have in the lab)."""

    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, 'DICT_4X4_1000'))

    board = aruco.CharucoBoard(
        (10, 14),
        squareLength=0.07,
        markerLength=0.04,
        dictionary=aruco_dict,
    )

    board.setLegacyPattern(True)

    board_points = {}
    for cid, pt in zip(board.getIds(), board.getObjPoints()):
        board_points[cid] = pt.mean(axis=0)
        if plot:
            plt.plot(board_points[cid][0], board_points[cid][1], 'r.')
            plt.text(board_points[cid][0], board_points[cid][1], str(cid))

    return board, board_points


def detect_markers(img_gray, board, board_points, plot=False):
    _, _, corners, ids = aruco.CharucoDetector(board).detectBoard(img_gray)

    dp = []
    op = []

    for cid, c in zip(ids, corners):
        c = c[0].T
        cid = cid[0]

        # intersection of diagonals
        mid = np.cross(
            np.cross(np.hstack((c[:, 0], 1)), np.hstack((c[:, 2], 1))),
            np.cross(np.hstack((c[:, 1], 1)), np.hstack((c[:, 3], 1)))
        )
        mid = mid[:2] / mid[2]

        if cid in board_points:
            if plot:
                plt.plot(c[0], c[1], color='b')
                plt.plot(mid[0], mid[1], 'ro')
                plt.text(mid[0], mid[1], f" {cid}",
                         color='r', fontweight='bold', fontsize=12,
                         horizontalalignment='left')
            dp.append(mid)
            op.append(board_points[cid])
        else:
            if plot:
                plt.plot(c[0], c[1], color='y')
                plt.plot(mid[0], mid[1], 'yo')

    op = np.vstack(op)
    dp = np.vstack(dp).astype(np.float32)

    return op, dp


def edges(img, opt):
    img_e = cv2.Canny(img, **opt.canny)
    y, x = np.nonzero(img_e)
    u_e = np.vstack([x, y])

    return img_e, u_e


def line_segmentsP(img_e, opt, ax_plot=None):
    lines_h = cv2.HoughLinesP(img_e, **opt.hough_p)

    lines = []
    if lines_h is not None:
        lines_h = lines_h.reshape(-1, 4)
        # sort by length, the longest first
        lines_h = sorted(lines_h, reverse=True,
                         key=lambda x: (x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2)

        for lh in lines_h:
            u1 = np.array([lh[0], lh[1]])
            u2 = np.array([lh[2], lh[3]])
            l = g.u2l_norm(u1, u2)
            lines.append(l)

        if ax_plot is not None:
            for i in range(0, len(lines_h)):
                l = lines_h[i]
                ax_plot.plot((l[0], l[2]), (l[1], l[3]), 'r', linewidth=1)

    return lines, lines_h


def line_segments(img_e, opt, ax_plot=None):
    t0 = time.time()
    lines_h = cv2.HoughLines(img_e, **opt.hough)
    # print(time.time() - t0)
    # print(lines_h.shape)

    w = img_e.shape[1]
    h = img_e.shape[0]

    lines_edpts = np.zeros((0, 0))
    lines = np.zeros((0, 0))
    if lines_h is not None:
        lines_edpts = []
        lines = []
        for i in range(0, len(lines_h)):
            rho = lines_h[i][0][0]
            theta = lines_h[i][0][1]

            l = np.array([[np.cos(theta)], [np.sin(theta)], [-rho]])
            u1, u2 = g.cropline(l, 0, w, 0, h)
            lines.append([l[0, 0], l[1, 0], l[2, 0]])
            lines_edpts.append([u1[0, 0], u1[1, 0], u2[0, 0], u2[1, 0]])

        lines_edpts = np.array(lines_edpts)
        lines = np.array(lines)
        if ax_plot is not None:
            for i in range(0, len(lines_edpts)):
                l = lines_edpts[i]
                ax_plot.plot((l[0], l[2]), (l[1], l[3]), linewidth=0.5,
                             color=[1.0, 0, 0])

    return lines, lines_edpts


def merge_inl(inl1, inl2):
    """Merge logical arrays the second points to true values of the first."""

    inx = np.nonzero(inl1)[0]
    inx_outl = inx[np.logical_not(inl2)]
    inl1 = inl1.copy()
    inl1[inx_outl] = False
    return inl1


def split_line_to_segments(line, u, max_gap, min_length):
    # orthogonal line passing the origin (d=0)
    n0 = np.array([line[1], -line[0]])
    # line parameter
    t = n0 @ u

    ord = np.argsort(t)
    t = t[ord]

    diff = t[1:] - t[:-1]
    br = np.nonzero(diff > max_gap)[0]
    br = np.hstack((br + 1, len(t)))
    seg = []
    last_i = 0
    for i in br:
        if (t[i - 1] - t[last_i]) > min_length:
            seg.append(ord[last_i:i])
        last_i = i

    return seg


def hsegment_to_segments(lh, u_e, opt: Opt):
    seg_u = []

    assert lh.shape == (3, 1)
    lh = lh.flatten()
    d: np.ndarray = lh[:2] @ u_e + lh[2]
    inl = np.abs(d) < opt.segment['thr0']
    # h = plt.gca().plot(u_e[0, inl], u_e[1, inl], 'w.')

    while inl.sum() > opt.segment['min_length']:
        lr, iinl = ransac_line(u_e[:, inl], thr=opt.segment['thr'],
                               log_prob=opt.segment['log_prob'])
        inl = merge_inl(inl, iinl)
        keep = np.logical_not(inl)

        seg = split_line_to_segments(lr, u_e[:, inl],
                                     max_gap=opt.segment['max_gap'],
                                     min_length=opt.segment['min_length'])
        if len(seg) == 0:
            # TODO warning?
            break

        inx = np.nonzero(inl)[0]

        for s in seg:
            i = inx[s]
            inl[i] = False
            seg_u.append(u_e[:, i])
            # plt.plot(u_e[0, i], u_e[1, i], 'x')

        # remaining inl not used, return to keep
        keep[inl] = True
        u_e = u_e[:, keep]

        d: np.ndarray = lh[:2] @ u_e + lh[2]
        inl = np.abs(d) < opt.segment['thr0']

    return seg_u, u_e


def ransac_line(points, thr, log_prob):
    u = g.e2p(points)

    def line_support(sample: VectorI, best_support: float = None):
        u1_ = u[:, sample[0]]
        u2_ = u[:, sample[1]]
        l_ = np.cross(u1_, u2_)
        l_ = l_ / np.sqrt(l_[0] ** 2 + l_[1] ** 2)

        d_ = l_ @ u
        inl_ = np.abs(d_) < thr
        n_inl_ = inl_.sum()

        return n_inl_, n_inl_, inl_, l_

    best_sample = ransac(u.shape[1], 2, log_prob, line_support,
                         verbose='none')
    _, _, inl, l = line_support(best_sample)

    return l, inl


def detect_all_segments(img_u, opt: Opt, mask=None, ax_detect=None
                        ) -> tuple[list[LineSegment], Any, Any]:
    # edges detection
    img_e, u_e = edges(img_u, opt)
    if ax_detect is not None:
        ax_detect.plot(u_e[0], u_e[1], '.')

    if mask is not None:
        img_e = img_e * mask

    # TODO limit computational time if there is too much segments
    u_ex = u_e.copy()
    if opt.use_hough_p:
        lines, lines_endpoints_h = line_segmentsP(img_e, opt, ax_detect)
    else:
        lines, lines_endpoints_h = line_segments(img_e, opt, ax_detect)

    seg_u_pt = []
    for lh in lines:
        seg_u_pt_, u_ex = hsegment_to_segments(lh, u_ex, opt)
        seg_u_pt = seg_u_pt + seg_u_pt_

    seg_u = []
    for s in seg_u_pt:
        # end-points: 0, -1
        ls = LineSegment(u1=s[:, [0]], u2=s[:, [-1]])
        seg_u.append(ls)

    # sort by length
    seg_u = sorted(seg_u, key=lambda x: x.len, reverse=True)

    if len(seg_u) > opt.segment['max_segments']:
        seg_u = seg_u[:opt.segment['max_segments']]

    return seg_u, img_e, u_e


def point_to_segment_distance_q(seg: LineSegment, u):
    # point pose along the line
    t = (seg.v.T @ u)[0]

    # squared point-line distance
    d = (seg.l[:2].T @ u + seg.l[2])[0] ** 2

    # point outside, use squared Euclidean distance to 11
    d1 = ((u - seg.u1) ** 2).sum(axis=0)
    d[t < seg.t1] = d1[t < seg.t1]

    # point outside, use squared Euclidean distance to u2
    d2 = ((u - seg.u2) ** 2).sum(axis=0)
    d[t > seg.t2] = d2[t > seg.t2]

    return d


def seg_distance_matrix(seg1: list[LineSegment], rot, t,
                        seg2: list[LineSegment]):
    mat = np.zeros((len(seg1), len(seg2)))

    u1 = np.hstack([x.u1 for x in seg1])
    u2 = np.hstack([x.u2 for x in seg1])

    u1 = rot @ u1 + t
    u2 = rot @ u2 + t

    for i in range(len(seg2)):
        d1 = point_to_segment_distance_q(seg2[i], u1)
        d2 = point_to_segment_distance_q(seg2[i], u2)
        mat[:, i] = np.vstack((d1, d2)).max(axis=0)

    return mat


def segment_pairs(seg_u: list[LineSegment], opt: Opt):
    pairs = []

    max_cos = np.cos(opt.match['min_join_angle'])

    for i in range(len(seg_u)):
        for j in range(i + 1, len(seg_u)):
            s1 = seg_u[i]
            s2 = seg_u[j]

            # lines
            l1 = s1.l
            l2 = s2.l

            # normals, angle
            d = (l1[:2] * l2[:2]).sum()
            if d < 0.0:
                l2 = -l2
                d = -d

            if d <= max_cos:
                # included angle serves as a descriptor
                angle = np.arccos(d)

                # intersection
                x = g.l2u(l1, l2)

                # direction vectors from intersection to a segment
                u1 = s1.u1 - x
                u1b = s1.u2 - x
                u2 = s2.u1 - x
                u2b = s2.u2 - x

                if g.vlen(u1b) > g.vlen(u1):
                    u1 = u1b

                if g.vlen(u2b) > g.vlen(u2):
                    u2 = u2b

                # TODO in case of T - junction two rotations should be considered

                # middle vector of direction vectors
                u1 = g.vnz(u1)
                u2 = g.vnz(u2)
                n_mid = g.vnz(u1 + u2)

                c = n_mid[0, 0]
                s = n_mid[1, 0]

                rot = np.array([[c, -s], [s, c]])
                # n_mid = rot @ [[1.0], [0]]
                # rot @ [0;0] + x -> x
                # rot @ [1;0] + x -> x + n_mid

                # if (57.9 < x[0,0] < 58.1 and 96.9 <  x[1,0] < 97.1 or
                #   2270 < x[0, 0] < 2290 and 350 < x[1, 0] < 370):
                pairs.append([angle, rot, x, s1, s2])

                if False:
                    plt.plot(x[0], x[1], 'ro')
                    plt.plot((s1[0][0], s1[1][0]), (s1[0][1], s1[1][1]), 'b')
                    plt.plot((s2[0][0], s2[1][0]), (s2[0][1], s2[1][1]), 'b')

                    plt.plot((x[0], x[0] + 100 * n_mid[0]),
                             (x[1], x[1] + 100 * n_mid[1]), 'g')
                    plt.axis('equal')

    return pairs


def match_pairs(pairs, pairs_ref, seg, seg_ref, opt: Opt):
    # x1 = np.hstack([p[2] for p in pairs])
    # xr = np.hstack([p[2] for p in pairs_ref])

    thr = opt.match['max_seg_distance'] ** 2  # dist matrix has squared values
    best_n = 0
    best_rot = None
    best_t = None

    for p1 in pairs:
        rot1 = p1[1]
        t1 = p1[2]
        for pr in pairs_ref:
            if np.abs(p1[0] - pr[0]) > opt.match['max_join_angle_diff']:
                continue

            # ref to image
            rot = rot1 @ pr[1].T
            t = - rot @ pr[2] + t1

            # image to ref
            mat = seg_distance_matrix(seg, rot.T, -rot.T @ t, seg_ref)
            dx = mat.min(axis=0)
            inl = dx < thr
            ninl = inl.sum()
            if ninl > best_n:
                best_n = ninl
                best_rot = rot
                best_t = t

            if best_n >= opt.match['max_match']:
                return best_rot, best_t, best_n

    if best_n >= opt.match['min_match']:
        return best_rot, best_t, best_n

    return None, None, best_n


def abs_ori(ux, uy, scale=False):
    xm = ux.mean(axis=1).reshape(-1, 1)
    ym = uy.mean(axis=1).reshape(-1, 1)

    ux1 = ux - xm
    uy1 = uy - xm

    mat = ux1 @ uy1.T

    u, d, vh = np.linalg.svd(mat)
    rot = vh.T @ u.T

    # absolute orientation with scale
    if scale:
        # Trace(Y_T @ R @ X ) / Trace (X_T @ X)
        s = d.sum() / (ux1**2).sum()
        trn = ym - s * rot @ xm
        return rot, trn, s

    trn = ym - rot @ xm

    return rot, trn


def nearest_points_lines(seg_ref: list[LineSegment], u, rot, trn, thr=None):
    uk = rot.T @ (u - trn)
    dq = point_to_segment_distance_q(seg_ref[0], uk)
    # points on the segment nearest to uk
    # TODO now works as a line, not segment
    uref = seg_ref[0].pt_mat @ uk + seg_ref[0].pt_vec

    for i in range(1, len(seg_ref)):
        di = point_to_segment_distance_q(seg_ref[i], uk)
        ui = seg_ref[i].pt_mat @ uk + seg_ref[i].pt_vec
        ok = di < dq
        dq[ok] = di[ok]
        uref[:, ok] = ui[:, ok]

    # optional thresholding
    if thr is not None:
        ok = dq < thr ** 2
        uref = uref[:, ok]
        u = u[:, ok]
        dq = dq[ok]

    return uref, u, dq


def icp_points_lines(seg_ref: list[LineSegment], u, rot, trn, opt: Opt):
    # fixed number of iteration of ICP
    for i in range(opt.icp['iter']):
        u_ref, u_ok, dq = nearest_points_lines(seg_ref, u, rot, trn,
                                               opt.icp['thr'])
        rot, trn = abs_ori(u_ref, u_ok)

    return rot, trn
