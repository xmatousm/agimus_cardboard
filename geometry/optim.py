"""Optimization of geometric models using data."""

# (c) 2024-04-20 Martin Matousek
# Last change: $Date$
#              $Revision$

from typing import Optional
import numpy as np
import numpy.linalg

import mathlib.optimize as opt
import geometry.basic as gb
from geometry.types import *


def hu_one_refine(h0: Matrix33, u1: Matrix2N, u2: Matrix2N,
                  **kwargs) -> Matrix33:
    """Iterative refinement of homography with one-directional err.
    ::
        h_opt = hu2_one_refine(h0, u1, u2, **kwargs)

    Iterative refinement of planar homography with one-directional error on
    corresponding points. The refinement minimises the sum of squared distances
                 || u2_i, h(u1_i) ||
    where h(u1_i) is mapping of point u1_i by the homography h.

    :param h0: initial estimate of homography; 3x3 regular matrix
    :param u1: planar points; 2xN matrix of column vectors
    :param u2: planar points corresponding to ``u1``; same size
    :kwargs: additional arguments passed to fmin_lsq

    :returns: updated homography
    """

    def efun(h):
        """Function of residues.
        One-directional transfer error on homography.
        """

        hu1 = h[0] * u1[0] + h[1] * u1[1] + h[2]
        hu2 = h[3] * u1[0] + h[4] * u1[1] + h[5]
        hu3 = h[6] * u1[0] + h[7] * u1[1] + h[8]

        return np.vstack((hu1 / hu3, hu2 / hu3)) - u2

    h_opt = opt.fmin_lsq(h0.reshape(-1, order='C'), efun, **kwargs)
    return h_opt.reshape(3, 3)


def rtxu_refine( rot0, t0, x2u, x_scene, u_image, **kwargs):

    def efun(x):
        """Function of residues.
        Scene-to-image reprojection error.
        """

        rot = gb.a2r(x[:3]) @ rot0
        t = x[3:].reshape(3,1) + t0

        ux = x2u(rot @ x_scene + t)
        return ux - u_image

    #opt_fmin = parseargs( 'mindf', 0.01, varargin );

    # optimize
    x_opt = opt.fmin_lsq(np.zeros(6), efun, **kwargs)

    # compose results
    rot_opt = gb.a2r(x_opt[:3]) @ rot0
    t_opt = x_opt[3:].reshape(3,1) + t0

    return rot_opt, t_opt
