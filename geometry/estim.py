"""Estimation of geometric models from data."""

from typing import Optional
from enum import Enum
import numpy as np
import numpy.linalg

import geometry.basic as gb
from geometry.types import *
from geometry.projection import ParamK

class CalibKMode(Enum):
    """Enum of camera internal parameters calibration modes."""
    FULL = 'full'
    """full matrix K (all entries) is estimated"""

    NOSKEW = 'noskew'
    """skew is zero(q=0)"""

    NOCENTER = 'nocenter'
    """image center coordinates are zero (dx = dy = 0)"""

    FXY = 'fxy'
    """skew and image center is zero, only fx and fy are estimated"""

    SQUARE = 'square'
    """square pixels (q = 0, fx = fy)"""

    F = 'f'
    """skew and image center is zero, only single fx = fy is estimated"""


def calib_k(mode: CalibKMode, constraints, k0: ParamK = None) -> ParamK:
    """Camera internal parameters auto-calibration from scene constraints.
    ::

        k = calib_k(mode, constraints, k0)

    :param mode: calibration mode, determines fixed entries of estimated K1

    :param constraints: list of scene constraints

    :param k0: initial calibration parameters

    :returns: estimated calibration parameters

    Estimation of internal camera calibration parameters (entries of matrix K)
    from scene constraints. Initial parameters (inverse of K0) is used for
    transforming the data of the constraints, then parameters (K1) are estimated
    with some entries optionally fixed, and the final calibration is composed
    (K = K0 * K1).

    All mentioned matrices are composed of the fields (fx, fy, dx, dy, q) of
    a :py:obj:`ParamK` object as
    ::

        K = [[fx, q, dx]
            [0, fy, dy]
            [0, 0, 1]]

    Calibration mode determines fixed entries of the matrix being estimated.

    Enough constraints must be given, depending on the number of estimated
    parameters. Every constraint is a list or tuple [name, data, ...]. Possible
    constraints are

        ['vp', u, v]
            u, v are vanishing points of two orthogonal directions
            (homogeneous 3x1 vectors)

        ['h', homo]
            homo is a scene plane-to-image 3x3 homography matrix

    Based on [Hartley-Zisserman2003] Sec.8.8/p.223+.
    """

    def one_row(u, v):
        """One row in the data matrix built from scene constraints.
        ::
            at = one_row(u, v)

        express
          u.T @ omega @ v
        (omega is symmetric 3x3 matrix) as
          at @ w
        where w.T = [omega_11, omega_12, omega_22, omega_13, omega_23, omega_33]
        """

        return np.array([u[0] * v[0],
                         u[0] * v[1] + u[1] * v[0],
                         u[1] * v[1],
                         u[0] * v[2] + u[2] * v[0],
                         u[1] * v[2] + u[2] * v[1],
                         u[2] * v[2]])


    mat_k0_inv = np.eye(3, 3)
    if k0 is not None:  # k0 is optional
        mat_k0_inv = k0.inv().matrix()

    # build data matrix (for full w/omega) from scene constraints; its columns
    # correspond to vector w, i.e. to
    # omega_11, omega_12, omega_22, omega_13, omega_23, omega_33, respectively

    # count constraints to preallocate the matrix
    row = 0
    for c in constraints:
        if c[0] == 'vp':
            row += 1
        elif c[0] == 'h':
            row += 2
        else:
            raise ValueError("Wrong constraint " + str(c))

    # build the matrix
    mat_a = np.zeros((row, 6))
    row = 0
    for c in constraints:
        if c[0] == 'vp':  # ['vp', vp1, vp2]
            if k0 is None:
                vp1, vp2 = c[1], c[2]
            else:
                vp1, vp2 = mat_k0_inv @ c[1], mat_k0_inv @ c[2]

            mat_a[row] = one_row(vp1, vp2)
            row += 1

        elif c[0] == 'h':  # ['h', homo]
            if k0 is None:
                h1, h2 = c[1][:, 0], c[1][:, 1]
            else:
                h1, h2 = mat_k0_inv @ c[1][:, 0], mat_k0_inv @ c[1][:, 1]

            mat_a[row] = one_row(h1, h2)
            row += 1
            mat_a[row] = one_row(h1, h1) - one_row(h2, h2)
            row += 1

        else:
            raise ValueError("Wrong constraint " + str(c))

    # treat different modes - different sets of parameters to be estimated
    # (data matrix is modified)
    w = np.zeros(6)

    if mode == CalibKMode.FULL:  # full estimation of K
        w = norm_dlt_solve(mat_a)

    elif mode == CalibKMode.NOSKEW:  # omega_12 = 0
        m = [0, 2, 3, 4, 5]
        mat_a_ = mat_a[:, m]
        w_ = norm_dlt_solve(mat_a_)
        w[m] = w_

    elif mode == CalibKMode.NOCENTER:  # omega_13 = omega_23 = 0
        m = [0, 1, 2, 5]
        mat_a_ = mat_a[:, m]
        w_ = norm_dlt_solve(mat_a_)
        w[m] = w_

    elif mode == CalibKMode.FXY:  # omega_12 = omega_13 = omega_23 = 0
        m = [0, 2, 5]
        mat_a_ = mat_a[:, m]
        w_ = norm_dlt_solve(mat_a_)
        w[m] = w_

    elif mode == CalibKMode.SQUARE:  # omega_12 = 0, omega_11 = omega_22
        m = [0, 3, 4, 5]
        mat_a_ = mat_a[:, m]
        mat_a_[:, 0] = mat_a_[:, 0] + mat_a[:, 2]  # omega_11 = omega_22
        w_ = norm_dlt_solve(mat_a_)
        w[m] = w_
        w[2] = w_[0]

    elif mode == CalibKMode.F:
        # omega_12 = omega_13, all others except omega_33 are zero
        mat_a_ = np.hstack((mat_a[:, 0] + mat_a[:, 2], mat_a[:, 5]))
        w_ = norm_dlt_solve(mat_a_)
        w[0] = w_[0]
        w[2] = w_[0]
        w[5] = w_[1]

    else:
        raise ValueError('Wrong K calibration mode ' + str(mode))

    # Solution of mat_a is up to a scale/sign, so we got alpha * w, where alpha
    # can be negative. We need to fix the sign, otherwise chol is impossible;
    # the value of alpha is eliminated using K(3, 3) = 1 later.

    if w[0] < 0:  # w[0] = omega_11 = 1/f_x^2, so it must be positive
        w = -w

    mat_w = np.array([[w[0], w[1], w[3]],
                      [w[1], w[2], w[4]],
                      [w[3], w[4], w[5]]])

    # TODO force positive definitness of W?

    mat_k = np.linalg.cholesky(mat_w)

    mat_k /= mat_k[2, 2] # better numerical stability of inversion
    mat_k = np.linalg.inv(mat_k).T

    if k0 is not None:
        mat_k = k0.matrix() @ mat_k

    return ParamK.from_matrix(mat_k)


x = np.array([[ 2.75988404e-01,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  2.75988404e-01,  0.00000000e+00],
       [-5.74484419e+00,  1.08690132e+01,  9.07630649e+02]])



def norm_dlt_solve(mat_a):
    """Solution of linear homogeneous system with normalization.
    ::

        x = norm_dlt_solve(mat_a)

    Solves (overdetermined) linear homogeneous system
    ::

        mat_a @ x = 0

    using SVD with normalization.

    Rows and columns of mat_a are balanced by coefficients that are adjusted
    to powers of two to avoid round-off error.
    """

    # Multiplication of rows and columns can be done by diagonal matrices, but
    # using singleton expansion is faster.

    # row normalization coefficients (adjusted to power of two)
    tl = 2.0**(-np.round(np.log2(np.sqrt((mat_a**2).sum(axis=1,
                                                        keepdims=True)))))

    # column normalization coefficients (adjusted to power of two)
    tr = 2.0**(-np.round(np.log2(np.sqrt(((tl * mat_a)**2).sum(axis=0)))))

    _, _, vt = np.linalg.svd(tl * mat_a * tr)

    # diag(tl) is regular, diag(tl) @ mat_a @ x = 0 <=> mat_a @ X = 0,
    # we need to apply to the result tr only
    x = tr * vt[-1]

    return x


def u2h_dltn(u1: Matrix2n, u2: Matrix2n) -> Optional[Matrix33]:
    """Planar homography est. from corresponding points by normalized DLT.
    ::

        h_mat = u2h_dltn(u1, u2)

    Planar homography estimation from corresponding points using direct linear
    transform (DLT) algorithm. Over-determined system constructed from
    normalized coordinates is solved by SVD. A specialized version of
    ``calproj``.

    See [fragments:homography-estimation].

    :param u1: planar points; 2xN matrix of column vectors
    :param u2: planar points corresponding to ``u1``; the same size

    :returns: estimated planar homography; 3x3 regular matrix defined up to
              noise as ``u2 = p2e(h_mat @ e2p(u1))``, or None if estimation
              fails
    """

    n = u1.shape[1]

    if n < 4:
        return None

    # normalization
    u1, trn1, _, _ = gb.normalize_coords_sd(u1)
    u2, trn2, _, _ = gb.normalize_coords_sd(u2)

    # DLT
    data_mat = np.zeros((n * 2, 9))

    for i in range(n):
        r1 = 2 * i
        r2 = 2 * i + 1

        data_mat[r1, 0:2] = u1[0:2, i]
        data_mat[r1, 2] = 1
        data_mat[r1, 6:8] = -u2[0, i] * u1[0:2, i]
        data_mat[r1, 8] = -u2[0, i]

        data_mat[r2, 3:5] = u1[0:2, i]
        data_mat[r2, 5] = 1
        data_mat[r2, 6:8] = -u2[1, i] * u1[0:2, i]
        data_mat[r2, 8] = -u2[1, i]

    u, d, vt = np.linalg.svd(data_mat)

    h_mat = vt[8].reshape((3, 3))

    # de-normalization
    h_mat = np.linalg.inv(trn2) @ h_mat @ trn1

    return h_mat


def u2h_p4_dlt(u1: Matrix2n, u2: Matrix2n) -> Matrix33:
    """Planar homography estimation from 4 corresponding points by DLT.
    ::

        h_mat = u2h_p4_dlt(u1, u2)

    Planar homography estimation from 4 corresponding points using direct linear
    transform (DLT) algorithm; a minimal problem, coordinates not normalized.

    See [fragments:homography-estimation-minimal].

    :param u1: planar points; 2x4 matrix of column vectors
    :param u2: planar points corresponding to ``u1``; the same size

    :returns: estimated planar homography; 3x3 regular matrix defined exactly
              as ``u2 = p2e(h_mat @ e2p(u1))``
    """

    # DLT: build matrix A^T, as qr is used for finding its left null-space
    at_mat = np.zeros((9, 8))
    u1h = gb.e2p(u1)

    for i in range(4):
        at_mat[0:3, 2 * i] = u1h[:, i]
        at_mat[6:9, 2 * i] = -u1h[:, i] * u2[0, i]
        at_mat[3:6, 2 * i + 1] = u1h[:, i]
        at_mat[6:9, 2 * i + 1] = -u1h[:, i] * u2[1, i]

    # DLT: solve h^T * A^T = 0 - left null-space of A^T
    q, _ = numpy.linalg.qr(at_mat, 'complete')
    return q[:, 8].reshape((3, 3))


def u2line_lsn(u: Matrix2n) -> (Vector2, float):
    """Planar line estimation from points by least squares.
    ::

        n, d = u2line_lsn(u)

    Planar line estimation from points by minimization of sum of squared
    point-line distances.

    See [fragments:line-estimation].

    :param u: planar points; 2xN matrix of column vectors
    :returns: estimated line as (n, d); n is the normal vector of the line,
              d is its distance from the origin

    """

    # the optimal line is passing through the centroid
    um = u.mean(axis=1, keepdims=True)
    u_ = u - um
    mat_u = u_ @ u_.T
    # Seek n minimizing n.T @ mat_u @ n such that n.T @ n = 1:
    # eigenvector for the smaller eigenvalue of the symmetric positive
    # (semi)definite matrix mat_u.

    # closed form solution for the symmetric 2x2 matrix
    a, b, c = mat_u[0, 0], mat_u[1, 1], mat_u[0, 1]
    discr = np.sqrt(4 * c**2 + (a - b)**2)

    # a, b are positive, the smaller eigenvalue is
    # (a + b - discr) / 2, the bigger is (a + b + discr) / 2
    eig = (a + b - discr) / 2

    # now eigenvector is multiple of [eig-b; c]
    q = np.sqrt((eig - b)**2 + c**2)  # normalization
    n = np.array([(eig - b) / q, c / q])

    # any point lying exactly on the line gives d
    d = -n @ um[:, 0]

    return n, d
