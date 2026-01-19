"""Basic tools for Euclidean and projective geometry."""

# (c) 2020 Martin Matousek
# Last change: $Date$
#              $Revision$

from typing import Union

import numpy as np

from geometry.types import *

_EPS = np.finfo(float).eps
_EYE3 = np.eye(3)


# noinspection DuplicatedCode
def a2r(a: Vector3) -> Matrix33:
    """Axis-angle to 3D rotation matrix (with the angle as the axis length).
    ::

        rot_mat = a2r(a)

    3D rotation matrix from axis-angle representation using Rodrigues'
    rotation formula.

    :param a: rotation axis; its length provides a rotation angle (3x1 vector)

    :returns: rotation matrix (3x3, orthonormal, det = +1)
    """

    alpha_q = a[0] ** 2 + a[1] ** 2 + a[2] ** 2

    if alpha_q < _EPS:
        # do not divide, sin(alpha)/alpha = 1, 1-cos(alpha) = 0,
        # matrix is E + sqc(a)

        return np.array([[1.0, -a[2], a[1]],
                         [a[2], 1.0, -a[0]],
                         [-a[1], a[0], 1.0]])

    # angle encoded as the axis length
    alpha = np.sqrt(alpha_q)

    # normalize the axis to unit length
    a = a / alpha

    # skew-symmetric matrix from the axis
    sqc_a = np.array([[0.0, -a[2], a[1]],
                      [a[2], 0.0, -a[0]],
                      [-a[1], a[0], 0.0]])

    # Rodrigues' formula; we can compute rot_mat = expm(sqc_a * alpha), but
    # the formula is 10x faster.
    return _EYE3 + sqc_a * np.sin(alpha) + sqc_a @ sqc_a * (1 - np.cos(alpha))


# noinspection DuplicatedCode
def aa2r(a: Union[Vector3, list, tuple], alpha: float) -> Matrix33:
    """Axis-angle to 3D rotation matrix.
    ::

        rot_mat = aa2r(a, alpha)

    3D rotation matrix from axis-angle representation using Rodrigues'
    rotation formula.

    :param a: rotation axis; its length is ignored (3x1 vector)
    :param alpha: angle of rotation around the axis

    :returns: rotation matrix (3x3, orthonormal, det = +1)
    """

    # normalize the axis to unit length
    len_q = a[0] ** 2 + a[1] ** 2 + a[2] ** 2
    a = a / np.sqrt(len_q)

    # skew-symmetric matrix from the axis
    sqc_a = np.array([[0.0, -a[2], a[1]],
                      [a[2], 0.0, -a[0]],
                      [-a[1], a[0], 0.0]])

    # Rodrigues' formula; we can compute rot_mat = expm(sqc_a * alpha), but
    # the formula is 10x faster.
    return _EYE3 + sqc_a * np.sin(alpha) + sqc_a @ sqc_a * (1 - np.cos(alpha))


def cropline(lines: Matrix3N,
             min_x: float, max_x: float, min_y: float, max_y: float):
    """Intersection of planar lines and a bounding box.
    ::

        u1, u2 = cropline(lines, min_x, max_x, min_y, max_y)

    :param lines: planar lines, homog. coordinates, matrix of 3xN column vectors
    :param min_x: x-axis lower bound of the area
    :param max_x: x-axis upper bound of the area
    :param min_y: y-axis lower bound of the area
    :param max_y: y-axis upper bound of the area

    :returns: tuple of starting and ending points of every line segment, 2xN
              column vectors, with nan when a particular point does not exist
    """

    size_x = max_x - min_x
    size_y = max_y - min_y

    o = np.ones((lines.shape[1]))

    # intersections with the x-axis lower bound, x = min_x, solve for y in
    #   min_x * l[0] + y * l[1] + l[2] = 0
    x1 = min_x * o
    y1 = (- min_x * lines[0] - lines[2])
    ok = lines[1] != 0.0
    y1[ok] /= lines[1][ok]
    y1[np.logical_not(ok)] = np.inf

    # intersections with the x-axis upper bound, x = max_x, solve for y in
    #   max_x * l[0] + y * l[1] + l[2] = 0
    x2 = max_x * o
    y2 = (- max_x * lines[0] - lines[2])
    y2[ok] /= lines[1][ok]
    y2[np.logical_not(ok)] = np.inf

    # intersections with the y-axis lower bound, y = min_y, solve for x in
    #   x * l[0] + min_yy * l[1] + l[2] = 0
    x3 = (- min_y * lines[1] - lines[2])
    y3 = min_y * o
    ok = lines[0] != 0.0
    x3[ok] /= lines[0][ok]
    x3[np.logical_not(ok)] = np.inf

    # intersections with the y-axis upper bound, y = max_y, solve for x in
    #   x * l[0] + max_y * l[1] + l[2] = 0
    x4 = (- max_y * lines[1] - lines[2])
    y4 = max_y * o
    x4[ok] /= lines[0][ok]
    x4[np.logical_not(ok)] = np.inf

    # Each point lies on one of the boundary lines, we compute its distances
    # (absolute value) from both boundaries orthogonal to this boundary. Sum of
    # these two distaces divided by the size of the bounding box equals
    # one for points inside, and is larger than one for points outside.

    d1 = (np.abs(y1 - min_y) + np.abs(max_y - y1)) / size_y
    d2 = (np.abs(y2 - min_y) + np.abs(max_y - y2)) / size_y
    d3 = (np.abs(x3 - min_x) + np.abs(max_x - x3)) / size_x
    d4 = (np.abs(x4 - min_x) + np.abs(max_x - x4)) / size_x

    # select the two points for which the normalized distance is minimal
    d = np.vstack((d1, d2, d3, d4))
    w = np.argsort(d.T).T

    x = np.vstack((x1, x2, x3, x4))
    y = np.vstack((y1, y2, y3, y4))

    # index
    w = w[:2]
    k = np.arange(w.shape[1])

    x = x[w, k]
    y = y[w, k]
    d = d[w, k]

    #  check if the 'worst' point is still inside (with some tolerance), put
    #  nan if it is not
    bad = d[1] > 1.001

    x[:, bad] = np.nan
    y[:, bad] = np.nan

    return np.vstack((x[0], y[0])), np.vstack((x[1], y[1]))


def e2p(u: Matrix) -> Matrix:
    """Euclidean to projective coordinates.
    ::

        y = e2p(x)

    Returns projective (homogeneous) coordinates of vectors; adds the last row
    of ones to the matrix of column vectors (any dimension).
    """

    return np.vstack((u, np.ones((1, u.shape[1]))))


def e2i(u: Matrix) -> Matrix:
    return np.vstack((u, np.zeros((1, u.shape[1]))))


def mnz(mat: Matrix) -> Matrix:
    """Matrix normalization to unit Frobenius norm.
    ::

        norm_mat = mnz(mat)

    :param mat: matrix of any shape

    :returns: the matrix scaled to have unit Frobenius norm.
    """

    return mat / np.sqrt((mat ** 2).sum())


def normalize_coords_sd(x: Matrix) -> (Matrix, Matrix, float, Vector):
    """Normalization of coordinates by scale and shift.
    ::

        y, trn, s, d = normalize_coords_sd(x)

    Transform coordinates such that its centroid is origin, and a mean distance
    is unity. The normalizing transformation is a shift followed by a scale.

    :param x: coordinates of any dimension; column vectors

    :returns: tuple (y, trn, scale, shift)
        - y - transformed coordinates
        - trn - homogeneous transformation matrix, its last row is [0, ..., 0, 1]
        - d - shift; column vector
        - s - scale; scalar

    A point x_i transforms as::
        y_i = s * (y_i + d)

    which can be also computed as::
        y_i = p2e(trn @ e2p(x_i))
    """

    dim = x.shape[0]

    # centroid to origin
    d = - x.mean(axis=1, keepdims=True)
    xd = x + d

    # mean distance to one
    lenq = (xd ** 2).sum(axis=0)
    s = 1 / np.sqrt(lenq.mean())
    y = s * xd

    s_mat = np.eye(dim) * s

    trn = np.block([[s_mat, s_mat @ d.reshape((-1, 1))],
                    [np.zeros(dim), 1]])

    return y, trn, s, d


def p2e(u: Matrix) -> Matrix:
    """Projective to Euclidean coordinates.
    ::

        y = p2e(x)

    Returns Euclidean coordinates of projective (homogeneous) vectors; divide
    the matrix of column vectors by the last row and remove the row.
    """

    return u[:-1] / u[-1]


def p2p1(u: Matrix) -> Matrix:
    """Normalize the last entry of projective vectors to 1.
    ::

        y = p2p1(x)

    Divide the matrix of column vectors by the last row.
    """

    return u / u[-1]


def elem_homo(steps: Union[list, tuple]):
    homo = np.eye(3)

    for step in steps:
        key, value = step

        match key:
            case 'rx':
                homo = rx(value) @ homo
            case 'ry':
                homo = ry(value) @ homo
            case 'rz':
                homo = rz(value) @ homo
            case 'sx':
                homo[0] *= value
            case 'sy':
                homo[1] *= value
            case 'dx':
                homo[0] += value * homo[2]
            case 'dy':
                homo[1] += value * homo[2]
            case 'q':
                homo[0] += value * homo[1]
            case '_':
                raise NotImplementedError('Wrong key ' + str(key))

    return homo


def rot(angle: float) -> Matrix22:
    """Matrix of planar rotation.
    ::

        rot_mat = rot(angle)

    :param angle: rotation angle

    For a positive angle, a rotation of a 2D vector ``v`` as ``rot_mat @ v``
    is counter-clockwise.

    :returns: 2D rotation matrix (2x2, orthonormal, det = +1)
    """

    s, c = np.sin(angle), np.cos(angle)

    return np.array([[c, -s], [s, c]])


def rx(angle_x: float) -> Matrix33:
    """Matrix of 3D elementary rotation around x-axis.
    ::

        rot_mat = rx(angle_x)

    For a positive angle, a rotation of a 3D vector ``v`` as ``rot_mat @ v``
    is counter-clockwise when the rotation axis points toward the observer.

    :param angle_x: rotation angle around x-axis

    :returns: 3D rotation matrix (3x3, orthogonal, det = +1)
    """

    s, c = np.sin(angle_x), np.cos(angle_x)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(angle_y: float) -> Matrix33:
    """Matrix of 3D elementary rotation around y-axis.
    ::

        rot_mat = ry(angle_y)

    For a positive angle, a rotation of a 3D vector ``v`` as ``rot_mat @ v``
    is counter-clockwise when the rotation axis points toward the observer.

    :param angle_y: rotation angle around x-axis

    :returns: 3D rotation matrix (3x3, orthogonal, det = +1)
    """

    s, c = np.sin(angle_y), np.cos(angle_y)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(angle_z: float) -> Matrix33:
    """Matrix of 3D elementary rotation around z-axis.
    ::

        rot_mat = rx(angle_z)

    For a positive angle, a rotation of a 3D vector ``v`` as ``rot_mat @ v``
    is counter-clockwise when the rotation axis points toward the observer.

    :param angle_z: rotation angle around z-axis

    :returns: 3D rotation matrix (3x3, orthogonal, det = +1)
    """

    s, c = np.sin(angle_z), np.cos(angle_z)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def sqc(x: Vector3) -> Matrix33:
    """Skew-symmetric matrix.
    ::

        s = sqc(x)

    Skew-symmetric matrix for cross-product, etc. Cross-product of two vectors
    x and y can be computed as ``sqc(x) @ y``.

    :param x: vector od 3 dimensions

    :returns: skew symmetric matrix built from x
    """

    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def l2u(l1: Vector3, l2: Vector3) -> Vector2:
    """Points given as an intersection of line pairs.

    :param l1: matrix (3 x N) of N column vectors representing the planar lines
    :param l2: matrix (3 x N) of the planar lines corresponding to ``l1``
    :returns: matrix (2 x N) of N column vectors representing the planar points
    """

    # cross-product followed by normalization to Euclidean coordinates (p2e)
    a = l1[1] * l2[2] - l2[1] * l1[2]
    b = l1[2] * l2[0] - l2[2] * l1[0]
    c = l1[0] * l2[1] - l2[0] * l1[1]

    return np.vstack((a / c, b / c))


def u2l(u1: Vector2, u2: Vector2) -> Vector3:
    """Lines given as a join of point pairs.

    :param u1: matrix (2 x N) of N planar column vectors
    :param u2: matrix (2 x N) of vectors corresponding to the vectors in u1
    :returns: matrix (3 x N) of N column vectors representing the joining lines
    """

    # cross-product of vectors extended by 1 to be homogeneous
    a = u1[1] - u2[1]
    b = u2[0] - u1[0]
    c = u1[0] * u2[1] - u1[1] * u2[0]

    return np.vstack((a, b, c))


def u2l_norm(u1: Matrix2N, u2: Matrix2N) -> Matrix3N:
    """Normalized lines given as a join point pairs.

    :param u1: matrix (2 x N) of N planar column vectors
    :param u2: matrix (2 x N) of vectors corresponding to the vectors in u1
    :returns: matrix (3 x N) of N column vectors representing the joining lines;
        the lines have unit normals
    """

    # cross-product of vectors extended by 1 to be homogeneous
    a = u1[1] - u2[1]
    b = u2[0] - u1[0]
    c = u1[0] * u2[1] - u1[1] * u2[0]
    n = np.sqrt(a ** 2 + b ** 2)

    return np.vstack((a / n, b / n, c / n))


def vang(x1: Matrix, x2: Matrix) -> float:
    """The Included angles of vector pairs.

    :param x1: matrix (D x N) of N column vectors with any dimension D
    :param x2: matrix (D x N) of vectors corresponding to the vectors in x1
    :returns: vector of N angles computed using the dot product
    """

    d = (vnz(x1) * vnz(x2)).sum(axis=0)

    # the d can be a little bit outside <-1,1> due to numerics; e.g.,
    # when x1 = x2 = np.array([[2.0],[2.0],[2.0]]) it would lead to
    # d = ((2.0/np.sqrt(12.0))**2)*3, which is 1 + eps

    d[d > 1.0] = 1.0
    d[d < -1.0] = -1.0

    return np.acos(d)


def vlen(x: Matrix) -> Vector:
    """Column vector(s) length (norm).
    ::

        l = vlen(x)

    :param x: matrix (D x N) of N column vectors of any dimension D

    :returns: vector of N lengths (norms)
    """

    return np.sqrt((x ** 2).sum(axis=0))


def vlenq(x: Matrix) -> Vector:
    """Column vector(s) squared length (norm).
    ::

        l = vlenq(x)

    :param x: matrix (D x N) of N column vectors of any dimension D

    :returns: vector of N squared lengths (norms)
    """
    return (x ** 2).sum(axis=0)


def vnz(x: Matrix) -> Matrix:
    """Column vector(s) normalization to unit length.
    ::

        y = vnz(x)

    :param x: matrix (D x N) of N column vectors of any dimension D

    :returns: matrix of the vectors from x, each scaled to unit length
    """

    return x / np.sqrt((x ** 2).sum(axis=0))


def vnzsub(x: Matrix) -> Matrix:
    """Column vector(s) normalization to unit length of [0:D-1] sub-vector.
    ::

        y = vnzsub(x)

    :param x: matrix (D x N) of N column vectors of dimension D > 1

    :returns: matrix of the vectors from x, each scaled such that its sub-vector
              [0:D-1] has unit length
    """

    return x / np.sqrt((x[:-1] ** 2).sum(axis=0))


def xyz2r(a: Vector3) -> Matrix33:
    """Angles (xyz) to rotation matrix.
    ::

       rot_mat = xyz2r(a)

    Build a rotation matrix from elementary rotations in the X-Y-Z order as::
       rot_mat = rz(a(3)) * ry(a(2)) * rx(a(1))

    This representation of rotation belongs to the group of Tait-Bryan angles.

    :param a: vector of three angles [angle_x, angle_y, angle_z] of
              elementary rotations around appropriate coordinate axes

    :returns: rotation matrix (3x3, orthogonal, det = +1)
    """

    # noinspection DuplicatedCode
    s1, c1 = np.sin(a[0]), np.cos(a[0])
    s2, c2 = np.sin(a[1]), np.cos(a[1])
    s3, c3 = np.sin(a[2]), np.cos(a[2])

    rot_1 = np.array([[1, 0, 0], [0, c1, -s1], [0, s1, c1]])
    rot_2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    rot_3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])

    return rot_3 @ rot_2 @ rot_1


# noinspection DuplicatedCode
def yxz2r(a: Vector3) -> Matrix33:
    """Angles (yxz) to rotation matrix.
    ::

       rot_mat = yxz2r(a)

    Build a rotation matrix from elementary rotations in the Y-X-Z order as::
       rot_mat = rz(a(3)) * rx(a(2)) * ry(a(1))

    This representation of rotation belongs to the group of Tait-Bryan angles.

    :param a: vector of three angles [angle_y, angle_x, angle_z] of
              elementary rotations around appropriate coordinate axes

    :returns: rotation matrix (3x3, orthogonal, det = +1)
    """

    s1, c1 = np.sin(a[0]), np.cos(a[0])
    s2, c2 = np.sin(a[1]), np.cos(a[1])
    s3, c3 = np.sin(a[2]), np.cos(a[2])

    rot_1 = np.array([[c1, 0, s1], [0, 1, 0], [-s1, 0, c1]])
    rot_2 = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]])
    rot_3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])

    return rot_3 @ rot_2 @ rot_1


# noinspection DuplicatedCode
def zxz2r(a: Vector3) -> Matrix33:
    """Angles (zxz) to rotation matrix.
    ::

       rot_mat = zxz2r(a)

    Build a rotation matrix from elementary rotations in the Z-X-Z order as::
       rot_mat = rz(a(3)) * rx(a(2)) * rz(a(1))

    This representation of rotation belongs to the group of proper Euler angles.

    :param a: vector of three angles [angle_z, angle_x, angle_z] of
              elementary rotations around appropriate coordinate axes

    :returns: rotation matrix (3x3, orthogonal, det = +1)
    """

    s1, c1 = np.sin(a[0]), np.cos(a[0])
    s2, c2 = np.sin(a[1]), np.cos(a[1])
    s3, c3 = np.sin(a[2]), np.cos(a[2])

    rot_1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    rot_2 = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]])
    rot_3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])

    return rot_3 @ rot_2 @ rot_1
