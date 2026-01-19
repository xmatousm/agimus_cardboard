"""Camera internal model and projection and back-projection functions."""

# (c) 2021 Martin Matousek
# Last change: $Date$
#              $Revision$

from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Optional, Callable, Any

import numpy as np

from geometry.types import *


@dataclass(kw_only=True)
class ParamK:
    """Entries of internal calibration matrix K.

    The fields can be set in the constructors as keyword arguments, those not
    given has a default value: fx=1.0, fy=1.0, dx=0.0, dy=0.0, q=0.0. The
    internal calibration matrix is then assumed as
    ::

        [[fx, q, dx]
         [0, fy, dy]
         [0, 0, 1]]

    I.e., ParamK() (all defaults) represents the identity.
    """

    fx: float = 1.0
    fy: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    q: float = 0.0

    @classmethod
    def from_matrix(cls, mat_k: Matrix33) -> "ParamK":
        """Build ParamK object from internal calibration matrix."""

        if (mat_k[1, 0] != 0. or mat_k[2, 0] != 0. or mat_k[2, 1] != 0.
                or mat_k[2, 2] != 1.0):
            raise ValueError('Internal calibration matrix must be upper '
                             'triangular with 1.0 as the last entry')

        return ParamK(fx=mat_k[0, 0], fy=mat_k[1, 1], dx=mat_k[0, 2],
                      dy=mat_k[1, 2],
                      q=mat_k[0, 1])

    @classmethod
    def from_dict(cls, d: dict) -> "ParamK":
        """De-serialize the object from a dict."""

        if d.keys() != {'fx', 'fy', 'dx', 'dy', 'q'}:
            raise ValueError('Wrong dict data:', d)

        return ParamK(**d)

    def inv(self) -> "ParamK":
        """Inverse of internal calibration matrix.

        :return: entries of the inverted matrix K
        """

        i_fx = 1. / self.fx
        i_fy = 1. / self.fy
        i_q = - self.q / self.fx / self.fy
        i_dx = - self.dx / self.fx - i_q * self.dy
        i_dy = - self.dy / self.fy

        return ParamK(fx=i_fx, fy=i_fy, dx=i_dx, dy=i_dy, q=i_q)

    def apply(self, u: Matrix2N) -> Matrix2N:
        """Apply internal calibration matrix to coordinates.

        :param u: coordinates of plane points; 2xn matrix of column vectors

        :return: transformed coordinates; 2xn matrix of column vectors
        """

        u_x = (self.fx * u[0] + self.q * u[1] + self.dx if self.q != 0.0
               else self.fx * u[0] + self.dx)
        u_y = self.fy * u[1] + self.dy

        return np.vstack((u_x, u_y))

    def matrix(self) -> Matrix33:
        """Returns the 3x3 internal calibration matrix."""

        return np.array([[self.fx, self.q, self.dx],
                         [0., self.fy, self.dy],
                         [0., 0., 1.]])

    def dict(self) -> dict[str, float]:
        """Serialize the object to a dict."""

        return self.__dict__


class InternalModelProjModel(Enum):
    """Enum of known projection models."""
    TAN = 0
    EQ = 1
    UNI = 2


@dataclass(kw_only=True)
class InternalModelParamsProj:
    """Parameters of projection model - base class."""
    model: InternalModelProjModel = InternalModelProjModel.TAN

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InternalModelParamsProj":
        """De-serialize the object from a dict."""

        if 'model' not in d:
            raise ValueError('Wrong dict data:', d)

        if d['model'] == 'uni':
            return InternalModelParamsProjUni.from_dict(d)

        # other models have no parameters, so no other dict keys are allowed
        if d.keys() != {'model'}:
            raise ValueError('Wrong dict data:', d)

        if d['model'] == 'tan':
            return cls(model=InternalModelProjModel.TAN)

        if d['model'] == 'eq':
            return cls(model=InternalModelProjModel.EQ)

        raise ValueError('Wrong dict data:', d)

    def dict(self) -> dict[str, Any]:
        """Serialize the object to a dict."""

        match self.model:
            case InternalModelProjModel.TAN:
                return {'model': 'tan'}
            case InternalModelProjModel.EQ:
                return {'model': 'eq'}

        raise RuntimeError(f'Unhandled model {self.model}')


@dataclass(kw_only=True)
class InternalModelParamsProjUni(InternalModelParamsProj):
    """Parameters of unified central panoramic model."""
    model: InternalModelProjModel = InternalModelProjModel.UNI
    xi: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InternalModelParamsProjUni":
        """De-serialize the object from a dict."""

        if d.keys() != {'model', 'xi'} or d['model'] != 'uni':
            raise ValueError('Wrong dict data:', d)

        return cls(model=InternalModelProjModel.UNI, xi=d['xi'])

    def dict(self) -> dict[str, Any]:
        """Serialize the object to a dict."""

        if self.model != InternalModelProjModel.UNI:
            raise RuntimeError(f'Unhandled model {self.model}')

        return {'model': 'uni', 'xi': self.xi}


InternalModelParamsProjTypes = Union[
    InternalModelParamsProj,
    InternalModelParamsProjUni]


class InternalModelRdModel(Enum):
    """Enum of known radial and tangential distortion models."""
    NONE = 0  # no radial distortion
    DIV = 1  # division model, forward is in projection
    IDIV = 2  # division model, forward is in back-projection
    POLY = 3  # polynomial model, forward is in projection
    IPOLY = 4  # polynomial model, forward is in back-projection


@dataclass(kw_only=True)
class InternalModelParamsRd:
    """Parameters of the radial (and tangential) distortion model.
    Base class with no distortion."""
    model: InternalModelRdModel = InternalModelRdModel.NONE

    @classmethod
    def from_dict(cls, d):
        """De-serialize the object from a dict."""

        if 'model' not in d:
            raise ValueError('Wrong dict data:', d)

        if d['model'] in ('poly', 'ipoly'):
            return InternalModelParamsRdPoly.from_dict(d)

        if d['model'] in ('div', 'idiv'):
            return InternalModelParamsRdDiv.from_dict(d)

        if d != {'model': 'none'}:
            raise ValueError('Wrong dict data:', d)

        return cls(model=InternalModelRdModel.NONE)

    def dict(self) -> dict[str, Any]:
        """Serialize the object to a dict."""

        if self.model != InternalModelRdModel.NONE:
            raise RuntimeError(f'Unhandled model {self.model}')

        return {'model': 'none'}


@dataclass(kw_only=True)
class InternalModelParamsRdDiv(InternalModelParamsRd):
    """Parameters of radial distortion division model."""
    model: InternalModelRdModel = InternalModelRdModel.DIV
    lam: float = 0.0
    dn: float = 1.0

    @classmethod
    def from_dict(cls, d):
        """De-serialize the object from a dict."""

        if (d.keys() != {'model', 'lam', 'dn'}
                or d['model'] not in ('div', 'idiv')):
            raise ValueError('Wrong dict data:', d)

        return cls(model=InternalModelRdModel.DIV if d['model'] == 'div'
        else InternalModelRdModel.IDIV,
                   lam=d['lam'], dn=d['dn'])

    def dict(self) -> dict[str, Any]:
        """Serialize the object to a dict."""

        if self.model == InternalModelRdModel.DIV:
            m = 'div'
        elif self.model == InternalModelRdModel.IDIV:
            m = 'idiv'
        else:
            raise RuntimeError(f'Unhandled model {self.model}')

        return {'model': m, 'dn': self.dn, 'lam': self.lam}


@dataclass(kw_only=True)
class InternalModelParamsRdPoly(InternalModelParamsRd):
    """Parameters of the radial and tangential distortion polynomial model."""
    model: InternalModelRdModel = InternalModelRdModel.POLY
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @classmethod
    def from_dict(cls, d):
        """De-serialize the object from a dict."""

        if (d.keys() != {'model', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2'}
                or d['model'] not in ('poly', 'ipoly')):
            raise ValueError('Wrong dict data:', d)

        return cls(model=InternalModelRdModel.IPOLY if d['model'] == 'ipoly'
        else InternalModelRdModel.POLY,
                   k1=d['k1'], k2=d['k2'], k3=d['k3'], k4=d['k4'],
                   p1=d['p1'], p2=d['p2'])

    def dict(self) -> dict[str, Any]:
        """Serialize the object to a dict."""

        if self.model == InternalModelRdModel.POLY:
            m = 'poly'
        elif self.model == InternalModelRdModel.IPOLY:
            m = 'ipoly'
        else:
            raise RuntimeError(f'Unhandled model {self.model}')

        return {'model': m, 'k1': self.k1, 'k2': self.k2, 'k3': self.k3,
                'k4': self.k4, 'p1': self.p1, 'p2': self.p2}


InternalModelParamsRdTypes = Union[
    InternalModelParamsRd,
    InternalModelParamsRdDiv,
    InternalModelParamsRdPoly]


@dataclass(kw_only=True)
class InternalModelParams:
    w: int = None
    h: int = None
    proj: Optional[InternalModelParamsProjTypes] = None
    rd: Optional[InternalModelParamsRdTypes] = None
    k: Optional[ParamK] = None

    @classmethod
    def from_dict(cls, d):
        """De-serialize the object from a dict."""

        if d.keys() != {'w', 'h', 'proj', 'rd', 'k'}:
            raise ValueError('Wrong dict data:', d)

        return cls(w=d['w'],
                   h=d['h'],
                   proj=InternalModelParamsProj.from_dict(d['proj']),
                   rd=InternalModelParamsRd.from_dict(d['rd']),
                   k=ParamK.from_dict(d['k']))

    def dict(self) -> dict[str, Any]:
        """Serialize the object to a dict."""

        return {'w': self.w,
                'h': self.h,
                'proj': self.proj.dict(),
                'rd': self.rd.dict(),
                'k': self.k.dict()}

class InternalModelHandles:
    x2l: Callable[[InternalModelParams], Projection32] = None
    l2x: Callable[[InternalModelParams], Projection23] = None
    x2p: Callable[[InternalModelParams], Projection32] = None
    p2x: Callable[[InternalModelParams], Projection23] = None
    x2u: Callable[[InternalModelParams], Projection32] = None
    u2x: Callable[[InternalModelParams], Projection23] = None
    p2u: Callable[[InternalModelParams], Projection22] = None
    u2p: Callable[[InternalModelParams], Projection22] = None
    p2l: Callable[[InternalModelParams], Projection22] = None
    l2p: Callable[[InternalModelParams], Projection22] = None
    u2l: Callable[[InternalModelParams], Projection22] = None
    l2u: Callable[[InternalModelParams], Projection22] = None
    K: Matrix33 = None


class InternalModel:
    x2l: Projection32 = None
    l2x: Projection23 = None
    x2p: Projection32 = None
    p2x: Projection23 = None
    x2u: Projection32 = None
    u2x: Projection23 = None
    p2u: Projection22 = None
    u2p: Projection22 = None
    p2l: Projection22 = None
    l2p: Projection22 = None
    u2l: Projection22 = None
    l2u: Projection22 = None
    K: Matrix33 = None


def div_bck(u: Matrix2N, lam: float, dn: float) -> Matrix2N:
    """Radial distortion in plane, division model - backward (inverse).
    ::

        ud = div_bck(u, lam, dn)

    :param u: coordinates of plane points; 2xn matrix of column vectors
    :param lam: radial distortion parameter of the division model
    :param dn: fixed point of the radial function of the distortion

    :return: transformed coordinates; 2xn matrix of column vectors
    """

    ll = 2.0 / (1 - lam)
    ll2_lam_dn2 = ll**2 * lam / dn**2
    ll_lam_dn2 = ll * lam / dn**2

    r2 = u[0]**2 + u[1]**2

    q = 1.0 + ll2_lam_dn2 * r2
    ko = q < 0.0

    q = (-1.0 + np.sqrt(q)) / r2 / ll_lam_dn2

    ud = u * q

    ud[:, ko] = np.nan
    return ud


def div_fwd(u: Matrix2N, lam: float, dn: float) -> Matrix2N:
    """Radial distortion in plane, division model - forward.
    ::

        ud = div_fwd(u, lam, dn)

    :param u: coordinates of plane points; 2xn matrix of column vectors
    :param lam: radial distortion parameter of the division model
    :param dn: fixed point of the radial function of the distortion

    :return: transformed coordinates; 2xn matrix of column vectors
    """

    r2 = u[0]**2 + u[1]**2
    q = (1 - lam) / (1 - lam * r2 / dn**2)

    ud = u * q
    return ud


def poly_bck(ud: Matrix2N, rd: List[float], td: List[float],
             tol: float = 1e-7,
             iter_max: int = 50) -> Matrix2N:
    """Radial distortion in plane, polynomial model - inverse (iterative).
    ::

        u = poly_bck(ud, rd, td, tol, iter_max)

    :param ud: coordinates of plane points; 2xn matrix of column vectors
    :param rd: radial distortion coefficients; [k1, k2, k3, k4]
    :param td: tangential distortion coefficients; [p1, p2]
    :param tol: maximal (absolute) err. of numeric inversion (stops iterations)
    :param iter_max: maximal number of iterations

    :return: transformed coordinates; 2xn matrix of column vectors
    """

    # inverse not implemented when tangential distortion is used
    if len(td) > 0 and (td[0] != 0.0 or td[1] != 0.0):
        # TODO inverse with tangential distortion
        raise NotImplementedError(
            'Inverse of tangential distortion not implemented')

    # polynom coefficients and derivative; use the smallest order as possible
    if len(rd) > 3 and rd[3] != 0.0:
        order = 4
        k1, k2, k3, k4 = rd[0], rd[1], rd[2], rd[3]
        poly_der = [9. * k4, 7. * k3, 5. * k2, 3. * k1, 1.]
    elif len(rd) > 2 and rd[2] != 0.0:
        order = 3
        k1, k2, k3, k4 = rd[0], rd[1], rd[2], 0.
        poly_der = [7. * k3, 5. * k2, 3. * k1, 1.]
    elif len(rd) > 1 and rd[1] != 0.0:
        order = 2
        k1, k2, k3, k4 = rd[0], rd[1], 0., 0.
        poly_der = [5. * k2, 3. * k1, 1]
    else:
        order = 1
        k1, k2, k3, k4 = rd[0], 0., 0., 0.
        poly_der = [3. * k1, 1.]

    # upper bound of polynom domain; below the bound it is an increasing
    # function: find a point where derivative is zero
    # TODO probably wrong - derivation must be in r and not r^2
    mx = np.roots(poly_der)
    good = (np.imag(mx) == 0) * (np.real(mx) > 0)

    # coordinates and radius
    x, y = ud[0], ud[1]
    r = np.sqrt(x**2 + y**2)

    if good.any():
        mx = mx[good].min()
        r[r > mx] = np.nan

    r_out = r

    for i in range(0, iter_max):
        if order == 4:
            q = (1 + k1 * r_out**2 + k2 * r_out**4 + k3 * r_out**6 +
                 k4 * r_out**8)
        elif order == 3:
            q = 1 + k1 * r_out**2 + k2 * r_out**4 + k3 * r_out**6
        elif order == 2:
            q = 1 + k1 * r_out**2 + k2 * r_out**4
        else:  # order == 1
            q = 1 + k1 * r_out**2

        err = r - r_out * q

        if not np.greater(np.abs(err), tol).any():  # negative test is nan-safe
            # compute output - it is one more iteration, so the result is even
            # better than required by tol
            return ud / q

        r_out = r / q

    # polynom inverse has not converged after `iter_max` iterations
    return np.full_like(ud, np.nan)


def poly_fwd(u: Matrix2N, rd: List[float], td: List[float]) -> Matrix2N:
    """Radial and tangential dist. in plane, polynomial model - forward.
    ::

        ud = poly_fwd(u, rd, td)

    :param u: coordinates of plane points; 2xn matrix of column vectors
    :param rd: radial distortion coefficients; [k1, k2, k3, k4]
    :param td: tangential distortion coefficients; [p1, p2]

    :return: transformed coordinates; 2xn matrix of column vectors
    """

    x, y = u[0], u[1]

    x2 = x**2
    y2 = y**2
    r2 = x2 + y2

    # radial distortion
    if len(rd) > 3 and rd[3] != 0.0:
        q = 1 + rd[0] * r2 + rd[1] * r2**2 + rd[2] * r2**3 + rd[3] * r2**4
    elif len(rd) > 2 and rd[2] != 0.0:
        q = 1 + rd[0] * r2 + rd[1] * r2**2 + rd[2] * r2**3
    elif len(rd) > 1 and rd[1] != 0.0:
        q = 1 + rd[0] * r2 + rd[1] * r2**2
    else:
        q = 1 + rd[0] * r2

    ud = u * q

    # tangential distortion
    if len(td) > 0 and td[0] != 0.0 and td[1] != 0.0:
        xy = x * y

        ud[0] += 2 * td[0] * xy + td[1] * (r2 + 2 * x2)
        ud[1] += 2 * td[1] * xy + td[0] * (r2 + 2 * y2)

    return ud


def p2u_div_k(p: Matrix2N, rd: InternalModelParamsRdDiv, k: ParamK) -> Matrix2N:
    """Camera to image plane transformation with div RD and K.
    ::

        u = p2u_div_k(p, rd, k)

    :param p: coordinates of camera plane points; 2xn matrix of column vectors
    :param rd: radial distortion parameters for the division model
    :param k: internal calibration parameters

    :return: coordinates of image plane points; 2xn matrix of column vectors
    """

    return k.apply(div_fwd(p, rd.lam, rd.dn))


def p2u_idiv_k(p: Matrix2N, rd: InternalModelParamsRdDiv, k: ParamK
               ) -> Matrix2N:
    """Camera to image plane transformation with idiv RD and K.
    ::

        u = p2u_idiv_k(p, rd, k)

    :param p: coordinates of camera plane points; 2xn matrix of column vectors
    :param rd: radial distortion parameters for the division model
    :param k: internal calibration parameters

    :return: coordinates of image plane points; 2xn matrix of column vectors
    """

    return k.apply(div_bck(p, rd.lam, rd.dn))


def p2u_ipoly_k(p: Matrix2N, rd: InternalModelParamsRdPoly, k: ParamK
                ) -> Matrix2N:
    """Camera to image plane transformation with iRD and K.
    ::

        u = p2u_ipoly_k(p, rd, k)

    :param p: coordinates of camera plane points; 2xn matrix of column vectors
    :param rd: radial distortion parameters for the polynomial model
    :param k: internal calibration parameters

    :return: coordinates of image plane points; 2xn matrix of column vectors
    """

    return k.apply(poly_bck(p, [rd.k1, rd.k2, rd.k3, rd.k4], [rd.p1, rd.p2]))


def p2u_k(p: Matrix2N, k: ParamK) -> Matrix2N:
    """Camera to image plane transformation with K.
    ::

        u = p2u_k(p, k)

    :param p: coordinates of camera plane points; 2xn matrix of column vectors
    :param k: internal calibration parameters; [fx, fy, dx, dy, q]

    :return: coordinates of image plane points; 2xn matrix of column vectors
    """

    return k.apply(p)


def p2u_poly_k(p: Matrix2N, rd: InternalModelParamsRdPoly, k: ParamK
               ) -> Matrix2N:
    """Camera to image plane transformation with poly RD and K.
    ::

        u = p2u_poly_k(p, rd, k)

    :param p: coordinates of camera plane points; 2xn matrix of column vectors
    :param rd: radial distortion parameters for the polynomial model
    :param k: internal calibration parameters

    :return: coordinates of image plane points; 2xn matrix of column vectors
    """

    return k.apply(poly_fwd(p, [rd.k1, rd.k2, rd.k3, rd.k4], [rd.p1, rd.p2]))


def u2p_div_k(u: Matrix2N, rd: InternalModelParamsRdDiv, k: ParamK) -> Matrix2N:
    """Image to camera plane transformation with div RD and K.
    ::

        u = u2p_div_k(p, rd, k)

    :param u: coordinates of image plane points; 2xn matrix of column vectors
    :param rd: radial distortion parameters for the division model
    :param k: internal calibration parameters

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """

    return div_bck(k.inv().apply(u), rd.lam, rd.dn)


def u2p_idiv_k(u: Matrix2N, rd: InternalModelParamsRdDiv, k: ParamK
               ) -> Matrix2N:
    """Image to camera plane transformation with div RD and K.
    ::

        u = u2p_div_k(p, rd, k)

    :param u: coordinates of image plane points; 2xn matrix of column vectors
    :param rd: radial distortion parameters for the division model
    :param k: internal calibration parameters

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """

    return div_fwd(k.inv().apply(u), rd.lam, rd.dn)


def u2p_ipoly_k(u: Matrix2N, rd: InternalModelParamsRdPoly, k: ParamK
                ) -> Matrix2N:
    """Image to camera plane transformation with ipoly RD and K.
    ::

        p = u2p_ipoly_k(u, rd, k)

    :param u: coordinates of image plane points; 2xn matrix of column vectors
    :param rd: radial distortion coefficients of the polynomial model
    :param k: internal calibration parameters

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """

    return poly_fwd(k.inv().apply(u), [rd.k1, rd.k2, rd.k3, rd.k4],
                    [rd.p1, rd.p2])


def u2p_k(u: Matrix2N, k: ParamK) -> Matrix2N:
    """Image to camera plane transformation with K.
    ::

        p = u2p_k(p, k)

    :param u: coordinates of image plane points; 2xn matrix of column vectors
    :param k: internal calibration parameters; [fx, fy, dx, dy, q]

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """

    return k.inv().apply(u)


def u2p_poly_k(u: Matrix2N, rd: InternalModelParamsRdPoly, k: ParamK
               ) -> Matrix2N:
    """Image to camera plane transformation with poly RD and K.
    ::

        p = u2p_poly_k(u, rd, k)

    :param u: coordinates of image plane points; 2xn matrix of column vectors
    :param rd: radial distortion coefficients for the polynomial model
    :param k: internal calibration parameters

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """

    return poly_bck(k.inv().apply(u), [rd.k1, rd.k2, rd.k3, rd.k4],
                    [rd.p1, rd.p2])


def x2p_eq(x: Matrix3N) -> Matrix2N:
    """Projection - equi-angular model.
    ::

        u = x2p_eq(x)

    :param x: coordinates of rays; 3xn matrix of column vectors

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """

    x, y, z = x[0], x[1], x[2]

    r = np.sqrt(x**2 + y**2)

    angle = np.arctan2(r, z)

    u = x / r * angle
    v = y / r * angle

    u[r == 0] = 0
    v[r == 0] = 0

    return np.vstack((u, v))


def x2p_tan(x: Matrix3N) -> Matrix2N:
    """Projection - perspective (tangential) model.
    ::

        u = x2p_tan(x)

    :param x: coordinates of rays; 3xn matrix of column vectors

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """
    u, v, w = x[0], x[1], x[2]

    u = u / w
    v = v / w

    # limit to points in front of the camera (x_z > 0) and with the view
    # direction in (-89, 89) degrees interval (tan(89deg)^2 = 3282)
    ko = (w <= 0.0) | (u**2 + u**2 > 3282.0)

    u[ko] = np.nan
    v[ko] = np.nan

    return np.vstack((u, v))


def x2p_uni(x: Matrix3N, xi: float) -> Matrix2N:
    """Projection - unified central panoramic model.
    ::

        u = x2p_uni(x)

    :param x: coordinates of rays; 3xn matrix of column vectors
    :param xi: omni parameter

    :return: coordinates of camera plane points; 2xn matrix of column vectors
    """

    if xi <= -1.0:
        # We do not allow xi <= -1, though a projection exists for xi < -1.
        return np.full((2, x.shape[1]), np.nan)

    x, y, z = x[0], x[1], x[2]

    # transformation by a 'mirror' - shift by xi
    r = np.sqrt(x**2 + y**2 + z**2)
    den = z + xi * r

    u = (1 + xi) * x / den
    v = (1 + xi) * y / den

    # Note: 1+xi is a scaling factor ensuring derivative w.r.t. alpha
    # at the origin to be 1.

    # Note: for xi < -1 both 1+xi and den are negative (as r >= z), thus
    # leading to correct sign of u, v (same as the sign of x, y). But we
    # do not allow xi <= 0 anyway.

    # visibility
    if xi > 1.0:  # xi outside the unit ball
        ko = z <= -r / xi
    else:  # xi in <-1, 1>, inside the unit ball
        ko = den <= 0

    u[ko] = np.nan
    v[ko] = np.nan

    return np.vstack((u, v))


def p2x_eq(p: Matrix2N) -> Matrix3N:
    """Back-projection - equi-angular model.
    ::

        x = p2x_eq(p)

    :param p: coordinates of camera plane points; 2xn matrix of column vectors

    :return: coordinates of rays; 3xn matrix of unit column vectors
    """

    u, v = p[0], p[1]

    angle = np.sqrt(u**2 + v**2)

    # radii larger than pi cannot be back-projected to a unique ray
    ko = angle >= np.pi

    x = u / angle * np.sin(angle)
    y = v / angle * np.sin(angle)
    z = np.cos(angle)

    x[angle == 0] = 0
    y[angle == 0] = 0

    x[ko] = np.nan
    y[ko] = np.nan
    z[ko] = np.nan

    return np.vstack((x, y, z))


def p2x_tan(p: Matrix2N) -> Matrix3N:
    """Back-projection - perspective (tangential) model.
    ::

        x = p2x_tan(p)

    :param p: coordinates of camera plane points; 2xn matrix of column vectors

    :return: coordinates of rays; 3xn matrix of unit column vectors
    """

    r = np.sqrt(p[0]**2 + p[1]**2 + 1)  # normalization to unit length
    return np.vstack((p[0] / r, p[1] / r, 1 / r))


def p2x_uni(p: Matrix2N, xi: float) -> Matrix3N:
    """Back-projection - unified central panoramic model.
    ::

        x = p2x_uni(p)

    Projection model described in [Geyer-ECCV2000], also used in [Mei-ICRA2007].

    :param p: coordinates of camera plane points; 2xn matrix of column vectors
    :param xi: omni parameter

    :return: coordinates of rays; 3xn matrix of unit column vectors
    """

    if xi <= -1.0:
        # We do not allow xi <= -1, though for xi < -1 a projection exists.
        return np.full((3, p.shape[1]), np.nan)

    # (1+xi) is a scaling factor ensuring derivative w.r.t. alpha
    # at the origin to be 1.

    u, v = p[0] / (1 + xi), p[1] / (1 + xi)

    r2 = u**2 + v**2

    z = (-r2 * xi + np.sqrt(r2 - r2 * xi**2 + 1)) / (1 + r2)

    x = u * (z + xi)
    y = v * (z + xi)

    # allowed range for xi outside the unit ball
    if xi > 1:
        ko = r2 >= 1 / (xi**2 - 1)
        x[ko] = np.nan
        y[ko] = np.nan
        z[ko] = np.nan

    return np.vstack((x, y, z))


def internal_model_handles(proj_model: InternalModelProjModel,
                           rd_model: InternalModelRdModel
                           ) -> InternalModelHandles:
    """TODO"""

    f = InternalModelHandles()

    # All must be function handles parameterized by c: InternalModelParams,
    # even when a particular transformation has no parameters (e.g., tan or eq.
    # projection).

    # x to l projection is allways the same
    f.x2l = lambda c: lambda x: x2p_tan(x)
    f.l2x = lambda c: lambda p: p2x_tan(p)

    # projection transformations (does not depend on rd_model)
    match proj_model:
        case InternalModelProjModel.TAN:
            f.x2p = f.x2l
            f.p2x = f.l2x

            f.l2p = lambda c: lambda p: p
            f.p2l = lambda c: lambda p: p

        case InternalModelProjModel.EQ:
            f.x2p = lambda c: lambda x: x2p_eq(x)
            f.p2x = lambda c: lambda p: p2x_eq(p)

            f.l2p = lambda c: lambda p: x2p_eq(p2x_tan(p))
            f.p2l = lambda c: lambda p: x2p_tan(p2x_eq(p))

        case InternalModelProjModel.UNI:
            f.x2p = lambda c: lambda x: x2p_uni(x, c.proj.xi)
            f.p2x = lambda c: lambda p: p2x_uni(p, c.proj.xi)

            f.l2p = lambda c: lambda p: x2p_uni(p2x_tan(p), c.proj.xi)
            f.p2l = lambda c: lambda p: x2p_tan(p2x_uni(p, c.proj.xi))

        case _:
            raise Exception('Wrong projection model ' + str(proj_model))

    # transformations for RD (does not depend on projection)
    if rd_model == InternalModelRdModel.NONE:
        f.p2u = lambda c: lambda p: p2u_k(p, c.k)
        f.u2p = lambda c: lambda u: u2p_k(u, c.k)

    elif rd_model == InternalModelRdModel.DIV:
        f.p2u = lambda c: lambda p: p2u_div_k(p, c.rd, c.k)
        f.u2p = lambda c: lambda u: u2p_div_k(u, c.rd, c.k)

    elif rd_model == InternalModelRdModel.IDIV:
        f.p2u = lambda c: lambda p: p2u_idiv_k(p, c.rd, c.k)
        f.u2p = lambda c: lambda u: u2p_idiv_k(u, c.rd, c.k)

    elif rd_model == InternalModelRdModel.POLY:
        f.p2u = lambda c: lambda p: p2u_poly_k(p, c.rd, c.k)
        f.u2p = lambda c: lambda u: u2p_poly_k(u, c.rd, c.k)

    elif rd_model == InternalModelRdModel.IPOLY:
        f.p2u = lambda c: lambda p: p2u_ipoly_k(p, c.rd, c.k)
        f.u2p = lambda c: lambda u: u2p_ipoly_k(u, c.rd, c.k)

    else:
        raise ValueError('Wrong RD model ' + str(proj_model))

    # combinations of RD and projection
    f.x2u = lambda c: lambda x: f.p2u(c)(f.x2p(c)(x))
    f.u2x = lambda c: lambda u: f.p2x(c)(f.u2p(c)(u))

    f.l2u = lambda c: lambda p: f.p2u(c)(f.l2p(c)(p))
    f.u2l = lambda c: lambda u: f.p2l(c)(f.u2p(c)(u))

    return f


def internal_model(c: InternalModelParams) -> InternalModel:
    """Internal projection model function factory.
    ::

        f = internal_model(c)

    :param c: internal calibration parameters - object with members
        c.proj['model'] - 'tan' or 'eq' or 'mei
        c.proj['proj']['xi'] - omni parameter in the case of mei model

        c['rd'] - distortion model; the field ['model'] and additional params

          model = 'none' - no distortion

          model = 'div' - division model, forward is in projection
            'l1', 'l2', 'l3'

          model = 'idiv' - division model, forward is in back-projection
            'l1', 'l2', 'l3'

          model = 'poly' - polynomial model, forward is in projection
            'k1', 'k2', 'k3', 'k4', 'p1', 'p2'

          model = 'ipoly' - polynomial model, forward is in projection
            'k1', 'k2', 'k3', 'p1', 'p2'

        c['k'] .. image coordinate system (i.e. internal calibration matrix);
                  fields: 'f', 'x0', 'y0'

    :returns: dictionary with handles of projecting and back-projecting functions.
        The functions transform between
        several domains in the internal camera projection model:

          x - 3D points/rays in the camera coordinate system,
          p - 2D points in the camera plane coordinate system,
          u - 2D points (pixels) in the image coordinate system,
          l - 2D points in the linear perspective camera coordinate system.

        Note that for the perspective (tangential) projection, the domain L
        is identical to U, for other projections it differs.
        For some types of projections or distortion models, not all functions
        (domain combinations) may be available.

        Every function is called as::

            b = f.fcn( a )

        where a and b are 2xn or 3xn matrices of 2D or 3D Euclidean vector
        coordinates.

        Available functions (fields):
             'X2u'
             'u2X'
             'u2U'
             'U2u'
             'u2L'
             'L2u'

           Additionally, the internal calibration matrix is available as
             f['K'] (but note, that is allready employed in f.X2u, e.g.)
   """

    f = InternalModel()
    fh = internal_model_handles(c.proj.model, c.rd.model)

    # TODO assert all params not None

    f.K = c.k.matrix()

    f.x2l = fh.x2l(c)
    f.l2x = fh.l2x(c)
    f.x2p = fh.x2p(c)
    f.p2x = fh.p2x(c)
    f.x2u = fh.x2u(c)
    f.u2x = fh.u2x(c)
    f.p2u = fh.p2u(c)
    f.u2p = fh.u2p(c)
    f.p2l = fh.p2l(c)
    f.l2p = fh.l2p(c)
    f.u2l = fh.u2l(c)
    f.l2u = fh.l2u(c)

    return f
