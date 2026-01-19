"""Parameterization of an internal camera model using a vector of selected
   parameters.
"""

# (c) 2025-09-09 Martin Matousek
# Last change: $Date$
#              $Revision$


from typing import Optional, Callable

import numpy as np

import geometry.projection as proj
from geometry.types import Vector


def _check(c: proj.InternalModelParams,
           rd: Optional[list[proj.InternalModelRdModel]] = None,
           pm: Optional[proj.InternalModelProjModel] = None):
    # check a RD model if required
    if rd is not None:
        if c.rd is None or c.rd.model not in rd:
            raise ValueError(
                f"Parameterization does not match the RD model {c.rd}")

    # check a projection model if required
    if pm is not None:
        if c.proj is None or c.proj.model != pm:
            raise ValueError(
                f"Parameterization does not match the projection model {c}")


def _decorate_uni(c: proj.InternalModelParams, p0_p2c):
    _check(c, None, proj.InternalModelProjModel.UNI)
    p0, p2c = p0_p2c

    def p2c_uni(p: Vector) -> proj.InternalModelParams:
        cu = p2c(p)
        cu.proj.xi = p[-1]
        return cu

    return np.hstack((p0, c.proj.xi)), p2c_uni


# The fields of c (that is bound to the inner p2c function handle) are modified
# in place, so those that are not part of the parameter vector are either kept
# unchanged or initialized (based on the model used).

# Each parameterization function checks the fields in c, initializes those
# needed, and returns the initial parameter vector p0, and the p2c function.
# This function fills appropriate fields from its input vector to the
# object c and returns it.

P2CType = Callable[[Vector], proj.InternalModelParams]


# functions for no radial distortion

def none_f(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.NONE])

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        return c

    return np.array([c.k.fx]), p2c


def none_f_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.NONE],
           proj.InternalModelProjModel.UNI)

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.proj.xi = p[1]
        return c

    return np.array([c.k.fx, c.proj.xi]), p2c


def none_uf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.NONE])

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy]), p2c


def none_uf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.NONE],
           proj.InternalModelProjModel.UNI)

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.proj.xi = p[3]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.proj.xi]), p2c


# functions for the division model of radial distortion
# (parameterization of DIV and IDIV is the same)

def div_l(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.DIV, proj.InternalModelRdModel.IDIV])

    def p2c(p) -> proj.InternalModelParams:
        c.rd.lam = p[0]
        return c

    return np.array([c.rd.lam]), p2c


def div_lu(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.DIV, proj.InternalModelRdModel.IDIV])

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.rd.lam = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        return c

    return np.array([c.rd.lam, c.k.dx, c.k.dy]), p2c


def div_luf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.DIV, proj.InternalModelRdModel.IDIV])

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.rd.lam = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.k.fx = p[3]
        c.k.fy = p[3]
        return c

    return np.array([c.rd.lam, c.k.dx, c.k.dy, c.k.fx]), p2c


def div_l_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.DIV, proj.InternalModelRdModel.IDIV],
           proj.InternalModelProjModel.UNI)

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.rd.lam = p[0]
        c.proj.xi = p[1]
        return c

    return np.array([c.rd.lam, c.proj.xi]), p2c


def div_lu_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.DIV, proj.InternalModelRdModel.IDIV],
           proj.InternalModelProjModel.UNI)

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.rd.lam = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.proj.xi = p[3]
        return c

    return np.array([c.rd.lam, c.k.dx, c.k.dy, c.proj.xi]), p2c


def div_luf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.DIV, proj.InternalModelRdModel.IDIV],
           proj.InternalModelProjModel.UNI)

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.rd.lam = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.k.fx = p[3]
        c.k.fy = p[3]
        c.proj.xi = p[4]
        return c

    return np.array([c.rd.lam, c.k.dx, c.k.dy, c.k.fx, c.proj.xi]), p2c


# functions for the polynomial model of radial only distortion
# (parameterization of POLY and IPOLY is the same)

def poly_1u(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k2 = 0.
    c.rd.k3 = 0.
    c.rd.k4 = 0.
    c.rd.p1 = 0.
    c.rd.p2 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.dx = p[0]
        c.k.dy = p[1]
        c.rd.k1 = p[2]
        return c

    return np.array([c.k.dx, c.k.dy, c.rd.k1]), p2c


def poly_1uf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k2 = 0.
    c.rd.k3 = 0.
    c.rd.k4 = 0.
    c.rd.p1 = 0.
    c.rd.p2 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1]), p2c


def poly_1uf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, None, proj.InternalModelProjModel.UNI)
    p0, p2c = poly_1uf(c)

    def p2c_uni(p: Vector) -> proj.InternalModelParams:
        cu = p2c(p)
        cu.proj.xi = p[4]
        return cu

    return np.hstack((p0, c.proj.xi)), p2c_uni


def poly_2(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k3 = 0.
    c.rd.k4 = 0.
    c.rd.p1 = 0.
    c.rd.p2 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.rd.k1 = p[0]
        c.rd.k2 = p[1]
        return c

    return np.array([c.rd.k1, c.rd.k2]), p2c


def poly_2u(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k3 = 0.
    c.rd.k4 = 0.
    c.rd.p1 = 0.
    c.rd.p2 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.dx = p[0]
        c.k.dy = p[1]
        c.rd.k1 = p[2]
        c.rd.k2 = p[3]
        return c

    return np.array([c.k.dx, c.k.dy, c.rd.k1, c.rd.k2]), p2c


def poly_2uf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k3 = 0.
    c.rd.k4 = 0.
    c.rd.p1 = 0.
    c.rd.p2 = 0.

    def p2c(p) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        c.rd.k2 = p[4]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1, c.rd.k2]), p2c


def poly_2uf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    return _decorate_uni(c, poly_2uf(c))


def poly_3uf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k4 = 0.
    c.rd.p1 = 0.
    c.rd.p2 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        c.rd.k2 = p[4]
        c.rd.k3 = p[5]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1, c.rd.k2, c.rd.k3]), p2c


def poly_3uf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    return _decorate_uni(c, poly_3uf(c))


def poly_4uf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.p1 = 0.
    c.rd.p2 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        c.rd.k2 = p[4]
        c.rd.k3 = p[5]
        c.rd.k4 = p[6]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1, c.rd.k2, c.rd.k3,
                     c.rd.k4]), p2c


def poly_4uf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    return _decorate_uni(c, poly_4uf(c))


# functions for the polynomial model of radial and tangential distortion
# (parameterization of POLY and IPOLY is the same)

def poly_1tuf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k2 = 0.
    c.rd.k3 = 0.
    c.rd.k4 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        c.rd.p1 = p[4]
        c.rd.p2 = p[5]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1, c.rd.p1, c.rd.p2]), p2c


def poly_1tuf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    return _decorate_uni(c, poly_1tuf(c))


def poly_2tuf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k3 = 0.
    c.rd.k4 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        c.rd.k2 = p[4]
        c.rd.p1 = p[5]
        c.rd.p2 = p[6]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1, c.rd.k2,
                     c.rd.p1, c.rd.p2]), p2c


def poly_2tuf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    return _decorate_uni(c, poly_2tuf(c))


def poly_3tuf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    c.rd.k4 = 0.

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        c.rd.k2 = p[4]
        c.rd.k3 = p[5]
        c.rd.p1 = p[6]
        c.rd.p2 = p[7]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1, c.rd.k2, c.rd.k3,
                     c.rd.p1, c.rd.p2]), p2c


def poly_3tuf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    return _decorate_uni(c, poly_3tuf(c))


def poly_4tuf(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    _check(c, [proj.InternalModelRdModel.POLY, proj.InternalModelRdModel.IPOLY])

    def p2c(p: Vector) -> proj.InternalModelParams:
        c.k.fx = p[0]
        c.k.fy = p[0]
        c.k.dx = p[1]
        c.k.dy = p[2]
        c.rd.k1 = p[3]
        c.rd.k2 = p[4]
        c.rd.k3 = p[5]
        c.rd.k4 = p[6]
        c.rd.p1 = p[7]
        c.rd.p2 = p[8]
        return c

    return np.array([c.k.fx, c.k.dx, c.k.dy, c.rd.k1, c.rd.k2, c.rd.k3,
                     c.rd.k4, c.rd.p1, c.rd.p2]), p2c


def poly_4tuf_uni(c: proj.InternalModelParams) -> tuple[Vector, P2CType]:
    return _decorate_uni(c, poly_4tuf(c))
