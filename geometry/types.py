"""Common data types for this package."""

# (c) 2024 Martin Matousek
# Last change: $Date$
#              $Revision$


# if we have the mathlib package, use the matrix/vector types from there
try:
    from mathlib.mat_vec_types import *
    from mathlib.mat_vec_types import __all__

except ModuleNotFoundError:
    from geometry.mat_vec_types import *
    from geometry.mat_vec_types import __all__

from typing import Callable

__all__ = __all__.copy()
__all__ += ["Projection32", "Projection22", "Projection23"]

Projection32 = Callable[[Matrix3N], Matrix2N]
Projection22 = Callable[[Matrix2N], Matrix2N]
Projection23 = Callable[[Matrix2N], Matrix3N]
