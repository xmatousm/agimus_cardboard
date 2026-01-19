"""Convenience numpy matrix and vector types."""

# (c) 2025-12-10 Martin Matousek
# Last change: $Date$
#              $Revision$

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

# list only the types defined here (and not imported) for use by 'import *'
__all__ = ["Vector", "Vector2", "Vector3",
           "VectorI", "VectorB",
           "ColVector", "ColVector2", "ColVector3",
           "RowVector",
           "Matrix", "Matrix33", "Matrix22", "Matrix2N", "Matrix3N",
           "MatrixI",
           ]

method = 0

if method == 0:
    Vector = np.ndarray
    Vector2 = np.ndarray
    Vector3 = np.ndarray

    VectorI = np.ndarray
    VectorB = np.ndarray

    ColVector = np.ndarray
    ColVector2 = np.ndarray
    ColVector3 = np.ndarray

    RowVector = np.ndarray

    Matrix = np.ndarray
    Matrix33 = np.ndarray
    Matrix22 = np.ndarray
    Matrix2N = np.ndarray
    Matrix3N = np.ndarray

    MatrixI = np.ndarray

elif method == 1:
    Vector = np.ndarray[tuple[int, ...], np.dtype[float]]
    Vector2 = np.ndarray[tuple[int, ...], np.dtype[float]]
    Vector3 = np.ndarray[tuple[int, ...], np.dtype[float]]

    VectorI = np.ndarray[tuple[int, ...], np.dtype[int]]
    VectorB = np.ndarray[tuple[int, ...], np.dtype[bool]]

    ColVector = np.ndarray[tuple[int, ...], np.dtype[float]]
    ColVector2 = np.ndarray[tuple[int, ...], np.dtype[float]]
    ColVector3 = np.ndarray[tuple[int, ...], np.dtype[float]]

    RowVector = np.ndarray[tuple[int, ...], np.dtype[float]]

    Matrix = np.ndarray[tuple[int, ...], np.dtype[float]]
    Matrix33 = np.ndarray[tuple[int, ...], np.dtype[float]]
    Matrix22 = np.ndarray[tuple[int, ...], np.dtype[float]]
    Matrix2N = np.ndarray[tuple[int, ...], np.dtype[float]]
    Matrix3N = np.ndarray[tuple[int, ...], np.dtype[float]]

    MatrixI = np.ndarray[tuple[int, ...], np.dtype[int]]

elif method == 2:
    Vector = Annotated[npt.NDArray[float], Literal["N"]]
    Vector2 = Annotated[npt.NDArray[float], Literal[2]]
    Vector3 = Annotated[npt.NDArray[float], Literal[3]]

    VectorI = Annotated[npt.NDArray[np.intp], Literal["N"]]
    VectorB = Annotated[npt.NDArray[bool], Literal["N"]]

    ColVector = Annotated[npt.NDArray[float], Literal["N", 1]]
    ColVector2 = Annotated[npt.NDArray[float], Literal[2, 1]]
    ColVector3 = Annotated[npt.NDArray[float], Literal[3, 1]]

    RowVector = Annotated[npt.NDArray[np.intp], Literal[1, "N"]]

    Matrix = Annotated[npt.NDArray[float], Literal["N", "N"]]
    Matrix33 = Annotated[npt.NDArray[float], Literal[3, 3,]]
    Matrix22 = Annotated[npt.NDArray[float], Literal[2, 2,]]
    Matrix2N = Annotated[npt.NDArray[float], Literal[2, "N"]]
    Matrix3N = Annotated[npt.NDArray[float], Literal[3, "N"]]

    MatrixI = Annotated[npt.NDArray[np.intp], Literal["N", "N"]]

else:
    raise RuntimeError("Internal error")
