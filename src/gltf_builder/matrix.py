'''
4x4 matrix class and operations.
'''

from typing import TypeAlias, Literal, Generic, TypeVar, cast, overload, Self, Any

import numpy as np

import gltf_builder.attribute_types as at
from gltf_builder.core_types import Scalar


MatrixDims: TypeAlias = Literal[2, 3, 4]
DIMS = TypeVar('DIMS', bound=MatrixDims)
'''Number of dimensions in the matrix.'''

class Matrix(Generic[DIMS]):
    '''
    A 2x2, 3x3, or 4x4 matrix.
    '''
    _data: np.ndarray[tuple[DIMS, DIMS], np.dtype[np.float32]]

    def __init__(self, data: 'MatrixSpec', nocopy:bool = False):
        match data:
            case np.ndarray():
                if data.dtype != np.float32:
                    arr = data.astype(np.float32)
                elif nocopy:
                    arr = data
                else:
                    arr = data.copy()
            case Matrix():
                arr = data._data
            case tuple():
                arr = np.array(data, dtype=np.float32)
            case _:
                raise ValueError("Invalid matrix format. Must be a 4x4 matrix or a flat list of 16 elements.")
        d = self.dims()
        self._data = cast(np.ndarray[tuple[DIMS, DIMS], np.dtype[np.float32]], arr.reshape((d, d)))

    def __matmul__(self, other: 'Self|at.Vector3|at.Point'):
        if isinstance(other, Matrix):
            return type(self)(np.matmul(self._data, other._data))

        if isinstance(other, at.Vector3):
            v4 = np.array([other.x, other.y, other.z, 0], dtype=np.float32)
            result = self._data @ v4
            return at.vector3(*result[:3])

        if isinstance(other, at.Point): # type: ignore
            v4 = np.array([other.x, other.y, other.z, 1], dtype=np.float32)
            result = self._data @ v4
            return at.point(*result[:3])

        return NotImplemented

    def __mul__(self, scalar: Scalar) -> 'Matrix[DIMS]':
        if not isinstance(scalar, (int, float, np.floating)):
            return NotImplemented
        return type(self)(cast('MatrixSpec', self._data * float(scalar)))

    def __rmul__(self, scalar: Scalar):
        return self.__mul__(scalar)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __eq__(self, other: Self|Any):
        if not isinstance(other, Matrix):
            return False
        o = cast(Matrix[DIMS], other)
        return np.allclose(self._data, o._data, rtol=1e-5, atol=1e-8)

    def __repr__(self):
        return f"Matrix({self._data.tolist()})"

    @classmethod
    def dims(cls) -> DIMS: ...

    def as_array(self) -> np.ndarray[tuple[DIMS, DIMS], np.dtype[np.float32]]:
        '''
        Access the underlying numpy array.
        '''
        return self._data

    def copy(self):
        # Our matrices are immutable, so copying is not needed.
        return self

    @classmethod
    def identity(cls) -> 'Matrix[DIMS]':
        return cls(cast('MatrixSpec', np.identity(cast(DIMS, cls.dims()), dtype=np.float32)))


class Matrix2(Matrix[2]):
    '''A 2x2 matrix.'''

    @classmethod
    def dims(cls) -> Literal[2]:
        return 2

    @classmethod
    def identity(cls) -> 'Matrix2':
        return cls(cast('MatrixSpec', np.identity(cls.dims(), dtype=np.float32)))

class Matrix3(Matrix[3]):
    '''A 3x3 matrix.'''

    @classmethod
    def dims(cls) -> Literal[3]:
        return 3

    @classmethod
    def identity(cls) -> 'Matrix3':
        return cls(cast('MatrixSpec', np.identity(cls.dims(), dtype=np.float32)))

class Matrix4(Matrix[4]):
    '''A 4x4 matrix.'''

    @classmethod
    def dims(cls) -> Literal[4]:
        return 4

    @classmethod
    def identity(cls) -> 'Matrix4':
        return cls(cast('MatrixSpec', np.identity(cls.dims(), dtype=np.float32)))

Matrix2Spec: TypeAlias = (
    Matrix2 | tuple[
        float, float,
        float, float,
    ]
    | tuple[
        tuple[float, float],
        tuple[float, float]
    ]
    | np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float32]]
    | np.ndarray[tuple[Literal[4]], np.dtype[np.float32]]
)
'''
A specification for a 2x2 matrix'
This includes:
    - _Matrix
    - numpy.ndarray[2, 2]
    - tuple[float] * 4
    - tuple[tuple[float] * 2] * 2
'''


Matrix3Spec: TypeAlias = (
    Matrix3 | tuple[
        float, float, float,
        float, float, float,
        float, float, float
    ]
    | tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float]
    ]
    | np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]
    | np.ndarray[tuple[Literal[9]], np.dtype[np.float32]]
)
'''
A specification for a 3x3 matrix'
This includes:
    - _Matrix
    - numpy.ndarray[3, 3]
    - tuple[float] * 9
    - tuple[tuple[float] * 3] * 3
'''


Matrix4Spec: TypeAlias = (
    Matrix[Literal[4]]
    | tuple[
        float, float, float, float,
        float, float, float, float,
        float, float, float, float,
        float, float, float, float,
    ]
    | tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    | np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]
    | np.ndarray[tuple[Literal[16]], np.dtype[np.float32]]
)
'''
A specification for a 4x4 matrix'
This includes:
    - _Matrix
    - numpy.ndarray[4, 4]
    - tuple[float] * 16
    - tuple[tuple[float] * 4] * 4
'''


MatrixSpec: TypeAlias = Matrix2Spec | Matrix3Spec | Matrix4Spec
'''
Any value acceptable as an affine matrix for 3D transformations.
This includes:
    - _Matrix
    - numpy.ndarray[4, 4]
    - tuple[float] * 16
    - tuple[tuple[float] * 4] * 4
'''

IDENTITY2 = Matrix2.identity()
'''
The 2D identity matrix.
This is a 2x2 matrix with ones on the diagonal and zeros elsewhere.
'''
IDENTITY3 = Matrix3.identity()
'''
The identity matrix for 2D transformations.
This is a 3x3 matrix with ones on the diagonal and zeros elsewhere.
'''
IDENTITY4 = Matrix4.identity()
'''
The identity matrix for 3D transformations.
This is a 4x4 matrix with ones on the diagonal and zeros elsewhere.
'''


@overload
def matrix(m: Matrix2Spec) -> Matrix2: ...
@overload
def matrix(m: Matrix3Spec) -> Matrix3: ...
@overload
def matrix(m: Matrix4Spec) -> Matrix4: ...
@overload
def matrix(m: Matrix) -> Matrix: ...
def matrix(m: MatrixSpec) -> Matrix:
    '''
    Verify and convert a Matrix to a standard _Matrix value.

    Parameters:
    ----------
    m : Matrix
        The matrix to convert.

    Returns:
    --------
    _Matrix
    '''
    match m:
        case Matrix():
            return m
        case np.ndarray():
            match m.shape:
                case (2, 2)|(4,):
                    return Matrix2(m)
                case (3, 3)|(9,):
                    return Matrix3(m)
                case (4, 4)|(16,):
                    return Matrix4(m)
        case tuple() if (
            len(m) in (2, 3, 4)
            and all(isinstance(v, tuple) and len(v) == len(m) for v in m)
            and all(isinstance(v, (int, float, np.floating, np.integer)) # type: ignore
                    for a in m
                    for v in cast(tuple[float,...], a))
        ):
            match len(m):
                case 2: return Matrix2(m)
                case 3: return Matrix3(m)
                case 4: return Matrix4(m)
                case _:
                    pass
        case tuple() if (
            len(m) in (4, 9, 16)
            and all(isinstance(v, (int, float, np.floating, np.integer)) for v in m)
        ):
            match len(m):
                case 4: return Matrix2(m)
                case 9: return Matrix3(m)
                case 16: return Matrix4(m)
                case _:
                    pass
        case _:
            pass
    raise ValueError("Invalid matrix format. Must be a 4x4 matrix or a flat list of 16 elements.")
