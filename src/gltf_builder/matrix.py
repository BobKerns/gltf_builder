'''
4x4 matrix class and operations.
'''

from typing import TypeAlias, Literal, Generic, TypeVar

import numpy as np

import gltf_builder.attribute_types as at
from gltf_builder.core_types import Scalar


DIMS = TypeVar('DIMS', bound=Literal[2, 3, 4])
'''Number of dimensions in the matrix.'''

class Matrix(Generic[DIMS]):
    def __init__(self, data: tuple|np.ndarray, nocopy:bool = False):
        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                arr = data.astype(np.float32)
            elif nocopy:
                arr = data
            else:
                arr = data.copy()
        else:
           arr = np.array(data, dtype=np.float32)
        if arr.size != 16:
            raise ValueError("Matrix must have 16 elements.")
        self._data = arr.reshape((4, 4))

    def __matmul__(self, other: 'Matrix|at.Vector3|at.Point'):
        if isinstance(other, Matrix):
            return Matrix(np.matmul(self._data, other._data))

        if isinstance(other, at.Vector3):
            v4 = np.array([other.x, other.y, other.z, 0], dtype=np.float32)
            result = self._data @ v4
            return at.vector3(*result[:3])

        if isinstance(other, at.Point):
            v4 = np.array([other.x, other.y, other.z, 1], dtype=np.float32)
            result = self._data @ v4
            return at.point(*result[:3])

        return NotImplemented

    def __mul__(self, scalar: Scalar) -> 'Matrix':
        if not isinstance(scalar, (int, float, np.float32)):
            return NotImplemented
        return matrix(self._data * scalar)

    def __rmul__(self, scalar: Scalar):
        return self.__mul__(scalar)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return np.allclose(self._data, other._data, rtol=1e-5, atol=1e-8)

    def __repr__(self):
        return f"Matrix({self._data.tolist()})"

    def as_array(self) -> np.ndarray[tuple[int, int], np.float32]:
        '''
        Acess the underlying numpy array.
        '''
        return self._data

    def copy(self):
        # Our matrices are immutable, so copying is not needed.
        return self

    @classmethod
    def identity(cls):
        return cls(np.identity(4, dtype=np.float32))


class Matrix2(Matrix[2]):
    '''A 2x2 matrix.'''
    pass

class Matrix3(Matrix[3]):
    '''A 3x3 matrix.'''
    pass


class Matrix4(Matrix[4]):
    '''A 4x4 matrix.'''
    pass


Matrix2Spec: TypeAlias = Matrix2 | tuple[
        float, float,
        float, float,
    ] | tuple[
        tuple[float, float],
        tuple[float, float]
    ]
'''
A specification for a 2x2 matrix'
This includes:
    - _Matrix
    - numpy.ndarray[2, 2]
    - tuple[float] * 4
    - tuple[tuple[float] * 2] * 2
'''


Matrix3Spec: TypeAlias = Matrix3 | tuple[
        float, float, float,
        float, float, float,
        float, float, float
    ] | tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float]
    ]
'''
A specification for a 3x3 matrix'
This includes:
    - _Matrix
    - numpy.ndarray[3, 3]
    - tuple[float] * 9
    - tuple[tuple[float] * 3] * 3
'''


Matrix4Spec: TypeAlias = tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]
'''
A specification for a 4x4 matrix'
This includes:
    - _Matrix
    - numpy.ndarray[4, 4]
    - tuple[float] * 16
    - tuple[tuple[float] * 4] * 4
'''


MatrixSpec: TypeAlias = Matrix | Matrix2 | Matrix3 | Matrix4
'''
Any value acceptable as an affine matrix for 3D transformations.
This includes:
    - _Matrix
    - numpy.ndarray[4, 4]
    - tuple[float] * 16
    - tuple[tuple[float] * 4] * 4
'''


IDENTITY: MatrixSpec = Matrix.identity()
'''
The identity matrix for 3D transformations.
This is a 4x4 matrix with ones on the diagonal and zeros elsewhere.
'''

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
            return Matrix(m)
        case tuple() if (
            len(m) == 4
            and all(isinstance(v, tuple) and len(v) == 4 for v in m)
            and all(isinstance(v, (int, float, np.float32))
                    for a in m
                    for v in a)
        ):
            return Matrix(m)
        case tuple() if (
            len(m) == 16
            and all(isinstance(v, (int, float, np.float32)) for v in m)
        ):
            return Matrix(m)
        case _:
            raise ValueError("Invalid matrix format. Must be a 4x4 matrix or a flat list of 16 elements.")
