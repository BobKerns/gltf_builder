'''
Simple common types for the gltf_builder module.
'''

from enum import IntEnum, StrEnum
from collections.abc import Sequence, Mapping
from typing import Any, NamedTuple, Optional, TypeAlias, Literal, overload
from types import MappingProxyType

import pygltflib as gltf
import numpy as np

class Phase(StrEnum):
    '''
    Enum for the phases of the compile process. Not all are implemented.
    '''
    PRIMITIVES = 'primitives'
    '''
    Process the data for the primitives for the glTF file.
    '''
    COLLECT = 'collect'
    '''
    Create the accessors and views for the glTF file, and collect all 
    subordinate objects.
    '''
    ENUMERATE = 'enumerate'
    '''
    Assign index values to each object
    '''
    VERTICES = 'vertices'
    '''
    Optimize the vertices for the glTF file.
    '''
    SIZES = 'sizes'
    '''
    Calculate sizes for the accessors and views for the glTF file.
    '''
    OFFSETS = 'offsets'
    '''
    Calculate offsets for the accessors and views for the glTF file.
    '''
    BUFFERS = 'buffers'
    '''
    Initialize buffers to receive data
    '''
    VIEWS = 'views'
    '''
    Initialize buffer views to receive data
    '''
    BUILD = 'build'
    '''
    Construct the binary data for the glTF file.
    '''
    

class PrimitiveMode(IntEnum):
    POINTS = gltf.POINTS
    LINES = gltf.LINES
    LINE_LOOP = gltf.LINE_LOOP
    LINE_STRIP = gltf.LINE_STRIP
    TRIANGLES = gltf.TRIANGLES
    TRIANGLE_STRIP = gltf.TRIANGLE_STRIP
    TRIANGLE_FAN = gltf.TRIANGLE_FAN
    
class BufferViewTarget(IntEnum):
    ARRAY_BUFFER = gltf.ARRAY_BUFFER
    ELEMENT_ARRAY_BUFFER = gltf.ELEMENT_ARRAY_BUFFER
    
class ElementType(StrEnum):
    SCALAR = "SCALAR"
    VEC2 = "VEC2"
    VEC3 = "VEC3"
    VEC4 = "VEC4"
    MAT2 = "MAT2"
    MAT3 = "MAT3"
    MAT4 = "MAT4"
    
class ComponentType(IntEnum):
    BYTE = gltf.BYTE
    UNSIGNED_BYTE = gltf.UNSIGNED_BYTE
    SHORT = gltf.SHORT
    UNSIGNED_SHORT = gltf.UNSIGNED_SHORT
    UNSIGNED_INT = gltf.UNSIGNED_INT
    FLOAT = gltf.FLOAT


BufferType: TypeAlias = Literal['b', 'B', 'h', 'H', 'l', 'L', 'f']



class _Vector2(NamedTuple):
    x: float
    y: float


class _Vector3(NamedTuple):
    x: float
    y: float
    z: float


class _Vector4(NamedTuple):
    x: float
    y: float
    z: float
    w: float


class _Point(NamedTuple):
    x: float
    y: float
    z: float


class _Scale(NamedTuple):
    x: float
    y: float
    z: float

NP2Vector: TypeAlias = np.ndarray[tuple[Literal[3]], float]
NP3Vector: TypeAlias = np.ndarray[tuple[Literal[3]], float]
NP4Vector: TypeAlias = np.ndarray[tuple[Literal[4]], float]


Vector2: TypeAlias = tuple[float, float]|_Vector2|NP2Vector
Vector3: TypeAlias = tuple[float, float, float]|_Vector3|NP3Vector
Vector4: TypeAlias = tuple[float, float, float, float]|_Vector4|NP4Vector

Matrix2: TypeAlias = tuple[
    float, float,
    float, float,
]
Matrix3: TypeAlias = tuple[
    float, float, float,
    float, float, float,
    float, float, float
]
Matrix4: TypeAlias = tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]

Scalar: TypeAlias = float
Point: TypeAlias = Vector3|_Point
Tangent: TypeAlias = Vector4
Normal: TypeAlias = Vector3
Scale: TypeAlias = Vector3|Scalar|_Scale

@overload
def point(p: Point, /) -> _Point: ...
@overload
def point(x: float, y: float, z: float) -> _Point: ...
def point(x: Optional[float|Point]=None,
        y: Optional[float]=None,
        z: Optional[float]=None) -> _Point:
    match x:
        case None:
            return _Point(0.0, 0.0, 0.0)
        case _Point():
            return x        
        case x, y, z:
            return _Point(float(x), float(y), float(z))
        case np.ndarray() if x.shape == (3,):
            return _Point(float(x[0]), float(x[1]), float(x[2]))
        case float():
            return _Point(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid point')
        

@overload
def vector2(v: Vector2, /) -> _Vector2: ...
@overload
def vector2(x: float, y: float) -> _Vector2: ...
def vector2(x: Optional[float|Point]=None,
            y: Optional[float]=None) -> _Vector2:
    match x:
        case None:
            return _Vector2(0.0, 0.0)
        case _Vector2():
            return x        
        case x, y:
            return _Vector2(float(x), float(y))
        case np.ndarray() if x.shape == (2,):
            return _Vector2(float(x[0]), float(x[1]))
        case float():
            return _Vector2(float(x), float(y))
        case _:
            raise ValueError('Invalid vector3')     


@overload
def vector3(v: Vector3, /) -> _Vector3: ...
@overload
def vector3(x: float, y: float, z: float) -> _Vector3: ...
def vector3(x: Optional[float|Point]=None,
            y: Optional[float]=None,
            z: Optional[float]=None) -> _Vector3:
    match x:
        case None:
            return _Vector3(0.0, 0.0, 0.0)
        case _Vector3():
            return x        
        case x, y, z:
            return _Vector3(float(x), float(y), float(z))
        case np.ndarray() if x.shape == (3,):
            return _Vector3(float(x[0]), float(x[1]), float(x[2]))
        case float():
            return _Vector3(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid vector3')   


@overload
def vector4(v: Vector4, /) -> _Vector4: ...
@overload
def vector4(x: float, y: float, z: float) -> _Vector4: ...
def vector4(x: Optional[float|Point]=None,
        y: Optional[float]=None,
        z: Optional[float]=None,
        w: Optional[float]=None
    ) -> _Vector4:
    match x:
        case None:
            return _Vector4(0.0, 0.0, 0.0, 0.0)
        case _Vector4():
            return x        
        case x, y, z, w:
            return _Vector4(float(x), float(y), float(z), float(w))
        case np.ndarray() if x.shape == (4,):
            return _Vector4(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case float():
            return _Vector4(float(x), float(y), float(z), float(w))
        case _:
            raise ValueError('Invalid vector3')        


@overload
def scale(p: Scale, /) -> _Scale: ...
@overload
def scale(x: float, y: float, z: float) -> _Scale: ...
def scale(x: Optional[float|Point]=None,
        y: Optional[float]=None,
        z: Optional[float]=None) -> _Point:
    match x:
        case None:
            return _Scale(1.0, 1.0, 1.0)
        case float() if y is None and z is None:
            return _Scale(x, x, x)
        case _Scale():
            return x        
        case x, y, z:
            return _Scale(float(x), float(y), float(z))
        case np.ndarray() if x.shape == (3,):
            return _Scale(float(x[0]), float(x[1]), float(x[2]))
        case _ if y is not None and z is not None:
            return _Scale(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid scale')
        


EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})


class NameMode(StrEnum):
    '''
    Enum for how to handle or generate names for objects.
    '''
    
    AUTO = 'auto'
    '''
    Automatically generate names for objects which do not have one.
    '''
    MANUAL = 'manual'
    '''
    Use the name provided.
    '''
    UNIQUE = 'unique'
    '''
    Ensure the name is unique.
    '''
    NONE = 'none'
    '''
    Do not use names.
    '''


AttributeDataItem: TypeAlias = int|float|tuple[int, ...]|tuple[float, ...]|np.ndarray[tuple[int, ...], Any] 

AttributeDataList: TypeAlias = (
    list[int]
    |list[float]
    |list[tuple[int, ...]]
    |list[tuple[float, ...]]
    |list[np.ndarray[tuple[int, ...], Any]]
)
'''
List of attribute data in various formats. Lists of:
- integers
- floats
- tuples of integers
- tuples of floats
- numpy arrays of integers
- numpy arrays of floats
'''

AttributeDataSequence: TypeAlias = (
    Sequence[int]
    |Sequence[float]
    |Sequence[tuple[int, ...]]
    |Sequence[tuple[float, ...]]
    |Sequence[np.ndarray[tuple[int, ...], Any]]
)
'''
Sequence of attribute data in various formats. Lists of:
- integers
- floats
- tuples of integers
- tuples of floats
- numpy arrays of integers
- numpy arrays of floats
'''
