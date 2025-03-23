'''
Simple common types for the gltf_builder module.
'''

from enum import IntEnum, StrEnum
from collections.abc import Sequence, Mapping
from typing import Any, TypeAlias
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

Vector2: TypeAlias = tuple[float, float]
Vector3: TypeAlias = tuple[float, float, float]
Vector4: TypeAlias = tuple[float, float, float, float]
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
Point: TypeAlias = Vector3
Tangent: TypeAlias = Vector4
Normal: TypeAlias = Vector3
Scale: TypeAlias = Vector3|Scalar

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
