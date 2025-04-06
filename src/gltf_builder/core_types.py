'''
Simple common types for the gltf_builder module.
'''

from enum import IntEnum, StrEnum
from typing import TypeAlias, Literal, Any
from collections.abc import Mapping
from types import MappingProxyType

import pygltflib as gltf
import numpy as np


_IntScalar: TypeAlias = int|np.int8|np.int16|np.int32|np.uint8|np.uint16|np.uint32
Scalar: TypeAlias = float|np.float32|_IntScalar
'''A scalar value: int, float, or numpy equivalent.'''


float01: TypeAlias = float|Literal[0,1]|np.float32
'''
A float value between 0 and 1, or the literals 0 or 1.
'''


ByteSize: TypeAlias = Literal[1, 2, 4]
'''
The size of the data in bytes for the glTF file. This is used to determine
the size of the data in the accessors and views for the glTF file.
The values are:
- 1: 1 byte integer
- 2: 2 bytes integer
- 4: 4 bytes float32
'''
ByteSizeAuto: TypeAlias = Literal[0, 1, 2, 4]
'''
The size of the data in bytes for the glTF file. This is used to determine
the size of the data in the accessors and views for the glTF file.
The values are:
- 0: Auto-detect the size of the data
- 1: 1 byte integer
- 2: 2 bytes integer
- 4: 4 bytes integer
'''


EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})


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
    '''
    The glTF primitive modes.
    '''
    POINTS = gltf.POINTS
    LINES = gltf.LINES
    LINE_LOOP = gltf.LINE_LOOP
    LINE_STRIP = gltf.LINE_STRIP
    TRIANGLES = gltf.TRIANGLES
    TRIANGLE_STRIP = gltf.TRIANGLE_STRIP
    TRIANGLE_FAN = gltf.TRIANGLE_FAN
    
class BufferViewTarget(IntEnum):
    '''
    The glTF target for a buffer view.
    '''
    ARRAY_BUFFER = gltf.ARRAY_BUFFER
    ELEMENT_ARRAY_BUFFER = gltf.ELEMENT_ARRAY_BUFFER
    
class ElementType(StrEnum):
    '''
    glTF element types—the composite group of values that live in the accessors.
    '''
    SCALAR = "SCALAR"
    VEC2 = "VEC2"
    VEC3 = "VEC3"
    VEC4 = "VEC4"
    MAT2 = "MAT2"
    MAT3 = "MAT3"
    MAT4 = "MAT4"
    
class ComponentType(IntEnum):
    '''
    glTF component types—the size of the values that live in the elements.
    '''
    BYTE = gltf.BYTE
    UNSIGNED_BYTE = gltf.UNSIGNED_BYTE
    SHORT = gltf.SHORT
    UNSIGNED_SHORT = gltf.UNSIGNED_SHORT
    UNSIGNED_INT = gltf.UNSIGNED_INT
    FLOAT = gltf.FLOAT


BufferType: TypeAlias = Literal['b', 'B', 'h', 'H', 'l', 'L', 'f']
'''
Type code for casting a memoryview of a buffer.
'''


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
