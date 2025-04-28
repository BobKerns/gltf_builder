'''
Simple common types for the gltf_builder module.
'''

from enum import IntEnum, StrEnum
from typing import TypeAlias, Literal

import pygltflib as gltf
import numpy as np


IntScalar: TypeAlias = int|np.int8|np.int16|np.int32|np.uint8|np.uint16
Scalar: TypeAlias = float|np.float32|IntScalar
'''
A scalar value: int, float, or numpy equivalent.
'''


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
    EXTENSIONS = 'extensions'
    '''
    Collect the set of used extensions for the glTF file.
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


ComponentSize: TypeAlias = Literal[1, 2, 4]
'''
The size in bytes of data components in the glTF file.
The values are:
- 1: 1 byte integer
- 2: 2 bytes integer
- 4: 4 bytes float32
'''

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

class IndexSize(IntEnum):
    '''
    The size of the index values in the glTF file.
    '''
    NONE = -1
    AUTO = 0
    UNSIGNED_BYTE = gltf.UNSIGNED_BYTE
    UNSIGNED_SHORT = gltf.UNSIGNED_SHORT
    UNSIGNED_INT = gltf.UNSIGNED_INT

ElementSize: TypeAlias = Literal[1, 2, 3, 4, 9, 16]
'''
The number of components in an element.
The values are:
- 1: 1 component (e.g. scalar)
- 2: 2 components (e.g. vec2)
- 3: 3 components (e.g. vec3)
- 4: 4 components or 2x2 matrix
- 9: 3x3 matrix
- 16: 4x4 matrix
'''


BufferType: TypeAlias = Literal['b', 'B', 'h', 'H', 'l', 'L', 'f']
'''
Type code for casting a memoryview of a buffer.
'''

NPAttrTypes: TypeAlias = np.uint8|np.uint16|np.float32
'''
The numpy types used in the glTF file for the standard attributes
'''

NPTypes: TypeAlias = np.int8|np.int16|np.uint32|NPAttrTypes
'''
The numpy types used in the glTF file!!
'''

NPAttrDType: TypeAlias = np.dtype[np.float32]|np.dtype[np.uint8]|np.dtype[np.uint16]
'''
The numpy dtypes used in the glTF file for the standard attributes
'''


NPDType: TypeAlias = np.dtype[np.int8]|np.dtype[np.int16]|NPAttrDType
'''
The numpy dtypes used in the glTF file
'''

class ScopeName(StrEnum):
    '''
    Enum for the scope of a policy.
    '''

    ASSET = 'asset'''
    '''
    The policy applies to the asset.
    '''
    PRIMITIVE = 'primitive'
    '''
    The policy applies to the primitives.
    '''
    MESH = 'mesh'
    '''
    The policy applies to the mesh.
    '''
    NODE = 'node'
    '''
    The policy applies to the node.
    '''
    BUFFER_VIEW = 'view'
    '''
    The policy applies to the buffer view.
    '''
    ACCESSOR = 'accessor'
    '''
    The policy applies to accessors.
    '''
    ACCESSOR_INDEX = 'accessor_index'
    '''
    The policy applies to the index of the primitive.
    '''
    BUFFER = 'buffer'
    '''
    The policy applies to the buffer.
    '''
    BUILDER = 'builder'
    '''
    The policy applies to the builder.
    '''
    IMAGE = 'image'
    '''
    The policy applies to the image.
    '''
    MATERIAL = 'material'
    '''
    The policy applies to the material.
    '''
    TEXTURE = 'texture'
    '''
    The policy applies to the texture.
    '''
    CAMERA = 'camera'
    '''
    The policy applies to the camera.
    '''
    SAMPLER = 'sampler'
    '''
    The policy applies to the sampler.
    '''
    SKIN = 'skin'
    '''
    The policy applies to the skin.
    '''
    SCENE = 'scene'
    '''
    The policy applies to the scene.
    '''
    EXTENSION = 'extension'
    '''
    The policy applies to the extension.
    '''
    ANIMATION = 'animation'
    '''
    The policy applies to the animation.
    '''
    ANIMATION_CHANNEL = 'animation_channel'
    '''
    The policy applies to the animation channel.
    '''
    ANIMATION_SAMPLER = 'animation_sampler'
    '''
    The policy applies to the animation sampler.
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


JsonObject: TypeAlias = dict[str,'JsonData']
'''
A JSON-compatible object type.
'''
JsonArray: TypeAlias = list['JsonData']
'''
A JSON-compatible array type.
'''
JsonAtomic: TypeAlias = str|int|float|bool|None
'''
A JSON-compatible atomic type.
'''
JsonData: TypeAlias = JsonObject|JsonArray|JsonAtomic
'''
A JSON-compatible data type.
'''

ExtrasData: TypeAlias = dict[str, JsonData]
'''
A dictionary of extra data to be stored with the object.
'''

ExtensionData: TypeAlias = JsonData

ExtensionsData: TypeAlias = dict[str, ExtensionData]
'''
A dictionary of extensions to be stored with the object.
'''

NamePolicy: TypeAlias = dict[ScopeName, NameMode]
'''
A policy for how to handle or generate names for objects.
The keys are the scope names, and the values are the name modes.
'''


class  ImageType(StrEnum):
    '''
    Enum for the supported image types.
    '''
    PNG = 'image/png'
    '''
    PNG image type.
    '''
    JPEG = 'image/jpeg'
    '''
    JPEG image type.
    '''

class MagFilter(IntEnum):
    '''
    Enum for the supported magnification filters.
    '''
    NEAREST = gltf.NEAREST
    '''
    Nearest neighbor filter.
    '''
    LINEAR = gltf.LINEAR
    '''
    Linear filter.
    '''

class MinFilter(IntEnum):
    '''
    Enum for the supported minification filters.
    '''
    NEAREST = gltf.NEAREST
    '''
    Nearest neighbor filter.
    '''
    LINEAR = gltf.LINEAR
    '''
    Linear filter.
    '''
    NEAREST_MIPMAP_NEAREST = gltf.NEAREST_MIPMAP_NEAREST
    '''
    Nearest neighbor filter with mipmaps.
    '''
    LINEAR_MIPMAP_NEAREST = gltf.LINEAR_MIPMAP_NEAREST
    '''
    Linear filter with mipmaps.
    '''
    NEAREST_MIPMAP_LINEAR = gltf.NEAREST_MIPMAP_LINEAR
    '''
    Nearest neighbor filter with linear mipmaps.
    '''
    LINEAR_MIPMAP_LINEAR = gltf.LINEAR_MIPMAP_LINEAR
    '''
    Linear filter with linear mipmaps.
    '''

class WrapMode(IntEnum):
    '''
    Enum for the supported wrap modes.
    '''
    CLAMP_TO_EDGE = gltf.CLAMP_TO_EDGE
    '''
    Clamp to edge wrap mode.
    '''
    MIRRORED_REPEAT = gltf.MIRRORED_REPEAT
    '''
    Mirrored repeat wrap mode.
    '''
    REPEAT = gltf.REPEAT
    '''
    Repeat wrap mode.
    '''


class AlphaMode(StrEnum):
    '''
    Enum for the supported alpha modes.
    '''
    OPAQUE = 'OPAQUE'
    '''
    Opaque alpha mode.
    '''
    MASK = 'MASK'
    '''
    Masked alpha mode.
    '''
    BLEND = 'BLEND'
    '''
    Blended alpha mode.
    '''

class AnimationInterpolation(StrEnum):
    '''
    Enum for the supported animation interpolation modes.
    '''
    LINEAR = 'LINEAR'
    '''
    Linear interpolation mode.
    '''
    STEP = 'STEP'
    '''
    Step interpolation mode.
    '''
    CUBICSPLINE = 'CUBICSPLINE'
    '''
    Cubic spline interpolation mode.
    '''


class AnimationTargetPath(StrEnum):
    '''
    Enum for the supported animation target paths.
    '''
    TRANSLATION = 'translation'
    '''
    Translation target path.
    '''
    ROTATION = 'rotation'
    '''
    Rotation target path.
    '''
    SCALE = 'scale'
    '''
    Scale target path.
    '''
    WEIGHTS = 'weights'
    '''
    Weights target path.
    '''


class AnimationSamplerInterpolation(StrEnum):
    '''
    Enum for the supported animation sampler interpolation modes.
    '''
    LINEAR = 'LINEAR'
    '''
    Linear interpolation mode.
    '''
    STEP = 'STEP'
    '''
    Step interpolation mode.
    '''
    CUBICSPLINE = 'CUBICSPLINE'
    '''
    Cubic spline interpolation mode.
    '''

class CameraType(StrEnum):
    '''
    Enum for the supported camera types.
    '''
    PERSPECTIVE = 'perspective'
    '''
    Perspective camera type.
    '''
    ORTHOGRAPHIC = 'orthographic'
    '''
    Orthographic camera type.
    '''
