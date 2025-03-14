'''
Base class for objects which will be referred to by their index
in the glTF. This also holds the name, defaulting it by the index.
'''

from typing import TypeAlias, Protocol, TypeVar, Generic, Optional, Any, runtime_checkable
from abc import abstractmethod
from enum import IntEnum, StrEnum
from collections.abc import Mapping

import numpy as np
import pygltflib as gltf

from gltf_builder.holder import Holder
    

T = TypeVar('T')


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
Quaternion: TypeAlias = Vector4

@runtime_checkable
class BuilderProtocol(Protocol):
    asset: gltf.Asset
    points: list[Point]
    meshes: Holder['BMeshProtocol']
    buffers: Holder['BBufferProtocol']
    views: Holder['BBufferViewProtocol']

EMPTY_SET: Mapping[str, Any] = frozenset()

class Compileable(Generic[T], Protocol):
    __compiled: T|None = None
    extensions: dict[str, Any]
    extras: dict[str, Any]
    def __init__(self,
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
                ):
        self.extras = extras
        self.extensions = extensions
    
    def compile(self, builder: BuilderProtocol) -> T:
        if self.__compiled is None:
            self.__compiled = self.do_compile(builder)
        return self.__compiled
            
    @abstractmethod
    def do_compile(self, builder: BuilderProtocol) -> T:
        ...
        

class Element(Compileable[T], Protocol):
    __index: int = -1 # -1 means not set
    name: str = ''    # '' means not set
    
    def __init__(self,
                 name: str='',
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
            ):
        super().__init__(extras, extensions)
        self.name = name
        self.extensions = dict(extras)
        self.extras = dict(extensions)
    
    @property
    def index(self):
        if self.__index == -1:
            raise ValueError(f'The index for {self} has not been set.')
        return self.__index
    
    @index.setter
    def index(self, value: int):
        if self.__index != -1 and self.__index != value:
            raise ValueError(f'The index for {self} has already been set.')
        self.__index = value

    def __hash__(self):
        return id(self)
        
    def __eq__(self, other):
        return self is other
    
    def __str__(self):
        if self.name == '':
            if self.__index == -1:
                return f'{type(self).__name__}-?'
            else:
                return f'{type(self).__name__}-{self.index}'
        else:
            return f'{type(self).__name__}-{self.name}'


class BBufferProtocol(Element[gltf.Buffer], Protocol):
    data: bytes
    views: Holder['BBufferViewProtocol']


class BBufferViewProtocol(Element[gltf.BufferView], Protocol):
    buffer: BBufferProtocol
    target: BufferViewTarget
    byteStride: int
    accessors: Holder['BAccessorProtocol']
    data: bytes


class BAccessorProtocol(Element[gltf.Accessor], Protocol):
    view: BBufferViewProtocol
    data: np.ndarray[tuple[int, ...], Any]
    count: int
    type: ElementType
    byteOffset: int
    componentType: int
    normalized: bool
    max: Optional[list[float]]
    min: Optional[list[float]]
    
    
class BPrimitiveProtocol(Compileable[gltf.Primitive], Protocol):
    '''
    Base class for primitives
    '''
    mode: PrimitiveMode
    points: list[Point]
    indicies: list[int]
    
    
class BMeshProtocol(Element[gltf.Mesh], Protocol):
    primitives: list[BPrimitiveProtocol]
    weights: list[float]
    
class BNodeProtocol(Element[gltf.Node]):
    mesh: BMeshProtocol
    root: bool
    translation: Optional[Vector3]
    rotation: Optional[Quaternion]
    scale: Optional[Vector3]
    matrix: Optional[Matrix4]