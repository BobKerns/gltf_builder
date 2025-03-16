'''
Base class for objects which will be referred to by their index
in the glTF. This also holds the name, defaulting it by the index.
'''

from typing import (
    TypeAlias, Protocol, TypeVar, Generic, Optional, Any, 
    runtime_checkable,
)
from abc import abstractmethod
from enum import IntEnum, StrEnum
from collections.abc import Mapping, Iterable

import numpy as np
import pygltflib as gltf

from gltf_builder.holder import Holder
    

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


EMPTY_SET: Mapping[str, Any] = frozenset()


class BNodeContainerProtocol(Protocol):
    children: Holder['BNode']
    @property
    def nodes(self):
        return self.children
    @nodes.setter
    def nodes(self, nodes: Holder['BNode']):
        self.children = nodes
    
    @abstractmethod
    def add_node(self,
                name: str='',
                children: Iterable['BNode']=(),
                mesh: Optional['BMesh']=None,
                translation: Optional[Vector3]=None,
                rotation: Optional[Quaternion]=None,
                scale: Optional[Vector3]=None,
                matrix: Optional[Matrix4]=None,
                extras: Mapping[str, Any]=EMPTY_SET,
                extensions: Mapping[str, Any]=EMPTY_SET,
                detached: bool=False,
                **attrs: tuple[float|int,...]
                ) -> 'BNode':
        ...
    
    @abstractmethod
    def instantiate(self, node: 'BNode', /,
                    name: str='',
                    translation: Optional[Vector3]=None,
                    rotation: Optional[Quaternion]=None,
                    scale: Optional[Vector3]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Mapping[str, Any]=EMPTY_SET,
                    extensions: Mapping[str, Any]=EMPTY_SET,
                ) -> 'BNode':
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> 'BNode':
        ...

    @abstractmethod
    def __setitem__(self, name: str, node: 'BNode'):
        ...

    @abstractmethod
    def __contains__(self, name: str) -> bool:
        ...

    @abstractmethod
    def __iter__(self):
        ...

@runtime_checkable
class BuilderProtocol(BNodeContainerProtocol, Protocol):
    asset: gltf.Asset
    '''
    The asset information for the glTF file.
    '''
    meshes: Holder['BMesh']
    '''
    The meshes in the glTF file.
    '''
    buffers: Holder['BBuffer']
    '''
    The buffers in the glTF file.'''
    views: Holder['BBufferView']
    '''
    The buffer views in the glTF file.
    '''
    accessors: Holder['BAccessor']
    '''
    The accessors in the glTF file.
    '''
    extras: dict[str, Any]
    '''
    The extras for the glTF file.
    '''
    extensions: dict[str, Any]
    '''
    The extensions for the glTF file.
    '''
    index_size: int = 32
    '''
    Number of bits to use for indices. Default is 32.
    
    This is used to determine the component type of the indices.
    8 bits will use UNSIGNED_BYTE, 16 bits will use UNSIGNED_SHORT,
    and 32 bits will use UNSIGNED_INT.

    A value of 0 will use the smallest possible size for a particular
    mesh.

    A value of -1 will disaable indexed goemetry.

    This is only used when creating the indices buffer view.

    '''
    attr_type_map: dict[str, tuple[ElementType, ComponentType]]
    
    @abstractmethod
    def add_mesh(self,
                 name: str='',
                 primitives: Iterable['BPrimitive']=(),
            ) -> 'BMesh':
        ...
    
    @abstractmethod
    def add_buffer(self,
                   name: str='') -> 'BBuffer':
        ...
    
    @abstractmethod
    def add_view(self,
                 name: str='',
                 buffer: Optional['BBuffer']=None,
                 data: Optional[bytes]=None,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
            ) -> 'BBufferView':
        ...
        
    @abstractmethod
    def get_view(self, name: str,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
       ) -> 'BBufferView':
        ...
    
    @abstractmethod
    def build(self) -> gltf.GLTF2:
        ...

    @abstractmethod
    def define_attrib(self, name: str, type: ElementType, componentType: ComponentType):
        ...

    @abstractmethod
    def get_attrib_info(self, name: str) -> tuple[ElementType, ComponentType]:
        ...

    @abstractmethod
    def get_index_size(self, max_value: int) -> int:
        ...


T = TypeVar('T')
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


class BBuffer(Element[gltf.Buffer], Protocol):
    @property
    @abstractmethod
    def blob(self) -> bytes:
        ...
    views: Holder['BBufferView']
    @abstractmethod
    def extend(self, data: bytes|np.typing.NDArray) -> None:
        ...
    @abstractmethod
    def __len__(self) -> int:
        ...


class BBufferView(Element[gltf.BufferView], Protocol):
    buffer: BBuffer
    target: BufferViewTarget
    byteStride: int
    accessors: Holder['BAccessor']

    @property
    @abstractmethod
    def blob(self):
        ...

    @property
    @abstractmethod
    def offset(self):
        ...

    @abstractmethod
    def add_accessor(self,
                    type: ElementType,
                    componentType: ComponentType,
                    data: np.ndarray[tuple[int, ...], Any]|Iterable[Any],
                    normalized: bool=False,
                    min: Optional[list[float]]=None,
                    max: Optional[list[float]]=None,
                    extras: Mapping[str, Any]=EMPTY_SET,
                    extensions: Mapping[str, Any]=EMPTY_SET,
            ) -> gltf.Accessor:
        ...

    @abstractmethod
    def extend(self, data: bytes|np.typing.NDArray) -> None:
        ...

    @abstractmethod
    def __len__(self):
        ...

class BAccessor(Element[gltf.Accessor], Protocol):
    view: BBufferView
    data: np.ndarray[tuple[int, ...], Any]
    count: int
    type: ElementType
    byteOffset: int
    componentType: int
    normalized: bool
    max: Optional[list[float]]
    min: Optional[list[float]]
    
    
class BPrimitive(Compileable[gltf.Primitive], Protocol):
    '''
    Base class for primitives
    '''
    mode: PrimitiveMode
    points: list[Point]
    indicies: list[int]
    attribs: dict[str, list[tuple[int|float,...]]]
    
    
class BMesh(Element[gltf.Mesh], Protocol):
    primitives: list[BPrimitive]
    weights: list[float]

    @abstractmethod
    def add_primitive(self, mode: PrimitiveMode,
                      *points: Point,
                      NORMAL: Optional[Iterable[Vector3]]=None,
                      TANGENT: Optional[Iterable[Vector4]]=None,
                      TEXCOORD_0: Optional[Iterable[Vector3]]=None,
                      TEXCOORD_1: Optional[Iterable[Vector3]]=None,
                      COLOR_0: Optional[Iterable[Vector4]]=None,
                      JOINTS_0: Optional[Iterable[Vector4]]=None,
                      WEIGHTS_0: Optional[Iterable[Vector4]]=None,
                      extras: Mapping[str, Any]=EMPTY_SET,
                      extensions: Mapping[str, Any]=EMPTY_SET,
                      **attribs: Iterable[tuple[int|float,...]]
                    ) -> BPrimitive:
        ...
    

class BNode(Element[gltf.Node]):
    mesh: BMesh
    root: bool
    translation: Optional[Vector3]
    rotation: Optional[Quaternion]
    scale: Optional[Vector3]
    matrix: Optional[Matrix4]
