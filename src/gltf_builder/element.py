'''
Base class for objects which will be referred to by their index
in the glTF. This also holds the name, defaulting it by the index.
'''

from pathlib import Path
from typing import (
    Protocol, Optional, Any, runtime_checkable,
)
from abc import abstractmethod
from collections.abc import Mapping, Iterable

import numpy as np
import pygltflib as gltf

from gltf_builder.holder import Holder
from gltf_builder.quaternion import Quaternion
from gltf_builder.core_types import (
    PrimitiveMode, BufferViewTarget, ElementType,
    Vector3, Vector4, Matrix4, Point, EMPTY_MAP,
    AttributeDataItem, AttributeDataList, AttributeDataSequence,
)
from gltf_builder.compile import Compileable, T, _Scope
from gltf_builder.log import GLTF_LOG


LOG = GLTF_LOG.getChild(Path(__file__).stem)
class Element(Compileable[T], Protocol):
    __index: int = -1 # -1 means not set
    @property
    def index(self) -> int:
        return self.__index
    @index.setter
    def index(self, index: int):
        if self.__index != -1 and self.__index != index:
            raise ValueError(f'Index already set old={self.__index}, new={index}')
        self.__index = index
    
    def __init__(self,
                 name: str='',
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]|None=EMPTY_MAP,
            ):
        super().__init__(extras, extensions)
        self.name = name
        self.extensions = dict(extras) if extras else None
        self.extras = dict(extensions) if extensions else None

    def __index__(self):
        return self.__index
    
    def __hash__(self):
        return id(self)
        
    def __eq__(self, other):
        return self is other
    
    def __repr__(self):
        return f'<{type(self).__name__} {self!s}>'
    
    def __str__(self):
        if self.name == '':
            if self.index == -1:
                return f'{type(self).__name__}-?'
            else:
                return f'{type(self).__name__}-{self.index}'
        else:
            return f'{type(self).__name__}-{self.name}'


class BBuffer(Element[gltf.Buffer], _Scope, Protocol):
    @property
    @abstractmethod
    def blob(self) -> bytes:
        ...
    views: Holder['BBufferView']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def _get_view(self,
                target: BufferViewTarget,
                name: str='',
                normalized: bool=False,
                byteStride: Optional[int]=None,
            ) -> 'BBufferView':
        ...


class BBufferView(Element[gltf.BufferView], Protocol):
    buffer: BBuffer
    target: BufferViewTarget
    byteStride: int
    accessors: list['BAccessor']

    @property
    @abstractmethod
    def blob(self):
        ...

    @abstractmethod
    def _memory(self, offset: int, size: int) -> memoryview:
        ...


@runtime_checkable
class BAccessor(Element[gltf.Accessor], Protocol):
    _view: BBufferView
    data: AttributeDataList
    count: int
    type: ElementType
    byteOffset: int
    componentType: int
    normalized: bool
    max: Optional[list[float]]
    min: Optional[list[float]]
    componentCount: int = 0
    '''The number of components per element.'''
    componentSize: int = 0
    '''The number of bytes per component.'''
    byteStride: int = 0
    '''The total number of bytes per element.'''
    dtype: np.dtype = np.float32
    '''The numpy dtype for the data.'''
    bufferType: str = 'f'
    '''The buffer type char for `memoryview.cast()`.'''

    @abstractmethod
    def _add_data(self, data: AttributeDataSequence) -> None:
        ...
    '''
    Add a Sequence of data to the accessor.
    '''
    
    @abstractmethod
    def _add_data_item(self, data: AttributeDataItem) -> None:
        ...
        
class BPrimitive(Compileable[gltf.Primitive], Protocol):
    '''
    Base class for primitives
    '''
    mode: PrimitiveMode
    points: list[Point]
    attribs: dict[str, list[tuple[int|float,...]]]
    indicies: list[int]
    mesh: Optional['BMesh']
    

@runtime_checkable
class BMesh(Element[gltf.Mesh], _Scope, Protocol):
    primitives: list[BPrimitive]
    weights: list[float]

    @property
    @abstractmethod
    def detached(self):
        '''
        A detached mesh is not added to the builder, but is returned
        to be used as the root of an instancable object, or to be added
        to multiple nodes and thus to the builder later.
        '''
        ...


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
                      extras: Mapping[str, Any]|None=EMPTY_MAP,
                      extensions: Mapping[str, Any]|None=EMPTY_MAP,
                      **attribs: Iterable[tuple[int|float,...]]
                    ) -> BPrimitive:
        ...

    __views: Holder[BBufferView] 

    def _get_view(self, target: BufferViewTarget, for_object: Compileable) -> BBufferView:
        ...
    

@runtime_checkable
class BNode(Element[gltf.Node], _Scope, Protocol):
    mesh: BMesh
    root: bool
    translation: Optional[Vector3]
    rotation: Optional[Quaternion]
    scale: Optional[Vector3]
    matrix: Optional[Matrix4]

    @property
    @abstractmethod
    def detached(self):
        '''
        A detached node is not added to the builder, but is returned
        to be used as the root of an instancable object.
        '''
        ...

    @abstractmethod
    def create_mesh(self,
                name: str='',
                primitives: Iterable['BPrimitive']=(),
                weights: Iterable[float]|None=(),
                extras: Mapping[str, Any]|None=EMPTY_MAP,
                extensions: Mapping[str, Any]|None=EMPTY_MAP,
                detached: bool=False,
            ) -> 'BMesh':
        '''
        Create a `BMesh` for this `BNode`, or if `detached` is `True`,
        just create a `BMesh` and return it for later use.
        '''
        ...