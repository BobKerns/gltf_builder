'''
Base class for objects which will be referred to by their index
in the glTF. This also holds the name, defaulting it by the index.
'''

from pathlib import Path
from typing import (
    Generic, Protocol, Optional, Any, TypeVar, runtime_checkable,
)
from abc import abstractmethod
from collections.abc import Mapping, Iterable, Sequence

import numpy as np
import pygltflib as gltf

from gltf_builder.holder import _Holder
from gltf_builder.quaternions import Quaternion
from gltf_builder.core_types import (
    ComponentType, JsonObject, PrimitiveMode,
    BufferViewTarget, ElementType, NPTypes, ScopeName
)
from gltf_builder.attribute_types import (
    ColorSpec, JointSpec, Scale, TangentSpec, UvSpec,
    Vector3, Vector3Spec, PointSpec,
    AttributeDataItem, WeightSpec,
    BTYPE, BType,
    vector3,
)
from gltf_builder.matrix import Matrix4
from gltf_builder.compile import (
    _Compileable, T,
    _Scope
)
from gltf_builder.protocols import _BNodeContainerProtocol
from gltf_builder.log import GLTF_LOG


LOG = GLTF_LOG.getChild(Path(__file__).stem)
@runtime_checkable
class Element(_Compileable[T], Protocol):
    def __init__(self,
                 name: str='',
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                index: int=-1,
            ):
        super().__init__(
            extras=extras,
            extensions=extensions,
            name=name,
            index=index,
        )
    
    def __hash__(self):
        return id(self)
        
    def __eq__(self, other: Any):
        return self is other
    
    def __repr__(self):
        return f'<{type(self).__name__} {self!s}>'
    
    def __str__(self):
        if self.name == '':
            if self._index == -1:
                return f'{type(self).__name__}-?'
            else:
                return f'{type(self).__name__}-{self._index}'
        else:
            return f'{type(self).__name__}-{self.name}'


class BBuffer(Element[gltf.Buffer], _Scope, Protocol):
    _scope_name = ScopeName.BUFFER
    @property
    @abstractmethod
    def blob(self) -> bytes:
        ...
    views: _Holder['BBufferView']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def bytearray(self) -> bytearray: ...

    @abstractmethod
    def create_view(self,
                  target: BufferViewTarget,
                  /, *,
                  name: str='',
                  byteStride: int=0,
                  extras:  Optional[JsonObject]=None,
                  extensions: Optional[JsonObject]=None,
                ) -> 'BBufferView':
        '''
        Create a `BBufferView` for this `BBuffer`.
        '''
        ...


class BBufferView(Element[gltf.BufferView], Protocol):
    _scope_name = ScopeName.BUFFER_VIEW
    buffer: BBuffer
    target: BufferViewTarget
    byteStride: int
    accessors: _Holder['BAccessor[NPTypes, BType]']

    @property
    @abstractmethod
    def blob(self) -> bytes: ...

    @abstractmethod
    def memoryview(self, offset: int, size: int) -> memoryview: ...

    @abstractmethod
    def _add_accessor(self, acc: 'BAccessor[NPTypes, BTYPE]') -> None: ...

NP = TypeVar('NP', bound=NPTypes)
NUM = TypeVar('NUM', bound=float|int, covariant=True)

@runtime_checkable
class BAccessor(Element[gltf.Accessor], Protocol, Generic[NP, BTYPE]):
    _scope_name = ScopeName.ACCESSOR
    view: BBufferView
    data: list[BTYPE]
    __array: np.ndarray[tuple[int], np.dtype[NP]]|None = None
    @property
    def array(self) -> np.ndarray[tuple[int], np.dtype[NP]]:
        if self.__array is None:
            self.__array = np.array(self.data, dtype=self.dtype)
        return self.__array
    count: int
    elt_type: ElementType
    componentType: ComponentType
    normalized: bool
    max: Optional[list[float]]
    min: Optional[list[float]]
    componentCount: int = 0
    '''The number of components per element.'''
    componentSize: int = 0
    '''The number of bytes per component.'''
    byteStride: int = 0
    '''The total number of bytes per element.'''
    dtype: type[NP]
    '''The numpy dtype for the data.'''
    bufferType: str = 'f'
    '''The buffer type char for `memoryview.cast()`.'''

    @abstractmethod
    def _add_data(self, data: Sequence[BTYPE]) -> None:
        ...
    '''
    Add a Sequence of data to the accessor.
    '''
    
    @abstractmethod
    def _add_data_item(self, data: BTYPE) -> None:
        ...
        
class BPrimitive(_Compileable[gltf.Primitive], Protocol):
    '''
    Base class for primitives
    '''
    _scope_name = ScopeName.PRIMITIVE
    mode: PrimitiveMode
    points: Sequence[PointSpec]
    attribs: Mapping[str, Sequence[AttributeDataItem]]
    indices: Sequence[int]
    mesh: Optional['BMesh']
    

@runtime_checkable
class BMesh(Element[gltf.Mesh], _Scope, Protocol):
    _scope_name = ScopeName.MESH
    primitives: list[BPrimitive]
    weights: list[float]

    @property
    @abstractmethod
    def detached(self) -> bool:
        '''
        A detached mesh is not added to the builder, but is returned
        to be used as the root of an instanceable object, or to be added
        to multiple nodes and thus to the builder later.
        '''
        ...


    @abstractmethod
    def add_primitive(self, mode: PrimitiveMode,
                      *points: PointSpec,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      JOINTS_0: Optional[Iterable[JointSpec]]=None,
                      WEIGHTS_0: Optional[Iterable[WeightSpec]]=None,
                      extras:  Optional[JsonObject]=None,
                      extensions:  Optional[JsonObject]=None,
                      **attribs: Iterable[AttributeDataItem]
                    ) -> BPrimitive:
        ...
    

@runtime_checkable
class BNode(Element[gltf.Node], _BNodeContainerProtocol, _Scope, Protocol):
    _scope_name = ScopeName.NODE
    mesh: BMesh|None
    root: bool
    __translation: Optional[Vector3]
    @property
    def translation(self) -> Optional[Vector3]:
        return self.__translation
    @translation.setter
    def translation(self, value: Vector3Spec|None):
        if value is not None:
            self.__translation = vector3(value)
        else:
            self.__translation = None
    rotation: Optional[Quaternion]
    scale: Optional[Scale]
    matrix: Optional[Matrix4]

    @property
    @abstractmethod
    def detached(self) -> bool:
        '''
        A detached node is not added to the builder, but is returned
        to be used as the root of an instancable object.
        '''
        ...

    @abstractmethod
    def create_mesh(self,
                name: str='',
                /, *,
                primitives: Iterable['BPrimitive']=(),
                weights: Iterable[float]|None=(),
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                detached: bool=False,
            ) -> 'BMesh':
        '''
        Create a `BMesh` for this `BNode`, or if `detached` is `True`,
        just create a `BMesh` and return it for later use.
        '''
        ...