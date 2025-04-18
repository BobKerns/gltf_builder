'''
Compilation interfce for the glTF builder.
'''

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import (
    Literal, Optional, TypeAlias, TypeVar, Protocol, Generic,
    Any, cast, overload, NamedTuple, TYPE_CHECKING
)
from pathlib import Path

import pygltflib as gltf

from gltf_builder.core_types import (
    JsonObject, NPTypes, Phase, ElementType,
    ComponentType, BufferViewTarget
)
from gltf_builder.attribute_types import BType
from gltf_builder.log import GLTF_LOG
from gltf_builder.utils import decode_stride
if TYPE_CHECKING:
    from gltf_builder.protocols import BufferViewKey, BuilderProtocol
    from gltf_builder.element import BAccessor, BBufferView, BBuffer
LOG = GLTF_LOG.getChild(Path(__name__).stem)


T = TypeVar('T', bound=gltf.Property, covariant=True)


Collected: TypeAlias = tuple[
    'Compileable[gltf.Property]',
    Sequence['Collected'],
]


ReturnCollect_: TypeAlias = Iterable[Collected]|None
ReturnSize_: TypeAlias = int|None
ReturnOffset_: TypeAlias = int|None
ReturnBuild_: TypeAlias = T|None
ReturnView_: TypeAlias = None
DoCompileReturn: TypeAlias = (
    ReturnCollect_|
    ReturnSize_|
    ReturnOffset_|
    ReturnBuild_[T]|
    ReturnView_
)

class Compileable(Generic[T], Protocol):
    __phases: list[Phase]
    __compiled: T|None = None
    _len: int = -1

    __byte_offset: int = -1
    @property
    def byteOffset(self) -> int:
        return self.__byte_offset
    @byteOffset.setter
    def byteOffset(self, offset: int):
        if self.__byte_offset == -1:
            self.__byte_offset = offset
        elif self.__byte_offset != offset:
            raise ValueError(f'Byte offset already set, old={self.__byte_offset}, new={offset}')
    
    def relocate(self, new_offset: int, nwq_index: int=-1):
        '''
        Relocate an item to a new byte offset ns iswzx.
        '''
        self.__byte_offset = new_offset
        self.__index = nwq_index
    
    extensions: JsonObject
    extras: JsonObject
    collected: Collected|None = None
    name: str = ''

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
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 name: str='',
                index: int=-1,
                ):
        self.__phases = []
        self.extensions = dict(extensions) if extensions else {}
        self.extras = dict(extras) if extras else {}
        self.name = name
        self.__index = index

    def log_offset(self):
        if self.byteOffset >= 0:
            LOG.debug(f'{self} has offset {self.byteOffset}')
    
    @overload
    def compile(self, builder: 'BuilderProtocol', scope: 'Scope_', phase: Literal[Phase.COLLECT]
                ) -> Collected: ...
    @overload
    def compile(self, builder: 'BuilderProtocol', scope: 'Scope_', phase: Literal[Phase.SIZES]) -> int: ...
    @overload
    def compile(self, builder: 'BuilderProtocol', scope: 'Scope_', phase: Literal[Phase.OFFSETS]) -> int: ...
    @overload
    def compile(self, builder: 'BuilderProtocol', scope: 'Scope_', phase: Literal[Phase.BUILD]) -> T: ...
    @overload
    def compile(self,
                builder: 'BuilderProtocol', 
                scope: 'Scope_',
                phase: Literal[
                        Phase.VIEWS,
                        Phase.ENUMERATE,
                        Phase.PRIMITIVES,
                        Phase.PRIMITIVES,
                        Phase.VERTICES,
                        Phase.BUFFERS
                    ],
            ) -> None: ...
    def compile(self, builder: 'BuilderProtocol', scope: 'Scope_', phase: Phase,
                ) -> 'T|int|Collected|None':
        from gltf_builder.element import BAccessor
        if phase in self.__phases:
            match phase:
                case Phase.COLLECT:
                    return self.collected
                case Phase.SIZES:
                    if hasattr(self, '_len'):
                        return self._len
                    return 0
                case Phase.OFFSETS:
                    if hasattr(self, 'byteOffset'):
                        return self.byteOffset
                    return -1
                case Phase.BUILD:
                    return self.__compiled
                case _:
                    return None
        else:
            LOG.debug('Compiling %s in phase %s', self, phase)
            self.__phases.append(phase)
            match phase:
                case Phase.COLLECT:
                    self.name = builder.gen_name_(self) or '' #type: ignore
                    items = cast(ReturnCollect_, (self._do_compile(builder, scope, phase) or ()))
                    return (self, tuple(items or ()),)
                case Phase.SIZES:
                    bytelen = cast(ReturnSize_, self._do_compile(builder, scope, phase))
                    if bytelen is not None:
                        self._len = bytelen
                        LOG.debug('%s has length %s', self, self._len)
                        return bytelen
                    return 0
                case Phase.OFFSETS:
                    self._do_compile(builder, scope, phase)
                    if self.byteOffset >= 0:
                        LOG.debug('%s has offset %d(+%d)',
                                self, self.byteOffset,
                                self.view.byteOffset if isinstance(self, BAccessor) else 0
                                )
                    else:
                        LOG.debug(f'{self} has offset {self.byteOffset}')
                        return self.byteOffset + self._len
                    return -1
                case Phase.BUILD:
                    if self.__compiled is None:
                        self.__compiled = cast(T, self._do_compile(builder, scope, phase))
                    return self.__compiled
                case _:
                    self._do_compile(builder, scope, phase)

    @abstractmethod
    def _do_compile(self, builder: 'BuilderProtocol', scope: 'Scope_', phase: Phase) -> DoCompileReturn[T]: ...

    def __len__(self) -> int:
        return self._len
    
    def __bool__(self) -> bool:
        if self._len < 0:
            return False
        return bool(self._len)

class AccessorKey(NamedTuple):
    type: ElementType
    componentType: ComponentType
    normalized: bool
    name: str = ''
    
    def __hash__(self):
        return hash((self.type, self.componentType, self.normalized, self.name))
    
    def __eq__(self, other: Any):
        if isinstance(other, AccessorKey):
            if self.type == other.type:
                if self.componentType == other.componentType:
                    if self.normalized == other.normalized:
                        return self.name == other.name
        return False


class Scope_(Protocol):
    '''
    Scope for allocating `BBufferView` and `BAccessor` objects.

    Scopes include meshes, nodes, and buffers, with meshes at the top
    and buffersat the bottom. Between, the nodes have their own hierarchy,
    so a parent node can have a view scope, but a child node can have an
    accessor scope. This allows control over sharing of verticies, accessors,
    and views. Vertex sharing follows the same scope as accessors, while
    views can follow the same scope as the accessors they contain, or any
    scope above them.
    '''
    __views: dict['BufferViewKey', 'BBufferView']
    __accessors: dict[AccessorKey, 'BAccessor[NPTypes, BType]']
    __is_accessor_scope: bool = False
    __is_view_scope: bool = False
    __buffer: 'BBuffer'
    @property
    def target_buffer(self) -> 'BBuffer':
        return self.__buffer
    
    __buidler: 'BuilderProtocol'
    @property
    def builder(self) -> 'BuilderProtocol':
        return self.__buidler
    
    def __init__(self,
                builder: 'BuilderProtocol',
                buffer: 'BBuffer',
                is_accessor_scope: bool=False,
                is_view_scope: bool=False,
            ):
        self.__buidler = builder
        self.__buffer = buffer
        self.__views = {}
        self.__accessors = dict()
        self.__is_accessor_scope = is_accessor_scope
        self.__is_view_scope = is_view_scope

    def _get_accessor(self, 
                    eltType: ElementType,
                    componenType: ComponentType,
                    btype: type[BType],
                    name: str = '',
                    normalized: bool=False,
                    BufferViewTarget: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
                    extras: Optional[JsonObject]=None,
                    extensions: Optional[JsonObject]=None,
                ) -> 'BAccessor[NPTypes, BType]':
        key = AccessorKey(eltType, componenType, normalized, name)
        byteStride = decode_stride(eltType, componenType)
        accessor = self.__accessors.get(key, None)
        if accessor is None:
            view = self.target_buffer.get_view(
                self.__buffer, BufferViewTarget,
                byteStride=byteStride,
                name=name,
                extras=extras,
                extensions=extensions,
            )
            builder = self.__buidler
            accessor = builder.create_accessor_(
                elementType=eltType,
                componentType=componenType,
                btype=btype,
                buffer=self.target_buffer,
                name=name,
                count=0,
                normalized=normalized,
                target=BufferViewTarget,
            )
            view.add_accessor(accessor)
            self.__accessors[key] = accessor
        return accessor

    def get_view(self,
                buffer: 'BBuffer',
                target: BufferViewTarget,
                byteStride: int=0,
                name: str='',
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
            ) -> 'BBufferView':
        key = BufferViewKey(buffer, target, byteStride, name)
        view = self.__views.get(key, None)
        if view is None:
            view = self.target_buffer.create_view(
                target,
                byteStride=byteStride,
                name=name,
                extras=extras,
                extensions=extensions,
            )
            self.__views[key] = view
        return view