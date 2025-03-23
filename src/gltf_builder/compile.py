'''
Compilation interfce for the glTF builder.
'''

from abc import abstractmethod
from typing import (
    TypeAlias, TypeVar, Protocol, Generic, Optional,
    Any, Mapping, overload, NamedTuple, TYPE_CHECKING
)
from pathlib import Path

from gltf_builder.types import (
    Phase, EMPTY_MAP, ElementType, ComponentType, BufferViewTarget
)
from gltf_builder.holder import Holder
from gltf_builder.protocols import BuilderProtocol
from gltf_builder.log import GLTF_LOG
if TYPE_CHECKING:
    from gltf_builder.element import BAccessor, BBufferView
LOG = GLTF_LOG.getChild(Path(__name__).stem)


Collected: TypeAlias = tuple['Compileable', list['Collected']]

T = TypeVar('T')
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
    
    def _relocate(self, new_offset: int):
        '''
        Relocate an item to a new byte offset.
        '''
        self.__byte_offset = new_offset
    
    extensions: dict[str, Any]
    extras: dict[str, Any]
    name: str = ''
    index: int = -1

    def __init__(self,
                 extras: Mapping[str, Any]=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
                ):
        self.__phases = []
        self.__collected: Collected
        self.extras = extras
        self.extensions = extensions
    
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.COLLECT
                ) -> list['Collected']: ...
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.SIZES) -> int|None: ...
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.OFFSETS) -> int: ...
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.BUILD) -> T: ...
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase) -> T|int|None:
        from gltf_builder.element import BAccessor
        if phase in self.__phases:
            match phase:
                case Phase.COLLECT:
                    return self.__collected
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
        else:
            LOG.debug('Compiling %s in phase %s', self, phase)
            self.__phases.append(phase)
            match phase:
                case Phase.COLLECT:
                    self.name = builder._gen_name(self)
                    self.__collected = self, self._do_compile(builder, scope, phase) or []
                    return self.__collected
                case Phase.SIZES:
                    bytelen = self._do_compile(builder, scope, phase)
                    if bytelen is not None:
                        self._len = bytelen
                        LOG.debug('%s has length %s', self, self._len)
                        return bytelen
                    return 0
                case Phase.OFFSETS:
                    self._do_compile(builder, scope, phase)
                    if self.byteOffset >= 0:
                        if isinstance(self, BAccessor):
                            LOG.debug('%s has offset %d(+%d)',
                                      self, self.byteOffset, self._view.byteOffset)
                        else:
                            LOG.debug(f'{self} has offset {self.byteOffset}')
                        return self.byteOffset + self._len
                    return -1
                case Phase.BUILD:
                    if self.__compiled is None:
                        self.__compiled = self._do_compile(builder, scope, phase)
                    return self.__compiled
                case _:
                    return self._do_compile(builder, scope, phase)

    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.COLLECT
                    ) -> list['Compileable']: ...
    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.SIZES) -> int: ...
    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.BUILD) -> T: ...
    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase) -> None: ...
    @abstractmethod
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase) -> T|int|None:
        ...

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
    
    def __eq__(self, other):
        return (self.type is other.type
                and self.componentType == other.componentType
                and self.elementType == other.elementType
                and self.name == other.name)


class _Scope(Protocol):
    __parent: Optional['_Scope'] = None
    __views: dict[BufferViewTarget, 'BBufferView']
    __accessors: dict['BAccessor']
    __is_accessor_scope: bool = False
    __is_view_scope: bool = False
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
    def __init__(self,
                parent: Optional['_Scope']=None,
                is_accessor_scope: bool=False,
                is_view_scope: bool=False,
            ):
        self.__parent = parent
        self.__views = {}
        self.__accessors = Holder()
        self.__is_accessor_scope = is_accessor_scope
        self.__is_view_scope = is_view_scope

    def _get_accessor(self, 
                    eltType: ElementType,
                    componenType: ComponentType,
                    name: str = '',
                    normalized: bool=False,
                    BufferViewTarget: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
                ) -> 'BAccessor':
        key = AccessorKey(eltType, componenType, normalized, name)
        accessor = self.__accessors.get(key, None)
        if accessor is None:
            view = self._get_view(BufferViewTarget)
            accessor = BAccessor(view, name,)
            accessor = view.add_accessor(eltType, componenType, normalized=normalized)



    def _get_view(self, target: BufferViewTarget, for_object: Compileable) -> 'BBufferView':
        ...
    