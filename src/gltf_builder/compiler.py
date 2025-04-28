'''
Compilation interface for the glTF builder.
'''

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from logging import DEBUG
from typing import (
    Literal, Optional, Self, TypeAlias, TypeVar, Protocol, Generic,
    Any, cast, overload, TYPE_CHECKING
)
from pathlib import Path

import pygltflib as gltf

from gltf_builder.core_types import (
    ExtensionsData, ExtrasData, JsonObject, Phase,
    BufferViewTarget, ScopeName
)
from gltf_builder.utils import std_repr
from gltf_builder.log import GLTF_LOG
from gltf_builder.utils import std_repr
if TYPE_CHECKING:
    from gltf_builder.protocols import _BufferViewKey
    from gltf_builder.elements import BBufferView, BBuffer
    from gltf_builder.global_state import _GlobalState


LOG = GLTF_LOG.getChild(Path(__name__).stem)


_GLTF = TypeVar('_GLTF', bound=gltf.Property|JsonObject, covariant=True)
'''
Type variable for glTF elements.
This is used to indicate the type of the gltf element being compiled.
'''


_Collected: TypeAlias = tuple[
    '_Compilable',
    Sequence['_Collected'],
]


_ReturnCollect: TypeAlias = Iterable[_Collected]|None
_ReturnSizes: TypeAlias = int|None
_ReturnOffsets: TypeAlias = int|None
_ReturnBuild: TypeAlias = _GLTF|None
_ReturnView: TypeAlias = None
_ReturnExtensions: TypeAlias = set[str]|None
_DoCompileReturn: TypeAlias = (
    _ReturnCollect|
    _ReturnSizes|
    _ReturnOffsets|
    _ReturnBuild[_GLTF]|
    _ReturnView|
    _ReturnExtensions
)


class _BaseCompileState(Generic[_GLTF]):
    '''
    Base state for compiling an element.

    Separate from `_CompileState` to allow for more generic use.
    '''
    name: str
    _index: int|None = None
    @property
    def index(self) -> int:
        if self._index is None:
            raise ValueError(f'Index not set for {type(self)}')
        return self._index
    @index.setter
    def index(self, index: int):
        if self._index is None:
            self._index = index
        elif self._index != index:
            raise ValueError(f'Index already set, old={self._index}, new={index}')
    _len: int|None = None

    _byteOffset: int|None = 0
    @property
    def byteOffset(self) -> int:
        if self._byteOffset is None:
            raise ValueError(f'Byte offset not set for {type(self)}')
        return self._byteOffset

    @byteOffset.setter
    def byteOffset(self, offset: int):
        if self._byteOffset is None:
            self._byteOffset = offset
        elif self._byteOffset != offset:
            raise ValueError(f'Byte offset already set, old={self._byteOffset}, new={offset}')

    phases: list[Phase]
    compiled: _GLTF|None
    collected: _Collected|None

    def __init__(self,
                 name: str,
                 byteOffset: int|None=0
                ) -> None:
        self.name = name
        self.phases = []
        self.compiled = None
        self.collected = None
        self._byteOffset = byteOffset

    def __len__(self) -> int:
        if self._len is None:
            raise ValueError(f'Length not set for {self}')
        return self._len

    def __bool__(self) -> bool:
        if self._len is None:
            return False
        return bool(self._len)


_STATE = TypeVar('_STATE', bound=_BaseCompileState)

'''
Type variable for the compile state.
This is used to indicate the type of the compile state.
'''


class _CompileState(_BaseCompileState[_GLTF], Generic[_GLTF, _STATE]):
    '''
    State for compiling an element.
    '''
    element: '_Compilable[_GLTF, _STATE]'

    def __init__(self,
                 element: '_Compilable[_GLTF, _STATE]',
                 name: str,
                byteOffset: int|None=0,
                ) -> None:
        super().__init__(
            name=name,
            byteOffset=byteOffset,
        )
        self.element = element

    @property
    def phase(self) -> Phase|None:
        '''
        The last phase of the compilation reached.
        '''
        return self.phases[-1] if self.phases else None

    def __repr__(self):
        return std_repr(self, (
            'name',
            ('index', self._index),
            ('byteOffset', self._byteOffset, "offset"),
            ('len', self._len),
            'phase',
        ))

class _Compilable(Generic[_GLTF, _STATE], Protocol):
    _scope_name: ScopeName

    extensions: ExtensionsData
    extras: ExtrasData
    _collected: _Collected|None = None
    name: str = ''

    @classmethod
    def state_type(cls) -> type[_STATE]:
        '''
        Create a new compile state for the element.
        '''
        return cast(type[_STATE], _CompileState) # pragma: no cover

    def __init__(self,
                 name: str='', /,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                ):
        self.extensions = dict(extensions) if extensions else {}
        self.extras = dict(extras) if extras else {}
        self.name = name

    def _clone_attributes(self) -> dict[str, Any]:
        '''
        Clone the attributes of the object.
        '''
        return {} # pragma: no cover

    def clone(self, name: str='', /,
              extras: Optional[ExtrasData]=None,
              extensions: Optional[ExtensionsData]=None,
              **kwargs: Any,
            ) -> Self:
        '''
        Clone the object, copying the name, extras, and extensions.
        '''
        kwargs = {
            'extras': {**self.extras, **(extras or {})},
            'extensions': {**self.extensions, **(extensions or {})},
            **self._clone_attributes(),
            **kwargs,
        }
        return self.__class__(
            name or self.name,
            **kwargs,
        )

    @overload
    def compile(self,
                gbl: '_GlobalState',
                scope: '_Scope',
                phase: Literal[Phase.COLLECT],
                /
                ) -> _Collected: ...
    @overload
    def compile(self,
                gbl: '_GlobalState',
                scope: '_Scope',
                phase: Literal[Phase.SIZES],
                /
            ) -> int: ...
    @overload
    def compile(self,
                gbl: '_GlobalState',
                scope: '_Scope',
                phase: Literal[Phase.OFFSETS],
                /
            ) -> int: ...
    @overload
    def compile(self,
                gbl: '_GlobalState',
                scope: '_Scope',
                phase: Literal[Phase.BUILD],
                /
                ) -> _GLTF: ...
    @overload
    def compile(self,
                gbl: '_GlobalState',
                scope: '_Scope',
                phase: Literal[
                        Phase.VIEWS,
                        Phase.ENUMERATE,
                        Phase.PRIMITIVES,
                        Phase.PRIMITIVES,
                        Phase.VERTICES,
                        Phase.BUFFERS,
                        Phase.EXTENSIONS,
                    ],
                /
            ) -> None: ...
    def compile(self,
                gbl: '_GlobalState',
                scope: '_Scope',
                phase: Phase,
                /
                ) -> '_GLTF|int|_Collected|set[str]|None':
        state = gbl.state(self)
        if phase in state.phases:
            match phase:
                case Phase.COLLECT:
                    return state.collected
                case Phase.SIZES:
                    return len(state)
                case Phase.OFFSETS:
                    return state.byteOffset
                case Phase.BUILD:
                    return cast(_GLTF, state.compiled)
                case _:
                    return None
        else:
            LOG.debug('Compiling %s in phase %s', self, phase)

            def _do_compile():
                return self._do_compile(gbl, scope, phase, state)
            state.phases.append(phase)
            match phase:
                case Phase.COLLECT:
                    state.name = gbl._gen_name(self)
                    items = cast(_ReturnCollect, (_do_compile() or ()))
                    return (self, tuple(items or ()),)
                case Phase.SIZES:
                    bytelen = cast(_ReturnSizes, _do_compile() or 0)
                    assert bytelen is not None
                    state._len = bytelen
                    LOG.debug('%s has length %s', self, state._len)
                    return bytelen
                case Phase.OFFSETS:
                    _do_compile()
                    if state.byteOffset >= 0:
                        if LOG.isEnabledFor(DEBUG):
                            LOG.debug('%s has offset %d',
                                    self, state.byteOffset,
                                    )
                    else:
                        LOG.debug(f'{self} has offset {state.byteOffset}')
                        return state.byteOffset + len(state)
                    return -1
                case Phase.EXTENSIONS:
                    if self.extensions:
                        return set(self.extensions.keys())
                    return None
                case Phase.BUILD:
                    if state.compiled is None:
                        state.compiled = cast(_GLTF, _do_compile())
                    return state.compiled
                case _:
                    _do_compile()


    @abstractmethod
    def _do_compile(self,
                    gbl: '_GlobalState',
                    scope: '_Scope',
                    phase: Phase,
                    state: _STATE,
                    /
                ) -> _DoCompileReturn[_GLTF]: ...


class _Scope(Protocol):
    '''
    Scope for allocating `BBufferView` and `BAccessor` objects.

    Scopes include meshes, nodes, and buffers, with meshes at the top
    and buffers at the bottom. Between, the nodes have their own hierarchy,
    so a parent node can have a view scope, but a child node can have an
    accessor scope. This allows control over sharing of vertices, accessors,
    and views. Vertex sharing follows the same scope as accessors, while
    views can follow the same scope as the accessors they contain, or any
    scope above them.
    '''
    __views: dict['_BufferViewKey', 'BBufferView']

    __target_buffer: 'BBuffer'
    @property
    def target_buffer(self) -> 'BBuffer':
        return self.__target_buffer

    __global: '_GlobalState'
    @property
    def gbl(self) -> '_GlobalState':
        return self.__global

    def __init__(self,
                gbl: '_GlobalState',
                buffer: 'BBuffer',
                is_accessor_scope: bool=False,
                is_view_scope: bool=False,
            ):
        self.__global = gbl
        self.__target_buffer = buffer
        self.__views = {}


    def _get_view(self,
                buffer: 'BBuffer',
                target: BufferViewTarget,
                byteStride: int=0,
                name: str='',
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
            ) -> 'BBufferView':
        from gltf_builder.protocols import _BufferViewKey
        key = _BufferViewKey(buffer, target, byteStride, name)
        view = self.__views.get(key, None)
        if view is None:
            from gltf_builder.views import _BufferView
            view = _BufferView(
                buffer, name,
                target=target,
                byteStride=byteStride,
                extras=extras,
                extensions=extensions,
            )
            self.__views[key] = view
        return view