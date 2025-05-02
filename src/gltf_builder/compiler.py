'''
Compilation interface for the glTF builder.
'''

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from enum import StrEnum
from logging import DEBUG
from typing import (
    Literal, Optional, Self, TypeAlias, TypeVar, Generic,
    Any, cast, overload, TYPE_CHECKING
)
from pathlib import Path

import pygltflib as gltf

from gltf_builder.core_types import (
    ExtensionData, ExtensionsData, ExtrasData, Phase,
    BufferViewTarget, ScopeName
)
from gltf_builder.utils import std_repr
from gltf_builder.log import GLTF_LOG
if TYPE_CHECKING:
    from gltf_builder.protocols import _BufferViewKey
    from gltf_builder.elements import (
        Element, BBuffer, BBufferView,
    )
    from gltf_builder.extensions import Extension
    from gltf_builder.global_state import GlobalState


LOG = GLTF_LOG.getChild(Path(__name__).stem)


_GLTF = TypeVar('_GLTF', bound=gltf.Property|ExtensionData, covariant=True)
'''
Type variable for glTF elements.
This is used to indicate the type of the gltf element being compiled.
'''


_Collected: TypeAlias = tuple[
    '_Compilable',
    Sequence['_Collected'],
]

_UNIMPLEMENTED_PHASES = (
    Phase.VERTICES,
)


_SLOTS: tuple[str, ...] = tuple(
    str(s)
    for ss in (
        Phase._member_names_,
        (
            'name',
            '_index',
            'element',
            'extension_objects',
            #'__dict__',
        )
    )
    for s in ss
)


_ReturnCollect: TypeAlias = Iterable[_Collected]|None
_ReturnSizes: TypeAlias = int|None
_ReturnOffsets: TypeAlias = int|None
_ReturnBuild: TypeAlias = _GLTF|None
_ReturnView: TypeAlias = None
_ReturnExtensions: TypeAlias = set['Extension']|None
_DoCompileReturn: TypeAlias = (
    _ReturnCollect|
    _ReturnSizes|
    _ReturnOffsets|
    _ReturnBuild[_GLTF]|
    _ReturnView|
    _ReturnExtensions
)


_STATE = TypeVar('_STATE', bound='_CompileState')
'''
Type variable for the compile state.
This is used to indicate the type of the compile state.
'''

_ELEMENT = TypeVar('_ELEMENT', bound='Element')


_STATEX = TypeVar('_STATEX', bound='_CompileState')
class _Scope:
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
    __slots__ = (
        '__views', '__target_buffer', '__global',
    )
    __views: dict['_BufferViewKey', 'BBufferView']

    __target_buffer: 'BBuffer'
    @property
    def target_buffer(self) -> 'BBuffer':
        return self.__target_buffer

    __global: 'GlobalState'
    @property
    def globl(self) -> 'GlobalState':
        return self.__global

    def __init__(self,
                globl: 'GlobalState',
                buffer: 'BBuffer',
                is_accessor_scope: bool=False,
                is_view_scope: bool=False,
            ):
        self.__global = globl
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

class Progress(StrEnum):
    '''
    Progress of the compilation.
    '''
    NONE = 'none'
    '''
    Indicates that a compilation phase has not yet started.
    '''
    IN_PROGRESS = 'in_progress'
    '''
    Indicates that a compilation phase is in progress but not yet complete.
    '''
    DONE = 'done'
    '''
    Indicates that a compilation phase is complete.
    '''

class _CompileState(Generic[_GLTF, _STATE, _ELEMENT], _Scope): # type: ignore[misc]
    '''
    State for compiling an element.
    '''
    __slots__ = _SLOTS
    name: str
    _index: int|None
    phase: Phase
    PRIMITIVES: Progress
    '''
    Process the data for the primitives for the glTF file.
    '''
    COLLECT: _Collected|Progress
    '''
    Create the accessors and views for the glTF file, and collect all
    subordinate objects.
    '''
    ENUMERATE: int|Progress
    '''
    Assign index values to each object
    '''
    VERTICES: Progress
    '''
    Optimize the vertices for the glTF file.
    '''
    SIZES: int|Progress
    '''
    Calculate sizes for the accessors and views for the glTF file.
    '''
    OFFSETS: int|Progress
    '''
    Calculate offsets for the accessors and views for the glTF file.
    '''
    BUFFERS: Progress
    '''
    Initialize buffers to receive data
    '''
    VIEWS: Progress
    '''
    Initialize buffer views to receive data
    '''
    EXTENSIONS: set[str]|None|Progress
    '''
    Collect the set of used extensions for the glTF file.
    '''
    BUILD: gltf.GLTF2|gltf.Property|Progress
    '''
    Construct the binary data for the glTF file.
    '''
    @property
    def index(self) -> int:
        if self._index is None:
            raise ValueError(f'Index not set for {type(self.element).__name__!s}')
        return self._index
    @index.setter
    def index(self, index: int):
        if self._index is None:
            self._index = index
        elif self._index != index:
            raise ValueError(f'Index already set, old={self._index}, new={index}')

    element: _ELEMENT
    extension_objects: set['Extension']

    def __init__(self,
                 element: _ELEMENT,
                 name: str,
                byteOffset: int|None=0,
                ) -> None:
        self.name = name
        self._byteOffset = byteOffset
        self._index = None
        self._len = None
        self.element = element
        self.extension_objects = set(element.extension_objects)
        self.PRIMITIVES: Progress = Progress.NONE
        self.COLLECT: _Collected|Progress = Progress.NONE
        self.ENUMERATE: int|Progress = Progress.NONE
        self.VERTICES: Progress = Progress.NONE
        self.SIZES: int|Progress = Progress.NONE
        self.OFFSETS: int|Progress = Progress.NONE
        self.BUFFERS: Progress = Progress.NONE
        self.VIEWS: Progress = Progress.NONE
        self.EXTENSIONS: set[str]|None|Progress = Progress.NONE
        self.BUILD: gltf.GLTF2|gltf.Property|Progress = Progress.NONE

    def __repr__(self):
        return std_repr(self, (
            'name',
            ('index', self._index),
            ('byteOffset', self._byteOffset, "offset"),
            ('len', self._len),
            'phase',
        ))


class _GlobalCompileState(Generic[_GLTF, _STATEX, _ELEMENT],
                          _CompileState[_GLTF, _STATEX, _ELEMENT]):
    __slots__ = (
        '_len', '_byteOffset',
    )
    _len: int|None
    _byteOffset: int|None
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

    def __len__(self) -> int:
        if self._len is None:
            raise ValueError(f'Length not set for {self}')
        return self._len

    def __bool__(self) -> bool:
        if self._len is None:
            return False
        return bool(self._len)

class _Compilable(Generic[_GLTF, _STATE]):
    __slots__ = (
        'name', 'extensions', 'extras', 'extension_objects',
    )
    _scope_name: ScopeName

    extensions: ExtensionsData
    '''
    The JSON extension data supplied for the element.
    '''
    extension_objects: set['Extension']
    '''
    A list of supplied extension instances to be compiled
    into this element.

    This is used to allow extensions to be added to elements
    at a higher level than supplying the JSON data.
    '''
    extras: ExtrasData
    name: str

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
                 extension_objects: Optional[Iterable['Extension']]=None,
                ):
        self.extensions = dict(extensions) if extensions else {}
        self.extras = dict(extras) if extras else {}
        self.name = name
        self.extension_objects = set(extension_objects or ())

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
                globl: 'GlobalState',
                phase: Literal[Phase.COLLECT],
                /
                ) -> _Collected: ...
    @overload
    def compile(self,
                globl: 'GlobalState',
                phase: Literal[Phase.SIZES],
                /
            ) -> int: ...
    @overload
    def compile(self,
                globl: 'GlobalState',
                phase: Literal[Phase.OFFSETS],
                /
            ) -> int: ...
    @overload
    def compile(self,
                globl: 'GlobalState',
                phase: Literal[Phase.BUILD],
                /
                ) -> _GLTF: ...
    @overload
    def compile(self,
                globl: 'GlobalState',
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
                globl: 'GlobalState',
                phase: Phase,
                /
                ) -> '_GLTF|int|_Collected|set[str]|None':
        state = cast(_STATE, globl.state(cast('Element', self)))
        progress = getattr(state, phase.name)
        match progress:
            case Progress.IN_PROGRESS:
                raise RuntimeError(f"Recursive compilation detected on {self} in phase {phase.name}")
            case Progress.NONE:
                setattr(state, phase.name, Progress.IN_PROGRESS)
                LOG.debug('Compiling %s in phase %s', self, phase.name)

                def _do_compile():
                    return self._do_compile(globl, phase, state)
                match phase:
                    case Phase.COLLECT:
                        def e_collect(ext: 'Extension'):
                            return ext.compile(globl, Phase.COLLECT)

                        globl.extension_objects.update(
                            globl.state(cast(Extension, e))
                            for group in (self.extension_objects,
                                        state.extension_objects)
                            for ext in group
                            for e in e_collect(ext)
                            if e is not None
                        )
                        state.name = globl._gen_name(self)
                        items = cast(_ReturnCollect, (_do_compile() or ()))
                        result= (
                            self,
                            tuple(
                                i
                                  for i in items or ()
                                  if i
                                )
                        )
                        state.COLLECT = result
                        return result
                    case Phase.SIZES:
                        assert isinstance(state, _GlobalCompileState)
                        bytelen = cast(_ReturnSizes, _do_compile() or 0)
                        assert bytelen is not None
                        state._len = bytelen
                        LOG.debug('%s has length %s', self, state._len)
                        state.SIZES = bytelen
                        return bytelen
                    case Phase.OFFSETS:
                        assert isinstance(state, _GlobalCompileState)
                        _do_compile()
                        if state.byteOffset >= 0:
                            if LOG.isEnabledFor(DEBUG):
                                LOG.debug('%s has offset %d',
                                        self, state.byteOffset,
                                        )
                            offset = state.byteOffset
                        else:
                            LOG.debug(f'{self} has offset {state.byteOffset}')
                            offset = state.byteOffset + len(state)
                        state.OFFSETS = offset
                        return offset
                    case Phase.EXTENSIONS:
                        extensions = None
                        if self.extensions:
                            extensions = set(self.extensions.keys())
                        state.EXTENSIONS = extensions
                        return extensions
                    case Phase.BUILD:
                        built = cast(_GLTF, _do_compile())
                        state.BUILD = cast(gltf.GLTF2|gltf.Property, built)
                        return built
                    case _:
                        _do_compile()
                        setattr(state, phase.name, Progress.DONE)
            case _:
                return getattr(state, phase.name)


    @abstractmethod
    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _STATE,
                    /
                ) -> _DoCompileReturn[_GLTF]: ...


    def compile_extensions(self,
                        globl: 'GlobalState',
                        phase: Phase,
                        state: _STATE,
                        /
                    ) -> set[str]|None:
        '''
        Compile the extensions for the element.
        '''
        if phase == Phase.EXTENSIONS:
            return self.compile(globl, phase)
        return None

