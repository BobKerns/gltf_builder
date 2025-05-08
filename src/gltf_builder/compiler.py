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
    EntityFlags, ExtensionData, ExtensionsData, ExtrasData, Phase,
    BufferViewTarget, EntityType
)
from gltf_builder.holders import _Holder
from gltf_builder.utils import std_repr
from gltf_builder.log import GLTF_LOG
if TYPE_CHECKING:
    from gltf_builder.entities import (
        Entity, BBuffer, BBufferView,
    )
    from gltf_builder.global_state import GlobalState
    from gltf_builder.extensions import Extension


LOG = GLTF_LOG.getChild(Path(__name__).stem)


_GLTF = TypeVar('_GLTF', bound=gltf.Property|ExtensionData, covariant=True)
'''
Type variable for glTF entities.
This is used to indicate the type of the gltf entity being compiled.
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
            'entity',
            '__extensions',
            '__extras',
            '_extension_objects',
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

_ENTITY = TypeVar('_ENTITY', bound='Entity')

class _BinaryDataScope:
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
    # __slots__ = (
    #     '__views', '__target_buffer', '__global',
    # )
    # __views: dict['_BufferViewKey', 'BBufferView']

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


    def get_view(self,
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

class _CompileState(Generic[_GLTF, _STATE, _ENTITY], _BinaryDataScope): # type: ignore[misc]
    '''
    State for compiling an entity.
    '''
    # __slots__ = _SLOTS
    name: str
    _index: int|None
    phase: Phase
    __extensions: ExtensionsData|None

    @property
    def extensions(self) -> ExtensionsData:
        '''
        The JSON extension data supplied for the entity.
        '''
        if self.__extensions is None:
            self.__extensions = {}
        return self.__extensions

    @extensions.setter
    def extensions(self, value: ExtensionsData):
        '''
        Individual extension keys are independent, so assignment merges.
        '''
        self.extensions.update(value)


    __extras: ExtrasData|None
    @property
    def extras(self) -> ExtrasData:
        '''
        The JSON extra data supplied for the entity.
        '''
        if self.__extras is None:
            self.__extras = {}
        return self.__extras
    @extras.setter
    def extras(self, value: ExtrasData):
        '''
        Extras are unmanaged data, so assignment replaces.
        '''
        self.__extras = value


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
            raise ValueError(f'Index not set for {type(self.entity).__name__!s}')
        return self._index
    @index.setter
    def index(self, index: int):
        if self._index is None:
            self._index = index
        elif self._index != index:
            raise ValueError(f'Index already set, old={self._index}, new={index}')

    entity: _ENTITY

    __extension_objects: _Holder['Extension']|None
    @property
    def extension_objects(self) -> _Holder['Extension']:
        '''
        The extension objects for the entity.
        '''
        if self.__extension_objects is None:
            from gltf_builder.extensions import Extension
            self.__extension_objects = _Holder(Extension)
        return self.__extension_objects
    @extension_objects.setter
    def extension_objects(self, value: Iterable['Extension']):
        '''
        Add extension objects to the entity.
        '''
        self.extension_objects.add_from(value)

    def __init__(self,
                 entity: _ENTITY,
                 name: str, /,
                ) -> None:
        self.name = name
        self._index = None
        self._len = None
        self.entity = entity
        self.__extensions = None
        self.__extras = None
        self.__extension_objects = None
        if self.extension_objects:
            self.extension_objects = entity.extension_objects
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
            ('len', self._len),
            'phase',
        ))

_BIN = TypeVar('_BIN')

class _BinaryCompileState(Generic[_BIN, _GLTF, _STATE, _ENTITY], _CompileState[_GLTF, '_STATE', '_ENTITY']):
    '''
    State for Entities that hold binary data.
    '''
   #  __slots__ = (
   #      *_CompileState.__slots__,
   #      'data'
   #      '_len', '_byteOffset',
   #  )
    data: _BIN
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

    def __init__(self,
                    entity: _ENTITY,
                    data: _BIN,
                    name: str = '',
                    /,
                    byteOffset: Optional[int] = None,
                    len: Optional[int] = None,
                    ) -> None:
        super().__init__(entity, name)
        self.data = data
        if byteOffset is not None:
            self._byteOffset = byteOffset
        else:
            self._byteOffset = None
        if len is not None:
            self._len = len
        else:
            self._len = None


    def __len__(self) -> int:
        if self._len is None:
            raise ValueError(f'Length not set for {self}')
        return self._len

    def __bool__(self) -> bool:
        if self._len is None:
            return False
        return bool(self._len)

    def __repr__(self):
        return std_repr(self, (
            'name',
            ('index', self._index),
            ('byteOffset', self._byteOffset, "offset"),
            ('len', self._len),
            'phase',
        ))

class _GlobalCompileState(Generic[_GLTF, _STATE, _ENTITY],
                          _CompileState[_GLTF, _STATE, _ENTITY]):
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
    '''
    Base implementation class for all entities that can be
    compiled into a glTF file.
    '''
    # __slots__ = (
    #     '_name',  '_flags',
    #     '_initial_state',
    # )
    name: str = ''
    _entity_type: EntityType
    '''
    CLASS VARIABLE

    The type of entity that this class represents.
    '''

    _initial_state: _STATE|None
    '''
    The initial state of the entity, or None if no non-default values have been set.
    '''
    @property
    def initial_state(self) -> _STATE:
        '''
        The initial state of the entity.
        '''
        if self._initial_state is None:
            t = cast(type['_CompileState'], self.state_type())
            x = t(self, self.name)
            self._initial_state = cast(_STATE, x)
        return self._initial_state

    @property
    def extensions(self) -> ExtensionsData:
        '''
        The JSON extension data supplied for the entity.
        '''
        return self.initial_state.extensions
    @extensions.setter
    def extensions(self, value: ExtensionsData):
        '''
        Individual extension keys are independent, so assignment merges.
        '''
        self.initial_state.extensions.update(value)

    @property
    def extension_objects(self) -> _Holder['Extension']:
        '''

        A set of supplied extension instances to be compiled
        into this entity.

        This is used to allow extensions to be added to entities
        at a higher level than supplying the JSON data.
        '''
        return self.initial_state.extension_objects

    @property
    def extras(self) -> ExtrasData:
        '''
        The JSON extra data supplied for the entity.
        '''
        return self.initial_state.extras
    @extras.setter
    def extras(self, value: ExtrasData):
        '''
        Extras are unmanaged data, so assignment replaces.
        '''
        self.initial_state.extras = value

    name: str = ''

    @property
    def name_scope(self) -> bool:
        '''
        Whether entity names are scoped within this entity.
        '''
        return bool(self._flags & EntityFlags.NAME_SCOPE)

    @name_scope.setter
    def name_scope(self, value: bool):
        if value:
            self._flags |= EntityFlags.NAME_SCOPE
        else:
            self._flags &= ~EntityFlags.NAME_SCOPE

    @property
    def view_scope(self) -> bool:
        '''
        Whether buffer views are scoped within this entity.
        '''
        return bool(self._flags & EntityFlags.VIEW_SCOPE)

    @view_scope.setter
    def view_scope(self, value: bool):
        if value:
            self._flags |= EntityFlags.VIEW_SCOPE
        else:
            self._flags &= ~EntityFlags.VIEW_SCOPE

    @classmethod
    def state_type(cls) -> type[_STATE]:
        '''
        Create a new compile state for the entity.
        '''
        return cast(type[_STATE], _CompileState) # pragma: no cover

    def __init__(self,
                 name: str='', /,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 extension_objects: Optional[Iterable['Extension']]=None,
                ):
        self.name = name
        self._initial_state = None
        self._flags = EntityFlags.NONE
        if extensions:
            self.extensions = extensions
        if extension_objects:
            self.extension_objects.add_from(extension_objects)
        if extras:
            self.extras = dict(extras)

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
        state = cast(_CompileState, globl.state(cast('Entity', self)))
        progress = getattr(state, phase.name)
        match progress:
            case Progress.IN_PROGRESS:
                raise RuntimeError(f"Recursive compilation detected on {self} in phase {phase.name}")
            case Progress.NONE:
                setattr(state, phase.name, Progress.IN_PROGRESS)
                LOG.debug('Compiling %s in phase %s', self, phase.name)

                def _do_compile():
                    return self._do_compile(globl, phase, cast(_STATE, state))
                match phase:
                    case Phase.COLLECT:
                        def e_collect(ext: 'Extension'):
                            return ext.compile(globl, Phase.COLLECT)

                        globl.extension_objects.add_from(
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
                        assert isinstance(state, _BinaryCompileState)
                        bytelen = cast(_ReturnSizes, _do_compile() or 0)
                        assert bytelen is not None
                        state._len = bytelen
                        LOG.debug('%s has length %s', self, state._len)
                        state.SIZES = bytelen
                        return bytelen
                    case Phase.OFFSETS:
                        assert isinstance(state, _BinaryCompileState)
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
        Compile the extensions for the entity.
        '''
        if phase == Phase.EXTENSIONS:
            return self.compile(globl, phase)
        return None
