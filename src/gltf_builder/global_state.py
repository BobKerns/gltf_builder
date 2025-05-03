'''
global compilation state
'''

from collections.abc import Callable, Iterable
import logging
from typing import Optional, TYPE_CHECKING, Self, cast
from itertools import count
from datetime import datetime
import sys

import pygltflib as gltf
import numpy as np

from gltf_builder.accessors import _Accessor
from gltf_builder.assets import _Asset, __version__
from gltf_builder.attribute_types import BTYPE
from gltf_builder.buffers import _Buffer
from gltf_builder.compiler import (
    _GLTF, _STATE, _Compilable, _CompileState, _GlobalCompileState,
    _Collected, _DoCompileReturn, _Scope,
)
from gltf_builder.core_types import (
    BufferViewTarget, ComponentType, ElementType, IndexSize,
    JsonObject, NPTypes, NameMode, Phase, ScopeName,
)
from gltf_builder.elements import (
    BAccessor, BAsset, BBuffer, BBufferView, BScene, Element,
)
from gltf_builder.nodes import _BNodeContainer
from gltf_builder.protocols import AttributeType
from gltf_builder.global_shared import _GlobalSharedState
from gltf_builder.scenes import scene
from gltf_builder.utils import (
    USER, USERNAME, decode_dtype, std_repr, count_iter,
)
from gltf_builder.log import GLTF_LOG
if TYPE_CHECKING:
    from gltf_builder.builder import Builder
    from gltf_builder.extensions import Extension
    from gltf_builder.treewalker import TreeWalker


LOG = GLTF_LOG.getChild(__name__.split('.')[-1])

_imported: bool = False

class GlobalState(_GlobalCompileState, _BNodeContainer, _GlobalSharedState):
    _scope_name: ScopeName = ScopeName.BUILDER

    _id_counters: dict[str, count]
    _states: dict[int, _CompileState] = {}
    __ordered_views: list[BBufferView]

    __builder: 'Builder'
    @property
    def builder(self) -> 'Builder':
        return self.__builder

    treewalker: Optional['TreeWalker'] = None
    returns: dict[Phase, _DoCompileReturn]


    def state(self, elt: Element[_GLTF, _STATE]) -> _STATE:
        '''
        Get the state for the given element.
        '''
        global ExtensionState, _imported
        if not _imported:
            from gltf_builder.extensions import ExtensionState
            self.__imported = True
        _key = id(elt)
        state = cast(_STATE, self._states.get(_key, None))
        if state is None:
            state_type = elt.state_type()
            state = state_type(elt, self._gen_name(elt))
            if isinstance(state, ExtensionState):
                self.extension_objects.add(state.extension)
            self._states[_key] = state
        return state

    __asset: Optional['BAsset'] = None
    @property
    def asset(self) -> Optional['BAsset']:
        '''
        The asset for the glTF document.
        '''
        return self.__asset

    __scene: Optional['BScene'] = None
    @property
    def scene(self) -> Optional['BScene']:
        '''
        The default scene for the glTF document.
        '''
        return self.__scene

    __buffer: BBuffer|None = None
    @property
    def buffer(self) -> BBuffer:
        '''
        The default buffer for the glTF document.
        '''
        return self.__buffer or self.buffers[0]

    @property
    def index_size(self) -> IndexSize:
        '''
        The size of the index type for the glTF document.
        '''
        return self.builder.index_size

    def get_attribute_type(self, name: str) -> AttributeType:
        '''
        Get the attribute type for the given name.
        '''
        return self.builder.get_attribute_type(name)

    '''
    Global state for the compilation of a glTF document.
    '''
    def __init__(self, builder: 'Builder') -> None:
        _GlobalCompileState.__init__(self, builder, 'GLOBAL')
        _BNodeContainer.__init__(self)
        _GlobalSharedState.__init__(self)
        buffer = (builder.buffers[0]
                  if builder.buffers
                  else _Buffer('main'))
        _Scope.__init__(self, self, buffer)
        self.add(buffer)
        self.buffers.add_from(builder.buffers)
        self.views.add_from(builder.views)
        self.accessors.add_from(builder.accessors)
        self.__builder = builder
        self.__asset = builder.asset
        self.meshes.add_from(builder.meshes)
        self.cameras.add_from(builder.cameras)
        self.images.add_from(builder.images)
        self.materials.add_from(builder.materials)
        self.nodes.add_from(builder.nodes)
        self.samplers.add_from(builder.samplers)
        self.scenes.add_from(builder.scenes)
        self.skins.add_from(builder.skins)
        self.textures.add_from(builder.textures)
        self.extras = builder.extras or {}
        self.extensions = builder.extensions or {}
        self.__scene = builder.scene
        self.extensionsUsed = set(builder.extensionsUsed or ())
        self.extensionsRequired = set(builder.extensionsRequired or ())
        self._id_counters = {}
        self._states = {}
        self._states[id(builder)] = self
        self.returns = {}

    def _get_index_size(self, max_value):
        return self.builder._get_index_size(max_value)


    __names: set[str] = set()

    def _gen_name(self,
                  obj: _Compilable[_GLTF, _STATE], /, *,
                  prefix: str|object='',
                  scope: ScopeName|None=None,
                  index: int|None=None,
                  suffix: str|None=None,
                  ) -> str:
        '''
        Generate a name according to the current name mode policy
        '''
        scope = scope or obj._scope_name
        def get_count(obj: object) -> int:
            tname = type(obj).__name__[1:]
            counters = self._id_counters
            if tname not in counters:
                counters[tname] = count()
            return next(counters[tname])

        def gen(obj: _Compilable[_GLTF, _STATE]) -> str:
            nonlocal prefix, suffix
            name_mode = self.builder.name_policy[scope]
            match obj:
                case Element() if obj.name and name_mode != NameMode.UNIQUE:
                    # Increment the count anyway for stability.
                    # Naming one node should not affect the naming of another.
                    get_count(obj)
                    return obj.name
                case _:
                    if prefix == '':
                        prefix = type(obj).__name__[1:]
                    else:
                        prefix = prefix
                    suffix = suffix or ''
                    if index is not None:
                        suffix = f'{suffix}[{index}]'
                    return f'{prefix}{get_count(obj)}{suffix}'

        def register(name: object|None) -> str:
            match name:
                case str():
                    name = name.strip()
                case Element():
                    name = name.name.strip()
                case _:
                    raise ValueError(f'Invalid name: {name}')
            if not name:
                return ''
            self.__names.add(name)
            return name
        name_mode = self.builder.name_policy[scope]
        match name_mode:
            case NameMode.AUTO:
                return register(gen(obj))
            case NameMode.MANUAL:
                return register(obj)
            case NameMode.UNIQUE:
                name = gen(obj)
                while obj in self.__names:
                    name = gen(obj)
                return register(name)
            case NameMode.NONE:
                return ''
            case _:
                raise ValueError(f'Invalid name mode: {name_mode}') # pragma: no cover

    def _create_accessor(self,
                elementType: ElementType,
                componentType: ComponentType,
                btype: type[BTYPE],
                name: str='',
                normalized: bool=False,
                buffer: Optional['BBuffer']=None,
                count: int=0,
                target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
                ) -> BAccessor[NPTypes, BTYPE]:
        dtype = decode_dtype(elementType, componentType)
        return _Accessor(
            elementType=elementType,
            componentType=componentType,
            btype=btype,
            buffer=buffer or self.buffers[0],
            name=name,
            dtype=dtype,
            count=count,
            normalized=normalized,
            target=target,
        )

    def build(self) -> gltf.GLTF2:
        '''
        Compile the glTF document.
        '''
        LOG.debug('Building glTF document.')
        python = sys.version_info
        if self.asset is None:
            self.__asset = asset = _Asset()
        else:
            asset = self.asset
        extras = asset.extras = asset.extras or {}
        builder_info = cast(JsonObject,
                            extras.get('gltf_builder', {}))
        # Supplied builder_info overrides the default.
        builder_info: JsonObject = {
                'version': __version__,
                'pygltflib': gltf.__version__,
                'numpy': np.__version__,
                'python': {
                    'major': python.major,
                    'minor': python.minor,
                    'micro': python.micro,
                    'releaselevel': python.releaselevel,
                    'serial': python.serial,
                },
                'creation_time': datetime.now().isoformat(),
                **builder_info
            }
        asset.extras = {
            'gltf_builder': builder_info,
            'username': USERNAME,
            'user': USER,
            'date': datetime.now().isoformat(),
            **asset.extras,
        }
        # Filter out empty values.
        asset.extras = {
            key: value
            for key, value in asset.extras.items()
            if value is not None
        }
        # Add a default scene if none provided.
        if len(self.scenes) == 0:
            self.add(scene('DEFAULT',
                                *(n for n in self.nodes if n.root)))
        self._states = {}
        for phase in Phase:
            if phase != Phase.BUILD:
               self.do_compile(phase)

        def build_list(elt: Iterable[Element[_GLTF, _STATE]]) -> list[_GLTF]:
            return [
                v.compile(self, Phase.BUILD)
                for v in elt
            ]
        nodes = build_list(self.nodes)
        cameras = build_list(self.cameras)
        meshes = build_list(self.meshes)
        materials = build_list(self.materials)
        samplers = build_list(self.samplers)
        skins = build_list(self.skins)
        textures = build_list(self.textures)
        images = build_list(self.images)
        accessors = build_list(a for a in self.accessors if a.count > 0)
        bufferViews = build_list(self.__ordered_views)
        _asset = asset.compile(self, Phase.BUILD)
        def check_buffer(b: BBuffer|None) -> bool:
            if b is None:
                return False
            s = self.state(b)
            return len(s.blob) > 0
        buffers = build_list(
            b
            for b in self.buffers
            if check_buffer(b)
        )
        scenes = build_list(self.scenes)
        self.__scene = self.scene or (self.scenes[0] if self.scenes else None)
        if self.scene is None:
            scene_info = {}
        else:
            scene_state = self.state(self.scene)
            scene_info = dict(scene=scene_state.index)
        g = gltf.GLTF2(
            asset=_asset,
            nodes=nodes,
            cameras=cameras,
            meshes=meshes,
            materials=materials,
            textures=textures,
            images=images,
            samplers=samplers,
            skins=skins,
            accessors=accessors,
            bufferViews=bufferViews,
            buffers=buffers,
            scenes=scenes,
            extras=self.extras,
            extensions=self.extensions,
            animations=[],
            extensionsUsed=list(self.extensionsUsed),
            extensionsRequired=list(self.extensionsRequired),
            **scene_info,
        )
        if len(self.buffers) == 1 :
            state = self.state(self.buffers[0])
            data = state.blob
        else:
            raise ValueError("Only one buffer is supported by pygltflib.")
        g.set_binary_blob(data) # type: ignore
        return g

    def do_compile(self, phase: Phase):
        def _do_compile(n):
            return n.compile(self, phase)
        def _do_compile_n(*n: Iterable[Element]):
            for g in n:
                for e in g:
                    e.compile(self, phase)

        match phase:
            case Phase.COLLECT:
                if self.scene:
                    self.add(self.scene)
                collected = [
                    *(_do_compile(n) for n in self.scenes),
                    *(_do_compile(n) for n in self.skins),
                    *(_do_compile(n) for n in self.nodes),
                    *(_do_compile(c) for c in self.cameras),
                    *(_do_compile(m) for m in self.meshes),
                    *(_do_compile(m) for m in self.materials),
                    *(_do_compile(s) for s in self.samplers),
                    *(_do_compile(t) for t in self.textures),
                    *(_do_compile(i) for i in self.images),
                    *(_do_compile(a) for a in self.accessors),
                    *(_do_compile(v) for v in self.views),
                    *(_do_compile(b) for b in self.buffers),
                ]
                ordered = sorted(list(self.views),
                                                key=lambda v: v.byteStride or 4,
                                                reverse=True)
                self.__ordered_views = ordered
                LOG.debug('Collected %s items.', len(collected))
                def log_collected(collected: Iterable[_Collected], indent: int = 0):
                    for item, children in collected:
                        LOG.debug('. ' * indent + str(item))
                        for child in children:
                            LOG.debug('. ' * (indent +1) + '=> ' + str(child))
                        log_collected(children, indent + 2)
                if LOG.isEnabledFor(logging.DEBUG):
                    log_collected(collected)
            case Phase.ENUMERATE:
                def assign_index(items: Iterable[Element]):
                    for i, n in enumerate(items):
                        s = self.state(n)
                        s.index = i
                assign_index(self.buffers)
                assign_index(self.__ordered_views)
                assign_index(self.accessors)
                assign_index(self.images)
                assign_index(self.cameras)
                assign_index(self.materials)
                assign_index(self.meshes)
                assign_index(self.scenes)
                assign_index(self.samplers)
                assign_index(self.skins)
                assign_index(self.textures)
                assign_index(self.nodes)
            case Phase.SIZES:
                _do_compile_n(self.accessors, self.views, self.buffers)
            case Phase.OFFSETS:
                _do_compile_n(self.buffers, self.views, self.accessors)
            case Phase.EXTENSIONS:
                actual = {
                            s
                            for elt in self._elements()
                            for s in cast(set[str]|None, _do_compile(elt)) or ()
                        }
                specified = {
                    *self.extensionsUsed,
                    *self.extensionsRequired
                }
                unused = specified - actual
                if unused:
                    LOG.warning(f'Unused extensions: {unused}')
                self.extensionsUsed |= specified | actual
            case _:
                _do_compile_n(
                    self.scenes,
                    self.skins,
                    self.nodes,
                    self.cameras,
                    self.meshes,
                    self.materials,
                    self.textures,
                    self.images,
                    self.samplers,
                    self.accessors,
                    self.__ordered_views if self.views else (),
                    self.buffers,
                )

    # Currently a placeholder.
    def compile_extensions_(self, contin: Callable[[Element], set['Extension']|None]):
        def _do_compile(elt: Element[_GLTF, _STATE]) -> set[Extension]|None:
            return elt.compile(self, Phase.EXTENSIONS)
        return {
            s
            for elt in self._elements()
            for s in cast(set[Extension]|None, _do_compile(elt)) or ()
        }

    # Currently a placeholder.
    def _do_compile(self, globl: Self, phase: Phase, state: Self):
        '''
        Compile the given element.
        '''
        match phase:
            case Phase.EXTENSIONS:
                def do_extensions(elt: Element[_GLTF, _STATE]) -> set[Extension]|None:
                    return elt.compile(self, Phase.EXTENSIONS)
                return self.compile_extensions_(do_extensions)
            case _: pass
        return None

    def _elements(self) -> Iterable[Element]:
        '''
        Get all the elements in the builder.
        '''
        yield from self.nodes
        yield from self.meshes
        yield from self.cameras
        yield from self.materials
        yield from self.textures
        yield from self.images
        yield from self.samplers
        yield from self.skins
        yield from self.scenes
        yield from self.accessors
        yield from self.views
        yield from self.buffers


    def __repr__(self):
        return std_repr(self, (
            ('cameras', count_iter(self.cameras)),
            ('meshes', count_iter(self.meshes)),
            ('images', count_iter(self.images)),
            ('materials', count_iter(self.materials)),
            ('nodes', count_iter(self.nodes)),
            ('samplers', count_iter(self.samplers)),
            ('skins', count_iter(self.skins)),
            ('scenes', count_iter(self.scenes)),
            ('textures', count_iter(self.textures)),
            ('accessors', count_iter(self.accessors)),
            ('views', count_iter(self.views)),
            ('extensions', count_iter(self.extensions)),
            ('buffers', count_iter(self.buffers)),
            'index_size',
        ))