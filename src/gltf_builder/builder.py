'''
The initial objedt that collects the geometry info and compiles it into
a glTF object.
'''

import sys
from collections.abc import Iterable, Mapping
from typing import Literal, Optional, cast
from itertools import count
from datetime import datetime
import logging
from pathlib import Path
import re
from warnings import warn

import pygltflib as gltf
import numpy as np

from gltf_builder.attribute_types import (
    BTYPE, Color, Joint, Point, Tangent,
    UvPoint, Vector3, Vector3Spec, Weight,
    color, point, tangent, uv, vector3,
)
from gltf_builder.core_types import (
     BufferViewTarget, ImageType, JsonObject,
     NPTypes, NameMode, NamePolicy, Phase,
     ElementType, ComponentType, ScopeName, NameMode,
)
from gltf_builder.assets import BAsset, __version__
from gltf_builder.holders import _Holder
from gltf_builder.buffers import _Buffer
from gltf_builder.matrix import Matrix4, Matrix4Spec
from gltf_builder.quaternions import QuaternionSpec
from gltf_builder.accessors import _Accessor
from gltf_builder.meshes import _Mesh
from gltf_builder.nodes import _BNodeContainer
from gltf_builder.images import _Image
from gltf_builder.scenes import scene
from gltf_builder.protocols import _AttributeParser, AttributeType, _BuilderProtocol
from gltf_builder.elements import (
     BAccessor, BBuffer, BBufferView, BCamera, BImage, BMaterial,
     BMesh, BNode, BPrimitive, BSampler, BScene, BSkin, BTexture,
     Element, T,
)
from gltf_builder.compile import _Compileable, _Collected
from gltf_builder.utils import USERNAME, USER, decode_dtype
from gltf_builder.log import GLTF_LOG


LOG = GLTF_LOG.getChild(Path(__file__).stem)

DEFAULT_NAME_MODE = NameMode.AUTO
DEFAULT_NAME_POLICY: NamePolicy = {
    ScopeName.NODE: NameMode.AUTO,
    ScopeName.MESH: NameMode.AUTO,
    ScopeName.PRIMITIVE: NameMode.AUTO,
    ScopeName.ACCESSOR: NameMode.NONE,
    ScopeName.ACCESSOR_INDEX: NameMode.NONE,
    ScopeName.BUFFER: NameMode.NONE,
    ScopeName.BUFFER_VIEW: NameMode.NONE,
    ScopeName.BUILDER: NameMode.NONE,
    ScopeName.IMAGE: NameMode.AUTO,
    ScopeName.MATERIAL: NameMode.AUTO,
    ScopeName.TEXTURE: NameMode.AUTO,
    ScopeName.SAMPLER: NameMode.AUTO,
    ScopeName.CAMERA: NameMode.AUTO,
    ScopeName.SKIN: NameMode.AUTO,
    ScopeName.SCENE: NameMode.AUTO,
}
'''
Default naming mode for each scope.
'''

_RE_ATTRIB_NAME = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)(?:_\d+)$')


class Builder(_BNodeContainer, _BuilderProtocol):
    _scope_name = ScopeName.BUILDER
    _id_counters: dict[str, count]
    name: str = ''
    __ordered_views: list[BBufferView] = []

    @property
    def builder(self) -> _BuilderProtocol:
        return self
    
    @builder.setter
    def builder(self, builder: _BuilderProtocol):
        raise ValueError('Builder is already attached to itself')
    
    '''
    The main object that collects all the geometry info and compiles it into a glTF object.
    '''
    def __init__(self, /,
                asset: gltf.Asset=BAsset(),
                cameras: Iterable[BCamera]=(),
                meshes: Iterable[BMesh]=(),
                images: Iterable[BImage]=(),
                materials: Iterable[BMaterial]=(),
                nodes: Iterable[BNode] =(),
                samplers: Iterable[BSampler]=(),
                skins: Iterable[BSkin]=(),
                scenes: Iterable[BScene]=(),
                textures: Iterable[BTexture]=(),
                buffers: Iterable[_Buffer]=(),
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                scene: Optional[BScene]=None,
                index_size: int=32,
                name_policy: Mapping[ScopeName, NameMode]|None = None,
                extensionsUsed: Optional[list[str]]=None,
                extensionsRequired: Optional[list[str]]=None,
        ):
        name_policy = name_policy or {}
        self.name_policy = {
            scope: name_policy.get(scope, DEFAULT_NAME_POLICY[scope])
            for scope in ScopeName
        }
        if not buffers:
            buffers = [_Buffer(self, 'main')]
        else:
            buffers = list(buffers)
        super().__init__(buffer=buffers[0], children=nodes)
        self.asset = asset
        self.meshes = _Holder(BMesh, *meshes)
        self.cameras = _Holder(BCamera, *cameras)
        self._buffers = _Holder(BBuffer, *buffers)
        self._views = _Holder(BBufferView)
        self._accessors = _Holder(BAccessor)
        self.images = _Holder(BImage, *images)
        self.materials = _Holder(BMaterial, *materials)
        self.samplers = _Holder(BSampler, *samplers)
        self.scenes = _Holder(BScene, *scenes)
        self.skins = _Holder(BSkin, *skins)
        self.textures = _Holder(BTexture, *textures)
        self.index_size = index_size
        self.extras = extras or {}
        self.extensions = extensions or {}
        self.scene = scene
        self.extensionsUsed = list(extensionsUsed or ())
        self.extensionsRequired = list(extensionsRequired or ())
        self.attr_type_map = {}
        self.define_attrib('POSITION', gltf.VEC3, gltf.FLOAT, Point, point)
        self.define_attrib('NORMAL', gltf.VEC3, gltf.FLOAT, Vector3, vector3)
        self.define_attrib('COLOR', gltf.VEC4, gltf.FLOAT, Color, color)
        self.define_attrib('TEXCOORD', gltf.VEC2, gltf.FLOAT, UvPoint, uv)
        self.define_attrib('TANGENT', gltf.VEC4, gltf.FLOAT, Tangent, tangent)
        self.define_attrib('JOINTS', gltf.VEC4, gltf.UNSIGNED_SHORT, Joint)
        self.define_attrib('WEIGHTS', gltf.VEC4, gltf.FLOAT, Weight)
        self._id_counters = {}
    
    def create_mesh(self,
                name: str='',
                primitives: Optional[Iterable[BPrimitive]]=None,
                weights: Optional[Iterable[float]]=None,
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                detached: bool=False,
                ):
        mesh = _Mesh(name,
                     primitives=primitives or (),
                     weights=weights or (),
                     extras=extras,
                     extensions=extensions,
                     detached=detached,
        )
        return mesh
    
    def _elements(self) -> Iterable[Element[gltf.Property]]:
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
        yield from self._accessors
        yield from self._views
        yield from self._buffers

    def compile(self, phase: Phase):
        def _do_compile(n):
            return n.compile(self, self, phase)
        def _do_compile_n(*n: Iterable[Element[gltf.Property]]):
            for g in n:
                for e in g:
                    e.compile(self, self, phase)
                    
        match phase:
            case Phase.COLLECT:
                if self.scene:
                    self.scenes.add(self.scene)
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
                    *(_do_compile(a) for a in self._accessors),
                    *(_do_compile(v) for v in self._views),
                    *(_do_compile(b) for b in self._buffers),
                ]
                ordered = sorted(list(self._views),
                                                key=lambda v: v.byteStride or 4,
                                                reverse=True)
                self.__ordered_views = ordered
                LOG.debug('Collected %s items.', len(collected))
                def log_collcted(collected: Iterable[_Collected], indent: int = 0):
                    for item, children in collected:
                        LOG.debug('. ' * indent + str(item))
                        for child in children:
                            LOG.debug('. ' * (indent +1) + '=> ' + str(child))
                        log_collcted(children, indent + 2)
                if LOG.isEnabledFor(logging.DEBUG):
                    log_collcted(collected)
            case Phase.ENUMERATE:
                def assign_index(items: Iterable[_Compileable[gltf.Property]]):
                    for i, n in enumerate(items):
                        n._index = i
                assign_index(self._buffers)
                assign_index(self.__ordered_views)
                assign_index(self._accessors)
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
                _do_compile_n(self.nodes, self._buffers)
            case Phase.OFFSETS:
                _do_compile_n(self._buffers, self.nodes)
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
                    warn(f'Unused extensions: {unused}')
                self.extensionsUsed = list(specified | actual)
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
                    self._accessors,
                    self.__ordered_views if self._views else (),
                    self._buffers,
                )
    
    def build(self, /,
            name_mode: Optional[NameMode]=None,
            index_size: Optional[int]=None,
        ) -> gltf.GLTF2:
        if name_mode is not None:
            self.name_mode = name_mode
        if index_size is not None:
            self.index_size = index_size
        def flatten(node: BNode) -> Iterable[BNode]:
            yield node
            for n in node.children:
                yield from flatten(n)
        
        nodes = list({
            i
            for n in self.nodes
            for i in flatten(n)
        })
        # Add all the child nodes.
        self.nodes.add(*(n for n in nodes if not n.root))
        python = sys.version_info
        self.asset.extras = self.asset.extras or {}
        builder_info = self.asset.extras.get('gltf-builder', {})
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
        self.asset.extras = {
            'gltf-builder': builder_info,
                'username': USERNAME,
                'user': USER,
                'date': datetime.now().isoformat(),
            **self.asset.extras,
        }
        # Filter out empty values.
        self.asset.extras = {
            key: value
            for key, value in self.asset.extras.items()
            if value is not None
        }
        # Add a default scene if none provided.
        if len(self.scenes) == 0:
            self.scenes.add(scene('DEFAULT', *(n for n in self.nodes if n.root)))
        for phase in Phase:
            if phase != Phase.BUILD:
               self.compile(phase)
        
        def build_list(l: Iterable[Element[T]]) -> list[T]:
            return [
                v.compile(self, self, Phase.BUILD)
                for v in l
            ]
        nodes = build_list(self.nodes)
        cameras = build_list(self.cameras)
        meshes = build_list(self.meshes)
        materials = build_list(self.materials)
        samplers = build_list(self.samplers)
        skins = build_list(self.skins)
        textures = build_list(self.textures)
        images = build_list(self.images)
        accessors = build_list(a for a in self._accessors if a.count > 0)
        bufferViews = build_list(self.__ordered_views)
        buffers = build_list(b for b in self._buffers if len(b.blob) > 0)
        scenes = build_list(self.scenes)
        _scene = self.scene or self.scenes[0]
        g = gltf.GLTF2(
            asset = self.asset,
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
            scene=_scene._index,
            extras=self.extras,
            extensions=self.extensions,
            animations=[],
            extensionsUsed=self.extensionsUsed,
            extensionsRequired=self.extensionsRequired,
        )
        if len(self._buffers) == 1 :
            data = self._buffers[0].blob
        else:
            raise ValueError("Only one buffer is supported by pygltfllib.")
        g.set_binary_blob(data) # type: ignore
        return g
    
    def define_attrib(self,
                      name: str,
                      elementType: ElementType,
                      componentType: ComponentType,
                      type: type[BTYPE],
                      parser: _AttributeParser[BTYPE]|None=None,
                ):
        '''
        Define the type of an attribute.
        - TANGENT: VEC4/FLOAT/Tangent
        - TEXCOORD_0: VEC2/FLOAT
        - TEXCOORD_1: VEC2/FLOAT
        - COLOR_0: VEC4/FLOAT
        - JOINTS_0: VEC4/UNSIGNED_SHORT
        - WEIGHTS_0: VEC4/FLOAT
        '''
        self.attr_type_map[name] = AttributeType(name, elementType, componentType, type, parser)

    def get_attribute_type(self, name: str) -> AttributeType:
        '''
        Get the type information for an attribute by name.

        If the attribute is not defined, but ends in _<digits>,
        the suffix is stripped and the attribute is looked up again.
        If the attribute is still not found, a ValueError is raised.
        '''
        attr = self.attr_type_map.get(name)
        if attr:
            return attr
        m = _RE_ATTRIB_NAME.match(name)
        if m:
            attr = self.attr_type_map.get(m.group(1))
        if attr is None:
            raise ValueError(f'Attribute {name} is not defined.')
        self.attr_type_map[name] = attr
        return attr

    def _get_index_size(self, max_value: int) -> ComponentType|Literal[-1]:
        '''
        Get the index size based on the configured size or the maximum value.
        '''
        match self.index_size:
            case size if size > 16 and size <= 32:
                if max_value < 4294967295:
                    return ComponentType.UNSIGNED_INT
                raise ValueError("Index size is too large.")
            case size if size > 8 and size <= 16:
                if max_value < 65535:
                    return ComponentType.UNSIGNED_SHORT
                return ComponentType.UNSIGNED_INT
            case size if size > 0 and size <= 8:
                if max_value < 255:
                    return ComponentType.UNSIGNED_BYTE
                return ComponentType.UNSIGNED_SHORT
            case 0:
                if max_value < 0:
                    raise ValueError("Index size is negative.")
                if max_value < 255:
                    return ComponentType.UNSIGNED_BYTE
                if max_value < 65535:
                    return ComponentType.UNSIGNED_SHORT
                if max_value < 4294967295:
                    return ComponentType.UNSIGNED_INT
                # Unlikely!
                raise ValueError("Index size is too large.")
            case -1:
                return -1
            case _:
                raise ValueError(f'Invalid index size: {self.index_size}')

    __names: set[str] = set()

    def _gen_name(self,
                  obj: _Compileable[gltf.Property], /, *,
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
        
        def gen(obj: _Compileable[gltf.Property]) -> str:
            nonlocal prefix, suffix
            name_mode = self.name_policy[scope]
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
        match self.name_policy[scope]:
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
                raise ValueError(f'Invalid name mode: {self.name_mode}') # pragma: no cover

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
                buffer=buffer or self._buffers[0],
                name=name,
                dtype=dtype,
                count=count,
                normalized=normalized,
                target=target,
            )
    
    def create_image(self,
                imageType: ImageType,
                name: str='',
                /, *,
                blob: Optional[bytes]=None,
                uri: Optional[str|Path]=None,
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
            ) -> BImage:
        '''
        Implementation of `BImage`.
        '''
        return _Image(
            name=name,
            blob=blob,
            uri=uri,
            imageType=imageType,
            extras=extras,
            extensions=extensions,
        )
    
    def instantiate(self, node_or_mesh: 'BNode|BMesh',
                    name: str='', /,
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4Spec]=None,
                    extras: Optional[JsonObject]=None,
                    extensions: Optional[JsonObject]=None,
                    detached: bool=False,
                ) -> 'BNode':
        node = super().instantiate(node_or_mesh,
                    name,
                    translation=translation,
                    rotation=rotation,
                    scale=scale,
                    matrix=matrix,
                    extras=extras,
                    extensions=extensions,
                )
        if not detached:
            self.nodes.add(node)
        return node

