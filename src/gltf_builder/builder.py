'''
The initial objedt that collects the geometry info and compiles it into
a glTF object.
'''

import sys
from collections.abc import Iterable
from typing import Literal, Optional
from itertools import count
from datetime import datetime
import logging
from pathlib import Path

import pygltflib as gltf
import numpy as np

from gltf_builder.attribute_types import (
    ColorSpec, JointSpec, TangentSpec, UvSpec, Vector3Spec, WeightSpec,
    BType, BTypeType,
)
from gltf_builder.core_types import (
     BufferViewTarget, JsonObject, NPTypes, NameMode, Phase,
     ElementType, ComponentType, Scalar,
)
from gltf_builder.asset import BAsset, __version__
from gltf_builder.holder import _Holder
from gltf_builder.buffer import _Buffer
from gltf_builder.view import _BufferView
from gltf_builder.accessor import _Accessor
from gltf_builder.mesh import _Mesh
from gltf_builder.node import _Node, _BNodeContainer
from gltf_builder.protocols import _AttributeInfo, _BuilderProtocol
from gltf_builder.element import (
     BTYPE, BAccessor, BBuffer, BBufferView, BMesh, BNode, BPrimitive, Element,
)
from gltf_builder.compile import _Compileable, _Collected
from gltf_builder.utils import USERNAME, USER, decode_dtype
from gltf_builder.log import GLTF_LOG


LOG = GLTF_LOG.getChild(Path(__file__).stem)

class Builder(_BNodeContainer, _BuilderProtocol):
    _id_counters: dict[str, count] # type: ignore
    name: str = ''
    __ordered_views: list[BBufferView] = []

    @property
    def builder(self) -> _BuilderProtocol:
        return self
    
    '''
    The main object that collects all the geometry info and compiles it into a glTF object.
    '''
    def __init__(self, /,
                asset: gltf.Asset= BAsset(),
                meshes: Iterable[_Mesh]=(),
                nodes: Iterable[_Node] = (),
                buffers: Iterable[_Buffer]=(),
                views: Iterable[_BufferView]=(),
                accessors: Iterable[_Accessor[NPTypes, BType]]=(),
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                index_size: int=32,
                name_mode: NameMode=NameMode.AUTO,
        ):
        if not buffers:
            buffers = [_Buffer(self, 'main')]
        else:
            buffers = list(buffers)
        super().__init__(buffer=buffers[0], children=nodes)
        self.asset = asset
        self.meshes = _Holder(BMesh, *meshes)
        self._buffers = _Holder(BBuffer, *buffers)
        self._views = _Holder(BBufferView, *views)
        self._accessors = _Holder(BAccessor, *accessors)
        self.index_size = index_size
        self.extras = extras or {}
        self.extensions = extensions or {}
        self.attr_type_map ={
            'TANGENT': _AttributeInfo(gltf.VEC4, gltf.FLOAT, type[TangentSpec]),
            'POSITION': _AttributeInfo(gltf.VEC3, gltf.FLOAT, type[Vector3Spec]),
            'NORMAL': _AttributeInfo(gltf.VEC3, gltf.FLOAT, type[Vector3Spec]),
            'COLOR': _AttributeInfo(gltf.VEC4, gltf.FLOAT, type[ColorSpec]),
            'TEXCOORD_0': _AttributeInfo(gltf.VEC2, gltf.FLOAT, type[UvSpec]),
            'TEXCOORD_1': _AttributeInfo(gltf.VEC2, gltf.FLOAT, type[UvSpec]),
            'COLOR_0': _AttributeInfo(gltf.VEC4, gltf.FLOAT, type[ColorSpec]),
            'JOINTS_0': _AttributeInfo(gltf.VEC4, gltf.UNSIGNED_SHORT, type[JointSpec]),
            'WEIGHTS_0': _AttributeInfo(gltf.VEC4, gltf.FLOAT, type[WeightSpec]),
            '__DEFAULT__': _AttributeInfo(gltf.SCALAR, gltf.FLOAT, type[Scalar]),
        }
        self._id_counters = {}
        self.name_mode = name_mode
    
    def create_mesh(self,
                name: str='',
                primitives: Iterable[BPrimitive]=(),
                weights: Iterable[float]|None=(),
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                detached: bool=False,
                ):
        mesh = _Mesh(name=name,
                     primitives=primitives,
                     weights=weights,
                     extras=extras,
                     extensions=extensions,
                     detached=detached,
        )
        return mesh

    def compile(self, phase: Phase):
        match phase:
            case Phase.ENUMERATE:
                def assign_index(items: Iterable[_Compileable[gltf.Property]]):
                    for i, n in enumerate(items):
                        n._index = i
                assign_index(self._buffers)
                assign_index(self.__ordered_views)
                assign_index(self._accessors)
                assign_index(self.meshes)
                assign_index(self.nodes)
            case _: pass

        match phase:
            case Phase.COLLECT:
                collected = [
                    *(n.compile(self, self, phase) for n in self.nodes),
                    *(m.compile(self, self, phase) for m in self.meshes),
                    *(a.compile(self, self, phase) for a in self._accessors),
                    *(v.compile(self, self, phase) for v in self._views),
                    *(b.compile(self, self, phase) for b in self._buffers),
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
            case Phase.SIZES:
                for n in self.nodes:
                    n.compile(self, self, phase)
                for v in self._buffers:
                    v.compile(self, self, phase)
            case Phase.OFFSETS:
                for b in self._buffers:
                    b.compile(self, self, phase)
                for n in self.nodes:
                    n.compile(self, self, phase)
            case _:
                for n in self.nodes:
                    n.compile(self, self, phase)
                for m in self.meshes:
                    m.compile(self, self, phase)
                for a in self._accessors:
                    a.compile(self, self, phase)
                for v in self.__ordered_views if self._views else ():
                    v.compile(self, self, phase)
                for b in self._buffers:
                    b.compile(self, self, phase)
    
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
        for phase in Phase:
            if phase != Phase.BUILD:
               self.compile(phase)
        g = gltf.GLTF2(
            asset = self.asset,
            nodes=[
                v
                for v in (
                    n.compile(self, self, Phase.BUILD)
                    for n in nodes
                )
            ],
            meshes=[
                m.compile(self, self, Phase.BUILD)
                for m in self.meshes
            ],
            accessors=[
                a.compile(self, self, Phase.BUILD)
                for a in self._accessors
                if a.count > 0
            ],
            # Sort the buffer views by alignment.
            bufferViews=[
                v.compile(self, self, Phase.BUILD)
                for v in self.__ordered_views
            ],
            buffers=[
                b.compile(self, self, Phase.BUILD)
                for b in self._buffers
                if len(b.blob) > 0
            ],
            scene=0,
            scenes=[
                gltf.Scene(
                    name=self.name,
                    nodes=[
                        n._index
                        for n in self.nodes
                        if n.root
                    ]
                )
            ]
        )
        if len(self._buffers) == 1 :
            data = self._buffers[0].blob
        else:
            raise ValueError("Only one buffer is supported by pygltfllib.")
        g.set_binary_blob(data) # type: ignore
        return g
    
    def define_attrib(self,
                      name: str,
                      type: ElementType,
                      componentType: ComponentType,
                      btype: BTypeType,
                ):
        '''
        Define the type of an attribute. The default is VEC3/FLOAT, except for the following:
        - TANGENT: VEC4/FLOAT
        - TEXCOORD_0: VEC2/FLOAT
        - TEXCOORD_1: VEC2/FLOAT
        - COLOR_0: VEC4/FLOAT
        - JOINTS_0: VEC4/UNSIGNED_SHORT
        - WEIGHTS_0: VEC4/FLOAT
        '''
        self.attr_type_map[name] = _AttributeInfo(type, componentType, btype)

    def get_attrib_info(self, name: str) -> _AttributeInfo:
        return self.attr_type_map.get(name) or self.attr_type_map['__DEFAULT__']

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
                  obj: str|_Compileable[gltf.Property]|None,
                  gen_prefix: str|object='',
                  ) -> str:
        '''
        Generate a name according to the current name mode policy
        '''
        def get_count(obj: object) -> int:
            tname = type(obj).__name__[1:]
            counters = self._id_counters
            if tname not in counters:
                counters[tname] = count()
            return next(counters[tname])
            
        def gen(obj: str|_Compileable[gltf.Property]|None) -> str:
            match obj:
                case str():
                    return obj
                case Element() if obj.name and self.name_mode != NameMode.UNIQUE:
                    # Increment the count anyway for stability.
                    # Naming one node should not affect the naming of another.
                    get_count(obj)
                    return obj.name
                case _:
                    if gen_prefix == '':
                        prefix = type(obj).__name__[1:]
                    else:
                        prefix = gen_prefix
                    return f'{prefix}{get_count(obj)}'
        
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
        match self.name_mode:
            case NameMode.AUTO:
                return register(gen(obj))
            case NameMode.MANUAL:
                return register(obj)
            case NameMode.UNIQUE:
                while obj in self.__names:
                    obj = gen(obj)
                return register(obj)
            case NameMode.NONE:
                return ''
            case _:
                raise ValueError(f'Invalid name mode: {self.name_mode}')

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