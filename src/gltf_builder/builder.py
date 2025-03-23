'''
The initial objedt that collects the geometry info and compiles it into
a glTF object.
'''

from collections.abc import Iterable, Mapping
from typing import Optional, Any
from itertools import count
from datetime import datetime
import os
import sys
import pwd
import ctypes
import ctypes.wintypes
import subprocess
import getpass
import logging
from pathlib import Path

import pygltflib as gltf
import numpy as np

from gltf_builder.asset import BAsset, __version__
from gltf_builder.holder import Holder
from gltf_builder.buffer import _Buffer
from gltf_builder.view import _BufferView
from gltf_builder.accessor import _Accessor
from gltf_builder.mesh import _Mesh
from gltf_builder.node import _Node, BNodeContainer
from gltf_builder.element import (
    EMPTY_MAP, BBuffer, BufferViewTarget, BPrimitive, Collected, Element,
    BuilderProtocol, ElementType, ComponentType, NameMode, Phase,
    Compileable, GLTF_LOG,
)


LOG = GLTF_LOG.getChild(Path(__file__).stem)

class Builder(BNodeContainer, BuilderProtocol):
    id_counters: dict[str, count]
    name: str = ''
    __ordered_views: list[_BufferView]
    '''
    The main object that collects all the geometry info and compiles it into a glTF object.
    '''
    def __init__(self, /,
                asset: gltf.Asset= BAsset(),
                meshes: Iterable[_Mesh]=(),
                nodes: Iterable[_Node] = (),
                buffers: Iterable[_Buffer]=(),
                views: Iterable[_BufferView]=(),
                accessors: Iterable[_Accessor]=(),
                extras: Mapping[str, Any]=EMPTY_MAP,
                extensions: Mapping[str, Any]=EMPTY_MAP,
                index_size: int=32,
                name_mode: NameMode=NameMode.AUTO,
        ):
        super().__init__(builder=self, children=nodes)
        self.asset = asset
        self.meshes = Holder(*meshes)
        self.nodes = Holder(*nodes)
        if not buffers:
            buffers = [_Buffer('main')]
        self._buffers = Holder(*buffers)
        self._views = Holder(*views)
        self._accessors = Holder(*accessors)
        self.index_size = index_size
        self.extras = dict(extras)
        self.extensions = dict(extensions)
        self.attr_type_map ={
            'TANGENT': (gltf.VEC4, gltf.FLOAT),
            'TEXCOORD_0': (gltf.VEC2, gltf.FLOAT),
            'TEXCOORD_1': (gltf.VEC2, gltf.FLOAT),
            'COLOR_0': (gltf.VEC4, gltf.FLOAT),
            'JOINTS_0': (gltf.VEC4, gltf.UNSIGNED_SHORT),
            'WEIGHTS_0': (gltf.VEC4, gltf.FLOAT),
            '__DEFAULT__': (gltf.VEC3, gltf.FLOAT),
        }
        self.id_counters = {}
        self.name_mode = name_mode
    
    def create_mesh(self,
                name: str='',
                primitives: Iterable[BPrimitive]=(),
                weights: Iterable[float]|None=(),
                extras: Mapping[str, Any] = EMPTY_MAP,
                extensions: Mapping[str, Any] = EMPTY_MAP,
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
    
    def _add_buffer(self,
                   name: str='') -> _Buffer:
        if len(self._buffers) > 0:
            raise ValueError("Only one buffer is supported by pygltflib.")
        buffer = _Buffer(name=name, index=len(self._buffers))
        self._buffers.add(buffer)
        return buffer
        
    def _add_view(self,
                 name: str='',
                 buffer: Optional[BBuffer]=None,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
            ) -> _BufferView:
        buffer = buffer or self._buffers[0]
        view = _BufferView(name=name, buffer=buffer, target=target)
        self._views.add(view)
        return view
    
    def _get_view(self, name: str,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
       ) -> _BufferView:
        if name in self._views:
            return self._views[name]
        return self._add_view(name=name, target=target)
    
    def compile(self, phase: Phase):
        match phase:
            case Phase.ENUMERATE:
                def assign_index(items: list[Compileable]):
                    for i, n in enumerate(items):
                        n.index = i
                assign_index(self._buffers)
                assign_index(self.__ordered_views)
                assign_index(self._accessors)
                assign_index(self.meshes)
                assign_index(self.nodes)

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
                def log_collcted(collected: list[Collected], indent: int = 0):
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
        def flatten(node: _Node) -> Iterable[_Node]:
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
        builder_info = {
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
                if v is not None
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
                {'name': 'main',
                 'nodes': [
                     n.index
                     for n in self.nodes
                     if n.root
                 ]}
            ]
        )
        if len(self._buffers) == 1 :
            data = self._buffers[0].blob
        else:
            raise ValueError("Only one buffer is supported by pygltfllib.")
        g.set_binary_blob(data)
        return g
    
    def define_attrib(self,
                      name: str,
                      type: ElementType,
                      componentType: ComponentType,
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
        self.attr_type_map[name] = (type, componentType)

    def get_attrib_info(self, name: str) -> tuple[ElementType, ComponentType]:
        return self.attr_type_map.get(name) or self.attr_type_map['__DEFAULT__']

    def _get_index_size(self, max_value: int) -> int:
        '''
        Get the index size based on the configured size or the maximum value.
        '''
        match self.index_size:
            case size if size > 16 and size <= 32:
                if max_value < 4294967295:
                    return gltf.UNSIGNED_INT
            case size if size > 8 and size <= 16:
                if max_value < 65535:
                    return gltf.UNSIGNED_SHORT
            case size if size > 0 and size <= 8:
                if max_value < 255:
                    return gltf.UNSIGNED_BYTE
            case 0:
                if max_value < 0:
                    raise ValueError("Index size is negative.")
                if max_value < 255:
                    return gltf.UNSIGNED_BYTE
                if max_value < 65535:
                    return gltf.UNSIGNED_SHORT
                if max_value < 4294967295:
                    return gltf.UNSIGNED_INT
                # Unlikely!
                raise ValueError("Index size is too large.")
            case -1:
                return -1
            case _:
                raise ValueError(f'Invalid index size: {self.index_size}')

    __names: set[str] = set()

    def _gen_name(self, obj: Element[Any]|str) -> str:
        '''
        Generate a name according to the current name mode policy
        '''
        def get_count(obj) -> int:
            tname = type(obj).__name__[1:]
            counters = self.id_counters
            if tname not in counters:
                counters[tname] = count()
            return next(counters[tname])
            
        def gen():
            if obj and isinstance(obj, str):
                return obj
            if obj.name and self.name_mode != NameMode.UNIQUE:
                # Increment the count anyway for stability.
                # Naming one node should not affect the naming of another.
                get_count(obj)
                return obj.name
            return f'{type(obj).__name__[1:]}{get_count(obj)}'
        
        def register(name: str|None) -> str|None:
            if not name:
                return None
            self.__names.add(name)
            return name
        match self.name_mode:
            case NameMode.AUTO:
                return register(gen())
            case NameMode.MANUAL:
                return register(obj.name or None)
            case NameMode.UNIQUE:
                name = obj.name
                while name in self.__names:
                    name = gen()
                return register(name)
            case NameMode.NONE:
                return None
            case _:
                raise ValueError(f'Invalid name mode: {self.name_mode}')


def _get_human_name():
    """Returns the full name of the current user, falling back to the username if necessary."""
    
    full_name = None

    if sys.platform.startswith("linux") or sys.platform == "darwin":  # macOS and Linux
        try:
            full_name = pwd.getpwuid(os.getuid()).pw_gecos.split(',')[0].strip()
        except KeyError:
            pass
        
        # Try getent as a fallback
        if not full_name:
            try:
                result = subprocess.check_output(["getent", "passwd", os.getlogin()], text=True)
                full_name = result.split(":")[4].split(",")[0].strip()
            except (subprocess.CalledProcessError, IndexError, FileNotFoundError, OSError):
                pass

    elif sys.platform.startswith("win"):  # Windows
        try:
            size = ctypes.wintypes.DWORD(0)
            ctypes.windll.advapi32.GetUserNameExW(3, None, ctypes.byref(size))  # Get required buffer size
            buffer = ctypes.create_unicode_buffer(size.value)
            if ctypes.windll.advapi32.GetUserNameExW(3, buffer, ctypes.byref(size)):
                full_name = buffer.value.strip()
        except Exception:
            pass

    # If full name is not found, fall back to the username
    if not full_name:
        full_name = getpass.getuser()

    return full_name

try:
    USERNAME = getpass.getuser()
except Exception:
    USERNAME = ''
try:
    USER = _get_human_name()
except Exception:
    USER = ''
