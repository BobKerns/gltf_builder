'''
The initial objedt that collects the geometry info and compiles it into
a glTF object.
'''

from collections.abc import Iterable, Mapping
from typing import Optional, Any

import pygltflib as gltf

from gltf_builder.asset import BAsset
from gltf_builder.holder import MasterHolder
from gltf_builder.buffer import _Buffer
from gltf_builder.view import _BufferView
from gltf_builder.accessor import _Accessor
from gltf_builder.mesh import _Mesh
from gltf_builder.node import _Node, BNodeContainer
from gltf_builder.element import (
    EMPTY_SET, BBuffer, BufferViewTarget, BPrimitive,
    BuilderProtocol, ElementType, ComponentType,
)


class Builder(BNodeContainer, BuilderProtocol):
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
                index_size: int=32,
                extras: Mapping[str, Any]=EMPTY_SET,
                extensions: Mapping[str, Any]=EMPTY_SET,
        ):
        super().__init__(children=nodes)
        self.asset = asset
        self.meshes = MasterHolder(*meshes)
        self.nodes = MasterHolder(*nodes)
        if not buffers:
            buffers = [_Buffer('main')]
        self.buffers = MasterHolder(*buffers)
        self.views = MasterHolder(*views)
        self.accessors = MasterHolder(*accessors)
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
    
    def add_mesh(self,
                 name: str='',
                 primitives: Iterable[BPrimitive]=()
                ):
        mesh = _Mesh(name=name, primitives=primitives)
        #self.meshes.add(mesh)
        return mesh
    
    def add_buffer(self,
                   name: str='') -> _Buffer:
        buffer = _Buffer(name=name, index=len(self.buffers))
        self.buffers.add(buffer)
        return buffer
        
    def add_view(self,
                 name: str='',
                 buffer: Optional[BBuffer]=None,
                 data: Optional[bytes]=None,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
            ) -> _BufferView:
        buffer = buffer or self.buffers[0]
        view = _BufferView(name=name, buffer=buffer, data=data, target=target)
        self.views.add(view)
        return view
    
    def get_view(self, name: str,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
       ) -> _BufferView:
        if name in self.views:
            return self.views[name]
        return self.add_view(name=name, target=target)
    
    def build(self) -> gltf.GLTF2:
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
        
        g = gltf.GLTF2(
            nodes=[
                v
                for v in (
                    n.compile(self)
                    for n in nodes
                )
                if v is not None
            ],
            meshes=[
                m.compile(self)
                for m in self.meshes
            ],
            accessors=[
                a.compile(self)
                for a in self.accessors
                if a.count > 0
            ],
            # Sort the buffer views by alignment.
            bufferViews=[
                *(
                    v.compile(self)
                    for v in self.views
                    if len(v) % 4 == 0
                ),
                *(
                    v.compile(self)
                    for v in self.views
                    if len(v) % 4 == 2
                ),
                *(
                    v.compile(self)
                    for v in self.views
                    if len(v) % 4 in (1, 3)
                ),
            ],
            buffers=[
                b.compile(self)
                for b in self.buffers
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
        data = bytes(())
        for buf in self.buffers:
            data = data + buf.blob
        g.set_binary_blob(data)
        return g
    
    def define_attrib(self, name: str, type: ElementType, componentType: ComponentType):
        self.attr_type_map[name] = (type, componentType)

    def get_attrib_info(self, name: str) -> tuple[ElementType, ComponentType]:
        return self.attr_type_map.get(name) or self.attr_type_map['__DEFAULT__']

    def get_index_size(self, max_value: int) -> int:
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