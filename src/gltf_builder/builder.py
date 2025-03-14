'''
The initial objedt that collects the geometry info and compiles it into
a glTF object.
'''

from collections.abc import Iterable, Mapping
from typing import Optional, Any

import pygltflib as gltf

from gltf_builder.asset import BAsset
from gltf_builder.holder import Holder, MasterHolder
from gltf_builder.primitives import Point, BPrimitive
from gltf_builder.buffer import BBuffer
from gltf_builder.view import BBufferView
from gltf_builder.accessor import BAccessor
from gltf_builder.mesh import BMesh
from gltf_builder.node import BNode, BNodeContainer
from gltf_builder.element import (
    EMPTY_SET, BufferViewTarget,
)


class Builder(BNodeContainer):
    '''
    The main object that collects all the geometry info and compiles it into a glTF object.
    '''
    asset: gltf.Asset
    points: list[Point]
    meshes: Holder[BMesh]
    buffers: Holder[BBuffer]
    views: Holder[BBufferView]
    accessors: Holder[BAccessor]
    extras: dict[str, Any]
    extensions: dict[str, Any]
    def __init__(self,
                asset: gltf.Asset= BAsset(),
                meshes: Iterable[BMesh]=(),
                nodes: Iterable[BNode] = (),
                buffers: Iterable[BBuffer]=(),
                views: Iterable[BBufferView]=(),
                accessors: Iterable[BAccessor]=(),
                extras: Mapping[str, Any]=EMPTY_SET,
                extensions: Mapping[str, Any]=EMPTY_SET,
        ):
        super().__init__(nodes)
        self.asset = asset
        self.points = []
        self.meshes = MasterHolder(*meshes)
        self.nodes = MasterHolder(*nodes)
        if not buffers:
            buffers = [BBuffer('main')]
        self.buffers = MasterHolder(*buffers)
        self.views = MasterHolder(*views)
        self.accessors = MasterHolder(*accessors)
        self.extras = dict(extras)
        self.extensions = dict(extensions)
        self.get_view(name='POSITION', target=BufferViewTarget.ARRAY_BUFFER)
        self.add_view(name='indices', target=BufferViewTarget.ELEMENT_ARRAY_BUFFER)
        
    def add_mesh(self,
                 name: str='',
                 primitives: Iterable[BPrimitive]=()
                ):
        mesh = BMesh(name=name, primitives=primitives)
        self.meshes.add(mesh)
        return mesh
    
    def add_buffer(self,
                   name: str=''):
        buffer = BBuffer(name=name, index=len(self.buffers))
        self.buffers.add(buffer)
        return buffer
        
    def add_view(self,
                 name: str='',
                 buffer: Optional[BBuffer]=None,
                 data: Optional[bytes]=None,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
            ) -> BBufferView:
        buffer = buffer or self.buffers[0]
        view = BBufferView(name=name, buffer=buffer, data=data, target=target)
        self.views.add(view)
        return view
    
    def get_view(self, name: str,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
       ) -> BBufferView:
        if name in self.views:
            return self.views[name]
        return self.add_view(name=name, target=target)
    
    def build(self):
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
            buffers=[
                b.compile(self)
                for b in self.buffers
            ],
            bufferViews=[
                v.compile(self)
                for v in self.views
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
            data = data + buf.data
        g.set_binary_blob(data)
        return g
    