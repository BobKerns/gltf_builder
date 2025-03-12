'''
The initial objedt that collects the geometry info and compiles it into
a glTF object.
'''

from collections.abc import Sequence, Iterable

import pygltflib as gltf
import numpy as np

from gltf_builder.primitives import Point
from gltf_builder.node import BNode, BNodeContainer


class Builder(BNodeContainer):
    asset: gltf.Asset
    points: list[Point]
    nodes: Sequence[BNode]
    def __init__(self,
                asset: gltf.Asset= gltf.Asset(),
                nodes: Sequence[BNode] = (),
        ):
        super().__init__(nodes)
        self.asset = asset
        self.points = []
        
    def build(self):
        def flatten(node: BNode) -> Iterable[BNode]:
            yield node
            for n in node.children:
                yield from flatten(n)
        
        nodes = [i for n in self.nodes
                               for i in flatten(n)]
        for i,n in enumerate(nodes):
            n.index = i
        
        points = [
            p for n in nodes
              for prim in n.primitives
              for p in set(prim.points)              
        ]
        points_idx = {
            id(p):p for p in points
        }
        indices = [
            points_idx[id(p)]
            for n in nodes
            for prim in n.primitives
            for p in prim.points
        ]
        points_blob = np.array(points, np.float32).flatten().tobytes()
        indices_blob = np.array(indices, np.uint32).flatten().tobytes()
        points_view = gltf.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(points_blob)
            )
        indices_view = gltf.BufferView(
            buffer=0,
            byteOffset=len(points_blob),
            byteLength=len(indices_blob),
        )
        
        buffer = gltf.Buffer(
            byteLength=len(points_blob) + len(indices_blob)
        )
        
        g = gltf.GLTF2(
            buffers=[buffer],
            bufferViews=[
                points_view,
                indices_view,
            ],
            nodes = [
                gltf.Node()
                for n in nodes
            ]
        )
        g.set_binary_blob(points_blob + indices_blob)
        return g
    