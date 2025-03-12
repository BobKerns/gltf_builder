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
        self.nodes = nodes
        
    def build(self):
        def flatten(node: BNode) -> Iterable[BNode]:
            for n in node.children:
                yield from flatten(n)
        
        nodes = [i for n in self.nodes
                               for i in flatten(n)]
        points = [
            p for n in nodes
              for p in set(n.points)
        ]
        points_idx = {
            id(p):p for p in points
        }
        indices = [
            points_idx[id(p)]
            for n in nodes
            for p in n.points
        ]
        points_blob = np.array(points, np.float32).flatten().tobytes()
        indices_blob = np.array(indices, np.unsignedinteger).flatten().tobytes()
        
        g = gltf.GLTF2()
        g.set_binary_blob(points_blob + indices_blob)
        return g
    