'''
Builder representation of a mesh to be compiled.
'''

from collections.abc import Iterable, Mapping
from typing import Any

import pygltflib as gltf

from gltf_builder.element import (
    BuilderProtocol, BMeshProtocol, EMPTY_SET,
)
from gltf_builder.primitives import Point, BPrimitive, PrimitiveMode


class BMesh(BMeshProtocol):
    def __init__(self, /,
                 name='',
                 primitives: Iterable[BPrimitive]=(),
                 weights: Iterable[float]=(),
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
            ):
        super().__init__(name, extras, extensions)
        self.primitives = list(primitives)
        self.weights = list(weights)
        
    def add_primitive(self, type: PrimitiveMode, *points: Point) -> BPrimitive:
        prim = BPrimitive(type, points)
        self.primitives.append(prim)
        return prim
    
    def do_compile(self, builder: BuilderProtocol):
        builder.meshes.add(self)
        return gltf.Mesh(
            name=self.name,
            primitives=[
                p.compile(builder)
                for p in self.primitives
            ]
        )
