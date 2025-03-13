'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Mapping
from typing import Any

import pygltflib as gltf

from gltf_builder.element import (
    BPrimitiveProtocol, BuilderProtocol, PrimitiveMode, Point, EMPTY_SET,
)

class BPrimitive(BPrimitiveProtocol):
    '''
    Base implementation class for primitives
    '''
    
    def __init__(self,
                 mode: PrimitiveMode,
                 points: Iterable[Point] = (),
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
            ):
        super().__init__(extras, extensions)
        self.mode = mode
        self.points = list(points)

    def do_compile(self, builder: BuilderProtocol):
        points_view = builder.views['points']
        points_accessor = points_view.add_accessor(gltf.VEC3, gltf.FLOAT, self.points)
        points_accessor.compile(builder)
        indices_view = builder.views['indices']
        indices_accessor = indices_view.add_accessor(gltf.SCALAR, gltf.UNSIGNED_INT, list(range(len(self.points))))
        indices_accessor.compile(builder)
        return gltf.Primitive(
            mode=self.mode,
            indices=indices_accessor.index,
            attributes=gltf.Attributes(POSITION=points_accessor.index)
        )