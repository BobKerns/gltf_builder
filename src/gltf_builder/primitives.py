'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import pygltflib as gltf

from gltf_builder.element import (
    BPrimitiveProtocol, BuilderProtocol, PrimitiveMode, Point, Vector3, Vector4,
    EMPTY_SET, BufferViewTarget,
)

class BPrimitive(BPrimitiveProtocol):
    '''
    Base implementation class for primitives
    '''
    
    def __init__(self,
                 mode: PrimitiveMode,
                 points: Optional[Iterable[Point]]=None,
                 NORMAL: Optional[Iterable[Vector3]]=None,
                 TANGENT: Optional[Iterable[Vector4]]=None,
                 TEXCOORD_0: Optional[Iterable[Point]]=None,
                 TEXCOORD_1: Optional[Iterable[Point]]=None,
                 COLOR_0: Optional[Iterable[Point]]=None,
                 JOINTS_0: Optional[Iterable[Point]]=None,
                 WEIGHTS_0: Optional[Iterable[Point]]=None,
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
            ):
        super().__init__(extras, extensions)
        self.mode = mode
        self.points = list(points)

    def do_compile(self, builder: BuilderProtocol):
        points_view = builder.get_view('POSITION', BufferViewTarget.ARRAY_BUFFER)
        points_accessor = points_view.add_accessor(gltf.VEC3, gltf.FLOAT, self.points)
        points_accessor.compile(builder)
        indices_view = builder.get_view('indices', BufferViewTarget.ELEMENT_ARRAY_BUFFER)
        indices_accessor = indices_view.add_accessor(gltf.SCALAR, gltf.UNSIGNED_INT, list(range(len(self.points))))
        indices_accessor.compile(builder)
        return gltf.Primitive(
            mode=self.mode,
            indices=indices_accessor.index,
            attributes=gltf.Attributes(POSITION=points_accessor.index)
        )