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

    attr_type_map ={
        'TANGENT': (gltf.VEC3, gltf.FLOAT),
        '__DEFAULT__': (gltf.VEC3, gltf.FLOAT),
    }
    
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
                 **attribs: Iterable[tuple[int|float,...]],
            ):
        super().__init__(extras, extensions)
        self.mode = mode
        self.points = list(points)
        explicit_attribs = {
            'NORMAL': NORMAL,
            'TANGENT': TANGENT,
            'TEXCOORD_0': TEXCOORD_0,
            'TEXCOORD_1': TEXCOORD_1,
            'COLOR_0': COLOR_0,
            'JOINTS_0': JOINTS_0,
            'WEIGHTS_0': WEIGHTS_0,
        }
        self.attribs = {
            'POSITION': self.points,
            **attribs,
            **{
                k:list(v)
                for k, v in explicit_attribs.items()
                if v is not None
            }
        }

    def do_compile(self, builder: BuilderProtocol):
        def compile_attrib(name: str, data: list[tuple[float,...]]):
            types = BPrimitive.attr_type_map
            eltType, componentType = types.get(name) or types['__DEFAULT__']
            view = builder.get_view(name, BufferViewTarget.ARRAY_BUFFER)
            accessor = view.add_accessor(eltType, componentType, data)
            accessor.compile(builder)
            return accessor.index
        indices_view = builder.get_view('indices', BufferViewTarget.ELEMENT_ARRAY_BUFFER)
        indices_accessor = indices_view.add_accessor(gltf.SCALAR, gltf.UNSIGNED_INT, list(range(len(self.points))))
        indices_accessor.compile(builder)
        attrib_indices = {
            name: compile_attrib(name, data)
            for name, data in self.attribs.items()
        }
        return gltf.Primitive(
            mode=self.mode,
            indices=indices_accessor.index,
            attributes=gltf.Attributes(**attrib_indices)
        )