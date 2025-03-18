'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import pygltflib as gltf

from gltf_builder.element import (
    BPrimitive, BuilderProtocol, PrimitiveMode, Point, Vector3, Vector4,
    EMPTY_SET, BufferViewTarget, BMesh,
)

class _Primitive(BPrimitive):
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
                 mesh: Optional[BMesh]=None,
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
        lengths = {len(v) for v in self.attribs.values()}
        if len(lengths) > 1:
            raise ValueError('All attributes must have the same length')
        self.mesh = mesh

    def do_compile(self, builder: BuilderProtocol):
        index = self.mesh.primitives.index(self)
        mesh = self.mesh
        mesh.name = mesh.name or builder.gen_name(mesh)        
        def compile_attrib(name: str, data: list[tuple[float,...]]):
            prim_name = f'{mesh.name}:{self.mode.name}/{name}[{index}]'
            eltType, componentType = builder.get_attrib_info(name)
            view = builder.get_view(name, BufferViewTarget.ARRAY_BUFFER)
            accessor = view.add_accessor(eltType, componentType, data,
                                         name=prim_name)
            accessor.compile(builder)
            return accessor.index
        
        index_size = builder.get_index_size(len(self.points))
        if index_size >= 0:
            indices_view = builder.get_view('indices', BufferViewTarget.ELEMENT_ARRAY_BUFFER)
            indices = list(range(len(self.points)))
            if index_size == 0:
                match len(indices):
                    case  size if size < 255:
                        index_size = gltf.UNSIGNED_BYTE
                    case  size if size < 65535:
                        index_size = gltf.UNSIGNED_SHORT
                    case  _:
                        index_size = gltf.UNSIGNED_INT
            indices_accessor = indices_view.add_accessor(gltf.SCALAR, index_size, indices)
            indices_accessor.compile(builder)
        else:
            indices_accessor = None
        attrib_indices = {
            name: compile_attrib(name, data)
            for name, data in self.attribs.items()
        }
        return gltf.Primitive(
            mode=self.mode,
            indices=indices_accessor.index if indices_accessor else None,
            attributes=gltf.Attributes(**attrib_indices)
        )
