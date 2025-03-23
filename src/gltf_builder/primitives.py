'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import pygltflib as gltf

from gltf_builder.types import (
    Phase, PrimitiveMode, EMPTY_MAP,
    Point, Vector3, Vector4, BufferViewTarget,
)
from gltf_builder.element import (
    BAccessor, BPrimitive, BMesh, BuilderProtocol, _Scope,
)
from gltf_builder.accessor import _Accessor


class _Primitive(BPrimitive):
    '''
    Base implementation class for primitives
    '''
    __attrib_accessors: Mapping[str, BAccessor]
    __indices_accessor: Optional[BAccessor] = None
    
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
                 extras: Mapping[str, Any]=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
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
        self.__attrib_accessors = {}

    def _do_compile(self, builder: BuilderProtocol, scope: _Scope, phase: Phase):
        mesh = self.mesh
        buffer = builder._buffers[0]
        def compile_attrib(name: str, data: list[tuple[float,...]]):
            index = self.mesh.primitives.index(self)
            prim_name = f'{mesh.name}:{self.mode.name}/{name}[{index}]'
            eltType, componentType = builder.get_attrib_info(name)
            accessor = _Accessor(buffer, len(data), eltType, componentType, 
                                 name=prim_name)
            accessor._add_data(data)
            accessor._do_compile(builder, scope, phase)
            return accessor
        match phase:
            case Phase.PRIMITIVES:
                index_size = builder._get_index_size(len(self.points))
                if index_size >= 0:
                    indices = list(range(len(self.points)))
                    if index_size == 0:
                        match len(indices):
                            case  size if size < 255:
                                index_size = gltf.UNSIGNED_BYTE
                            case  size if size < 65535:
                                index_size = gltf.UNSIGNED_SHORT
                            case  _:
                                index_size = gltf.UNSIGNED_INT
                    index = self.mesh.primitives.index(self)
                    self.__indices_accessor = _Accessor(buffer, len(indices), gltf.SCALAR, index_size,
                                                        name=f'{mesh.name}:{self.mode.name}/indices[{index}]',
                                                        target=BufferViewTarget.ELEMENT_ARRAY_BUFFER)
                    self.__indices_accessor._add_data(indices)
            case Phase.COLLECT:
                mesh.name = mesh.name or builder._gen_name(mesh)        
                self.__attrib_accessors = {
                    name: compile_attrib(name, data)
                    for name, data in self.attribs.items()
                }
                accessors =  [
                    (a, [a.compile(builder, scope, phase)])
                    for a in self.__attrib_accessors.values()
                ]
                ia = self.__indices_accessor
                if ia:
                    accessors.append((ia, [ia.compile(builder, scope, phase)]))
                return accessors
            case phase.SIZES:
                size = sum(
                    acc.compile(builder, scope, phase)
                    for acc in self.__attrib_accessors.values()
                )
                if self.__indices_accessor:
                    size += self.__indices_accessor.compile(builder, scope, phase)
                return size
            case Phase.OFFSETS:
                for acc in self.__attrib_accessors.values():
                    acc.compile(builder, scope, phase)
                if self.__indices_accessor:
                    self.__indices_accessor.compile(builder, scope, phase)
            case Phase.BUILD:
                attributes = {
                    name: acc.index
                    for name, acc in self.__attrib_accessors.items()
                }
                indices = self.__indices_accessor.index if self.__indices_accessor else None
                return gltf.Primitive(
                    mode=self.mode,
                    indices=indices,
                    attributes=gltf.Attributes(**attributes)
                )
            case _:
                for acc in self.__attrib_accessors.values():
                    acc.compile(builder, scope, phase)
                if self.__indices_accessor:
                    self.__indices_accessor.compile(builder, scope, phase)
                return None
            
    def __repr__(self):
        return f'<{self.mode.name} {self.mesh}[{self.index}]'

    def __str__(self):
        return f'{self.mesh}[{self.index}]({self.mode.name})'