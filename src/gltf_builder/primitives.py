'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, cast, overload

import pygltflib as gltf

from gltf_builder.compile import _Collected
from gltf_builder.core_types import (
    JsonObject, NPTypes, Phase, PrimitiveMode, BufferViewTarget,
)
from gltf_builder.attribute_types import (
    AttributeDataItem, ColorSpec, JointSpec, PointSpec,
    TangentSpec, UvSpec, Vector3Spec, WeightSpec, color, joint, point, tangent, uv, vector3, weight,
)
from gltf_builder.protocols import BType, _BuilderProtocol
from gltf_builder.element import (
    BTYPE, BAccessor, BPrimitive, BMesh, _Scope,
)
from gltf_builder.accessor import _Accessor
from gltf_builder.utils import decode_dtype


class _Primitive(BPrimitive):
    '''
    Base implementation class for primitives
    '''
    __attrib_accessors: Mapping[str, BAccessor[NPTypes,BType]]
    __indices_accessor: Optional[BAccessor[NPTypes, int]] = None
    
    def __init__(self,
                 mode: PrimitiveMode,
                 points: Iterable[PointSpec],
                 NORMAL: Optional[Iterable[Vector3Spec]]=None,
                 TANGENT: Optional[Iterable[TangentSpec]]=None,
                 TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                 TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                 COLOR_0: Optional[Iterable[ColorSpec]]=None,
                 JOINTS_0: Optional[Iterable[JointSpec]]=None,
                 WEIGHTS_0: Optional[Iterable[WeightSpec]]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 mesh: Optional[BMesh]=None,
                 **attribs: Iterable[AttributeDataItem],
            ):
        super().__init__(extras, extensions)
        self.mode = mode
        self.points = list(points)
        self.indices = []
        @overload
        def collect_attrib(data: Iterable[AttributeDataItem]
                        ) -> Sequence[AttributeDataItem]: ...
        @overload
        def collect_attrib(data: None) -> None: ...
        @overload
        def collect_attrib(data: Iterable[AttributeDataItem]|None
                        ) -> Sequence[AttributeDataItem]|None: ...
        def collect_attrib(data: Iterable[AttributeDataItem]|None
                        ) -> Sequence[AttributeDataItem]|None:
            '''
            If only an Iterable is given, convert it to a list, as we
            need to know the length of the data.
            '''
            if data is None:
                return None
            if isinstance(data, Sequence):
                return data
            return list(data)
        explicit_attribs: dict[str, Iterable[AttributeDataItem]|None] = {
            'NORMAL': [vector3(n) for n in NORMAL or ()],
            'TANGENT': [tangent(t) for t in TANGENT or ()],
            'TEXCOORD_0': [uv(u) for u in TEXCOORD_0 or ()],
            'TEXCOORD_1': [uv(u) for u in TEXCOORD_1 or ()],
            'COLOR_0': [color(c) for c in COLOR_0 or ()],
            'JOINTS_0': [j for j0 in JOINTS_0 or () for j in joint(j0)],
            'WEIGHTS_0': [w for w0 in WEIGHTS_0 or () for w in weight(w0)],
        }
        self.attribs: Mapping[str, Sequence[AttributeDataItem]] = {
            'POSITION': [point(p) for p in self.points],
            **{
                n: collect_attrib(v)
                for n, v in attribs.items()
                if v
            },
            **{
                k: collect_attrib(v)
                for k, v in explicit_attribs.items()
                if v
            }
        }
        lengths = {len(v) for v in self.attribs.values() if v}
        if len(lengths) > 1:
            raise ValueError('All attributes must have the same length')
        self.mesh = mesh
        self.__attrib_accessors = {}

    def _do_compile(self, builder: _BuilderProtocol, scope: _Scope, phase: Phase):
        mesh = self.mesh
        assert mesh is not None
        buffer = builder._buffers[0]
        def compile_attrib(name: str,
                           data: Sequence[BTYPE],
                        ) -> BAccessor[NPTypes, BType]:
            index = mesh.primitives.index(self)
            name = buffer.builder._gen_name(self) or ''
            prim_name = f'{mesh.name}:{self.mode.name}/{name}[{index}]'
            eltType, componentType, btype = builder.get_attrib_info(name)
            dtype = decode_dtype(eltType, componentType)
            accessor = _Accessor(buffer, len(data), eltType, componentType,
                                 btype=btype,
                                 dtype=dtype, 
                                 name=prim_name)
            accessor._add_data(data)
            accessor.compile(builder, scope, phase)
            return accessor
        match phase:
            case Phase.PRIMITIVES:
                index_size = builder._get_index_size(len(self.points))
                if index_size != -1:
                    indices = list(range(len(self.points)))
                    idtype = decode_dtype(gltf.SCALAR, index_size)
                    index = mesh.primitives.index(self)
                    self.__indices_accessor = _Accessor(
                        buffer, len(indices), gltf.SCALAR, index_size,
                        btype=int,
                        dtype=idtype,
                        name=f'{mesh.name}:{self.mode.name}/indices[{index}]',
                        target=BufferViewTarget.ELEMENT_ARRAY_BUFFER
                    )
                    self.__indices_accessor._add_data(indices)
            case Phase.COLLECT:
                mesh.name = mesh.name or builder._gen_name(mesh.name) or ''       
                self.__attrib_accessors = {
                    name: compile_attrib(name, cast(Sequence[BType], data))
                    for name, data in self.attribs.items()
                }
                accessors: list[tuple[BAccessor[NPTypes, BType]|BAccessor[NPTypes, int], list[_Collected]]] = [
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
                    name: acc._index
                    for name, acc in self.__attrib_accessors.items()
                }
                indices = self.__indices_accessor._index if self.__indices_accessor else None
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
        return f'<{self.mode.name} {self.mesh}[{self._index}]'

    def __str__(self):
        return f'{self.mesh}[{self._index}]({self.mode.name})'