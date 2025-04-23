'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, cast

import pygltflib as gltf

from gltf_builder.compile import _Collected
from gltf_builder.core_types import (
    JsonObject, NPTypes, Phase, PrimitiveMode, BufferViewTarget, ScopeName,
)
from gltf_builder.attribute_types import (
AttributeData, AttributeDataIterable, AttributeDataList, AttributeDataSpec, PointSpec,
     point,
)
from gltf_builder.protocols import BType, _BuilderProtocol
from gltf_builder.element import (
    BTYPE_co, BAccessor, BPrimitive, BMesh, _Scope,
)
from gltf_builder.accessor import _Accessor
from gltf_builder.utils import decode_dtype


class _Primitive(BPrimitive):
    '''
    Base implementation class for primitives
    '''
    __attrib_accessors: Mapping[str, BAccessor[NPTypes, AttributeData]]
    __indices_accessor: Optional[BAccessor[NPTypes, int]] = None
    
    def __init__(self,
                 mode: PrimitiveMode,
                 points: Iterable[PointSpec] = (), /, *,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 mesh: Optional[BMesh]=None,
                 **attribs: AttributeDataIterable|None,
            ):
        super().__init__(extras, extensions)
        self.mode = mode
        if not points:
            points = cast(Iterable[PointSpec], attribs.pop('POSITION', None))
        self.points = list(point(p) for p in points)
        self.indices = []
        self.attribs = cast(dict[str, AttributeDataList], {
            k: list(v)
            for k,v in attribs.items()
            if v
        })
        self.attribs['POSITION'] = self.points
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
                           data: Sequence[BTYPE_co],
                        ) -> BAccessor[NPTypes, AttributeData]:
            index = mesh.primitives.index(self)
            aname = buffer.builder._gen_name(self,
                                            prefix=f'{mesh.name}:{self.mode.name}/',
                                            scope=ScopeName.ACCESSOR,
                                            index=index,
                                            )
            attr_type = builder.get_attribute_type(name)
            dtype = decode_dtype(attr_type.elementType, attr_type.componentType)
            accessor = _Accessor(buffer, len(data),
                                 attr_type.elementType,
                                 attr_type.componentType,
                                 btype=attr_type.type,
                                 dtype=dtype, 
                                 name=aname)
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
                    name = buffer.builder._gen_name(self,
                                                   prefix=f'{mesh.name}:{self.mode.name}/',
                                                   scope=ScopeName.ACCESSOR_INDEX,
                                                   index=index,
                                                   suffix='/indices',
                                                   )
                    self.__indices_accessor = _Accessor(
                        buffer, len(indices), gltf.SCALAR, index_size,
                        btype=int,
                        dtype=idtype,
                        name=name,
                        target=BufferViewTarget.ELEMENT_ARRAY_BUFFER
                    )
                    self.__indices_accessor._add_data(indices)
            case Phase.COLLECT:
                mesh.name = mesh.name or builder._gen_name(mesh)     
                self.__attrib_accessors = {
                    name: compile_attrib(name, cast(Sequence[AttributeData], data))
                    for name, data in self.attribs.items()
                }
                accessors: list[tuple[BAccessor[NPTypes,
                                                AttributeData]
                                     |BAccessor[NPTypes, int], list[_Collected]]] = [
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