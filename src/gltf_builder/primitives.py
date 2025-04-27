'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Self, cast

import pygltflib as gltf

from gltf_builder.compiler import _GLTF, _STATE, _Collected, _CompileState, _DoCompileReturn
from gltf_builder.core_types import (
    IndexSize, JsonObject, NPTypes, Phase, PrimitiveMode, BufferViewTarget, ScopeName,
)
from gltf_builder.attribute_types import (
    BTYPE, AttributeData, AttributeDataIterable, AttributeDataList, Point, PointSpec,
     point,
)
from gltf_builder.protocols import _BuilderProtocol
from gltf_builder.elements import (
    BAccessor, BPrimitive, BMesh, _Scope, Element,
)
from gltf_builder.accessors import _Accessor
from gltf_builder.utils import decode_dtype

class _PrimitiveState(_CompileState[gltf.Primitive, '_PrimitiveState']):
    '''
    State for the compilation of a primitive.
    '''
    _accessors: list[BAccessor[NPTypes, AttributeData]]
    _indices_accessor: Optional[BAccessor[NPTypes, int]] = None
    def __init__(self,
                 primitive: '_Primitive',
                 name: str='',
                 /,
                ) -> None:
        super().__init__(name, primitive)
        self._accessors = []

class _Primitive(BPrimitive):
    '''
    Base implementation class for primitives
    '''

    @classmethod
    def state_type(cls):
        return _PrimitiveState


    __attrib_accessors: Mapping[str, BAccessor[NPTypes, AttributeData]]
    __indices_accessor: Optional[BAccessor[NPTypes, int]] = None
    
    def __init__(self,
                 mode: PrimitiveMode,
                 points: Iterable[Point] = (), /, *,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 mesh: Optional[BMesh]=None,
                 **attribs: AttributeDataIterable|None,
            ):
        super().__init__(extras=extras, extensions=extensions)
        self.mode = mode
        if not points:
            points = cast(Iterable[Point], attribs.pop('POSITION', None))
        if not points:
            raise ValueError('At least one point is required')
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

    
    def clone(self, name: str='', /,
              extras: Optional[JsonObject]=None,
              extensions: Optional[JsonObject]=None,
              **kwargs: Any,
            ) -> Self:
        '''
        Clone the object, copying the name, extras, and extensions.
        '''
        return self.__class__(
            self.mode,
            self.points,
            mesh=self.mesh,
            extras={**self.extras, **(extras or {})},
            extensions={**self.extensions, **(extensions or {})},
        )


    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState[gltf.Primitive, '_PrimitiveState'],
                    /
                ) -> _DoCompileReturn[gltf.Primitive]:
        def _compile(elt: Element[_GLTF, _STATE]):
            return elt.compile(builder, scope, phase)
        
        mesh = self.mesh
        assert mesh is not None
        buffer = builder._buffers[0]
        def compile_attrib(name: str,
                           data: Sequence[BTYPE],
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
            _compile(accessor)
            return accessor
        match phase:
            case Phase.PRIMITIVES:
                index_size = builder._get_index_size(len(self.points))
                assert index_size != IndexSize.AUTO
                if index_size != IndexSize.NONE:
                    indices = list(range(len(self.points)))
                    idtype = decode_dtype(gltf.SCALAR, index_size)
                    index = mesh.primitives.index(self)
                    name = buffer.builder._gen_name(self,
                                                   prefix=f'{mesh.name}:{self.mode.name}/',
                                                   scope=ScopeName.ACCESSOR_INDEX,
                                                   index=index,
                                                   suffix='/indices',
                                                   )
                    match index_size:
                        case IndexSize.UNSIGNED_BYTE:
                            index_type = gltf.UNSIGNED_BYTE
                        case IndexSize.UNSIGNED_SHORT:
                            index_type = gltf.UNSIGNED_SHORT
                        case IndexSize.UNSIGNED_INT:
                            index_type = gltf.UNSIGNED_INT
                    self.__indices_accessor = _Accessor(
                        buffer, len(indices), gltf.SCALAR, index_type,
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
                    cast(int, _compile(acc))
                    for acc in self.__attrib_accessors.values()
                )
                if self.__indices_accessor:
                    size += cast(int, _compile(self.__indices_accessor))
                return size
            case Phase.OFFSETS:
                for acc in self.__attrib_accessors.values():
                    _compile(acc)
                if self.__indices_accessor:
                    _compile(self.__indices_accessor)
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
        builder = self.mesh.builder if self.mesh else None
        if builder and builder._get_index_size(len(self.points)) > 0:
            indexed = 'indexed '
        else:
            indexed = ''
        return f'<{self.mode.name} {indexed}{self.mesh}[{self._index}]'

    def __str__(self):
        return f'{self.mesh}[{self._index}]({self.mode.name})'
    
def primitive(
    mode: PrimitiveMode,
    points: Iterable[PointSpec] = (), /, *,
    extras: Optional[JsonObject]=None,
    extensions: Optional[JsonObject]=None,
    **attribs: AttributeDataIterable|None,
) -> _Primitive:
    '''
    Create a detached primitive with the given attributes.

    Parameters
    ----------
    mode : PrimitiveMode
        The primitive mode.
    points : Iterable[PointSpec], optional
        The points of the primitive.
    extras : Optional[JsonObject], optional
        Extra data to be attached to the primitive.
    extensions : Optional[JsonObject], optional
        Extensions to be attached to the primitive.
    '''
    return _Primitive(mode, (point(p) for p in points),
                      extras=extras,
                      extensions=extensions,
                      mesh=None,
                      **attribs)