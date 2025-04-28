'''
Definitions for GLTF primitives
'''

from collections.abc import Iterable, Sequence
from typing import Any, Optional, Self, cast, TYPE_CHECKING

import pygltflib as gltf

from gltf_builder.compiler import (
    _GLTF, _STATE, _Collected, _CompileState, _DoCompileReturn,
)
from gltf_builder.core_types import (
    ExtensionsData, ExtrasData, IndexSize, NPTypes,
    Phase, PrimitiveMode, BufferViewTarget, ScopeName,
)
from gltf_builder.attribute_types import (
    BTYPE, AttributeData, AttributeDataIterable, AttributeDataList,
    point, Point, PointSpec,
)
from gltf_builder.elements import (
    BAccessor, BPrimitive, BMesh, _Scope, Element,
)
from gltf_builder.accessors import _Accessor
from gltf_builder.utils import decode_dtype
if TYPE_CHECKING:
    from gltf_builder.global_state import _GlobalState
    from gltf_builder.compiler import _Compilable


class _PrimitiveState(_CompileState[gltf.Primitive, '_PrimitiveState']):
    '''
    State for the compilation of a primitive.
    '''
    accessors: dict[str, BAccessor[NPTypes, AttributeData]]
    indices_accessor: Optional[BAccessor[NPTypes, int]] = None
    def __init__(self,
                 primitive: '_Primitive',
                 name: str='',
                 /,
                ) -> None:
        super().__init__(cast('_Compilable', primitive), name)
        self.accessors = {}

class _Primitive(BPrimitive):
    '''
    Base implementation class for primitives
    '''

    @classmethod
    def state_type(cls):
        return _PrimitiveState

    def __init__(self,
                 mode: PrimitiveMode,
                 points: Iterable[Point] = (), /, *,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
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


    def clone(self, name: str='', /,
              extras: Optional[ExtrasData]=None,
              extensions: Optional[ExtensionsData]=None,
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
                    gbl: '_GlobalState',
                    scope: _Scope,
                    phase: Phase,
                    state: _PrimitiveState,
                    /
                ) -> _DoCompileReturn[gltf.Primitive]:
        def _compile(elt: Element[_GLTF, _STATE]):
            return elt.compile(gbl, scope, phase)

        mesh = self.mesh
        assert mesh is not None
        buffer = gbl.buffers[0]
        def compile_attrib(name: str,
                           data: Sequence[BTYPE],
                        ) -> BAccessor[NPTypes, AttributeData]:
            index = mesh.primitives.index(self)
            aname = gbl._gen_name(self,
                                        prefix=f'{mesh.name}:{self.mode.name}/',
                                        scope=ScopeName.ACCESSOR,
                                        index=index,
                                    )
            attr_type = gbl.get_attribute_type(name)
            dtype = decode_dtype(attr_type.elementType, attr_type.componentType)
            accessor = _Accessor(buffer, len(data),
                                 attr_type.elementType,
                                 attr_type.componentType,
                                 btype=attr_type.type,
                                 dtype=dtype,
                                 name=aname)
            astate = gbl.state(cast('_Compilable', accessor))
            astate.add_data(data)
            _compile(accessor)
            return accessor
        match phase:
            case Phase.PRIMITIVES:
                index_size = gbl._get_index_size(len(self.points))
                assert index_size != IndexSize.AUTO
                if index_size != IndexSize.NONE:
                    indices = list(range(len(self.points)))
                    idtype = decode_dtype(gltf.SCALAR, index_size)
                    index = mesh.primitives.index(self)
                    name = gbl._gen_name(self,
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
                        case _:
                            raise ValueError(f'Invalid index size: {index_size}')
                    state.indices_accessor = _Accessor(
                        buffer, len(indices),
                        gltf.SCALAR,
                        index_type,
                        btype=int,
                        dtype=idtype,
                        name=name,
                        target=BufferViewTarget.ELEMENT_ARRAY_BUFFER
                    )
                    istate = gbl.state(cast('_Compilable', state.indices_accessor))
                    istate.add_data(indices)
            case Phase.COLLECT:
                mesh.name = mesh.name or gbl._gen_name(mesh)
                state.accessors = {
                    name: compile_attrib(name, cast(Sequence[AttributeData], data))
                    for name, data in self.attribs.items()
                }
                accessors: list[tuple[BAccessor[NPTypes,
                                                AttributeData]
                                     |BAccessor[NPTypes, int], list[_Collected]]] = [
                    (a, [a.compile(gbl, scope, phase)])
                    for a in state.accessors.values()
                ]
                ia = state.indices_accessor
                if ia:
                    accessors.append((ia, [ia.compile(gbl, scope, phase)]))
                return accessors
            case phase.SIZES:
                size = sum(
                    cast(int, _compile(acc))
                    for acc in state.accessors.values()
                )
                if state.indices_accessor:
                    size += cast(int, _compile(state.indices_accessor))
                return size
            case Phase.OFFSETS:
                for acc in state.accessors.values():
                    _compile(acc)
                if state.indices_accessor:
                    _compile(state.indices_accessor)
            case Phase.BUILD:
                attributes = {
                    name: gbl.idx(acc)
                    for name, acc in state.accessors.items()
                }
                i_accessor = state.indices_accessor
                indices = gbl.idx(i_accessor) if i_accessor else None
                return gltf.Primitive(
                    mode=self.mode,
                    indices=indices,
                    attributes=gltf.Attributes(**attributes)
                )
            case _:
                for acc in state.accessors.values():
                    acc.compile(gbl, scope, phase)
                if state.indices_accessor:
                    state.indices_accessor.compile(gbl, scope, phase)
                return None

    def __repr__(self):
        builder = self.mesh.gbl if self.mesh else None
        if builder and builder._get_index_size(len(self.points)) > 0:
            indexed = 'indexed '
        else:
            indexed = ''
        if self.mesh:
            idx = self.mesh.primitives.index(self)
            return f'<{self.mode.name} {indexed}{self.mesh}[{idx}] points={len(self.points)}>'

        return f'<{self.mode.name} {indexed} points={len(self.points)}>'

    def __str__(self):
        if self.mesh is None:
            return f'<{self.mode.name} points={len(self.points)}>'
        idx = self.mesh.primitives.index(self)
        return f'{self.mesh}[{idx}]<{self.mode.name} points={len(self.points)}>'

def primitive(
    mode: PrimitiveMode,
    points: Iterable[PointSpec] = (), /, *,
    extras: Optional[ExtrasData]=None,
    extensions: Optional[ExtensionsData]=None,
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