'''
Builder representation of a glTF Accessor
'''

from typing import Optional, cast
from collections.abc import Iterable, Sequence
from pathlib import Path

import pygltflib as gltf
import numpy as np
from gltf_builder.core_types import (
    ComponentType, ElementType, JsonObject, Phase, BufferViewTarget, ScopeName,
)
from gltf_builder.attribute_types import BTYPE, BTYPE_co, BType
from gltf_builder.protocols  import _BuilderProtocol
from gltf_builder.elements import (
    BAccessor, BBuffer, NP
)
from gltf_builder.compiler import _CompileState, _DoCompileReturn, _Scope
from gltf_builder.utils import decode_dtype, decode_stride, decode_type
from gltf_builder.log import GLTF_LOG


LOG = GLTF_LOG.getChild(Path(__name__).stem)

class _Accessor(BAccessor[NP, BTYPE]):
    __memory: memoryview
    dtype: type[NP]
    btype: BType
    
    @property
    def memory(self):
        return self.__memory
    
    def __init__(self, /,
                 buffer: BBuffer,
                 count: int,
                 elementType: ElementType,
                 componentType: ComponentType,
                 dtype: type[NP],
                 btype: type[BTYPE_co],
                 name: str='',
                 normalized: bool=False,
                 max: Optional[list[float]]=None,
                 min: Optional[list[float]]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 target: BufferViewTarget = BufferViewTarget.ARRAY_BUFFER,
    ):
        super().__init__(name=name,
                         extras=extras,
                         extensions=extensions,
                    )
        byteStride = decode_stride(elementType, componentType)
        self.dtype = cast(type[NP], decode_dtype(elementType, componentType))
        vname = buffer.builder._gen_name(self, scope=ScopeName.BUFFER_VIEW, suffix='/view')
        self.view = buffer._get_view(buffer, target, byteStride=byteStride, name=vname)
        self.view._add_accessor(self)
        self.count = count
        self.elt_type = elementType
        self.name = name
        self.componentType = componentType
        self.normalized = normalized
        self.max = max
        self.min = min
        self.dtype = dtype
        self.btype = btype
        self.data = []

    def log_offset(self):
        if self.byteOffset >= 0:
            LOG.debug('%s has offset %d(+%d)',
                    self, self.byteOffset,
                    self.view.byteOffset
                    )

    def _add_data(self, data: Sequence[BTYPE]):
        self.data.extend(data)
    
    def _add_data_item(self, data: BTYPE):
        self.data.append(data)
    
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState,
                    /
                    ) -> _DoCompileReturn[gltf.Accessor]:
        match phase:
            case Phase.COLLECT:
                builder._accessors.add(self)
                return [(self,())]
            case Phase.SIZES:
                (
                    self.componentCount,
                    self.componentSize,
                    self.byteStride,
                    self.dtype, # type: ignore
                    self.bufferType
                ) = decode_type(self.elt_type, self.componentType)
                ldata = sum(
                    len(d) if isinstance(d, (Sequence, np.ndarray)) else 1
                    for d in self.data
                )
                return ldata * self.componentSize
            case Phase.OFFSETS:
                self.view.compile(builder, scope, phase)
                self.__memory = self.view.memoryview(self.byteOffset, len(self))
            case Phase.BUILD:
                data = np.array(self.data, self.dtype)
                if len(self.data) == 0:
                    min_axis = max_axis = [0]
                else:
                    min_axis = self.min or data.min(axis=0)
                    max_axis = self.max or data.max(axis=0)
                if isinstance(min_axis, Iterable):
                    min_axis = [float(v) for v in min_axis] # type: ignore
                else:
                    min_axis = [float(min_axis)] * self.componentCount
                if isinstance(max_axis, Iterable):
                    max_axis = [float(v) for v in max_axis] # type: ignore
                else:
                    max_axis = [float(max_axis)] * self.componentCount
                self.__memory[:] = data.tobytes()
                return gltf.Accessor(
                    bufferView=self.view._index,
                    count=self.count,
                    type=self.elt_type,
                    componentType=self.componentType,
                    name=self.name,
                    byteOffset=self.byteOffset,
                    normalized=self.normalized,
                    max=max_axis,
                    min=min_axis,
                )
            case _: pass
                 