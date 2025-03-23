'''
Builder representation of a glTF Accessor
'''

from typing import Optional, Any
from collections.abc import Mapping, Iterable

import pygltflib as gltf
import numpy as np

from gltf_builder.element import (
    BAccessor, BufferViewTarget, BuilderProtocol, BBuffer, EMPTY_MAP,
    ElementType, _Scope, Phase, AttributeDataSequence, AttributeDataItem
)
from gltf_builder.utils import decode_dtype, decode_stride, decode_type

class _Accessor(BAccessor):
    __memory: memoryview
    dtype: np.dtype
    @property
    def memory(self):
        return self.__memory
    
    def __init__(self, /,
                 buffer: BBuffer,
                 count: int,
                 type: ElementType,
                 componentType: int,
                 name: str='',
                 normalized: bool=False,
                 max: Optional[list[float]]=None,
                 min: Optional[list[float]]=None,
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]|None=EMPTY_MAP,
                 target: BufferViewTarget = BufferViewTarget.ARRAY_BUFFER,
    ):
        super().__init__(name, extras, extensions)
        byteStride = decode_stride(type, componentType)
        self.dtype = decode_dtype(type, componentType)
        self._view = buffer._get_view(target, byteStride)
        self._view._add_accessor(self)
        self.count = count
        self.type = type
        self.name = name
        self.componentType = componentType
        self.normalized = normalized
        self.max = max
        self.min = min
        self.data = []

    def _add_data(self, data: AttributeDataSequence):
        self.data.extend(data)
    
    def _add_data_item(self, data: AttributeDataItem):
        self.data.append(data)
    
    def _do_compile(self, builder: BuilderProtocol, scope: _Scope, phase: Phase):
        match phase:
            case Phase.COLLECT:
                builder._accessors.add(self)
                return []
            case Phase.ENUMERATE:
                pass
            case Phase.SIZES:
                (
                    self.componentCount,
                    self.componentSize,
                    self.byteStride,
                    self.dtype,
                    self.bufferType
                ) = decode_type(self.type, self.componentType)
                return len(self.data) * self.byteStride
            case Phase.OFFSETS:
                self._view.compile(builder, scope, phase)
                self.__memory = self._view._memory(self.byteOffset, len(self))
            case Phase.BUILD:
                data = np.array(self.data, self.dtype)
                if len(self.data) == 0:
                    min_axis = max_axis = 0
                else:
                    min_axis = self.min or data.min(axis=0)
                    max_axis = self.max or data.max(axis=0)
                if isinstance(min_axis, Iterable):
                    min_axis = [float(v) for v in min_axis]
                else:
                    min_axis = [float(min_axis)] * self.componentCount
                if isinstance(max_axis, Iterable):
                    max_axis = [float(v) for v in max_axis]
                else:
                    max_axis = [float(max_axis)] * self.componentCount
                self.__memory[:] = data.tobytes()
                return gltf.Accessor(
                    bufferView=self._view.index,
                    count=self.count,
                    type=self.type,
                    componentType=self.componentType,
                    name=self.name,
                    byteOffset=self.byteOffset,
                    normalized=self.normalized,
                    max=max_axis,
                    min=min_axis,
                )
                 