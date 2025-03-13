'''
Builder description that compiles to a BufferView
'''

from typing import Optional, Any
from collections.abc import Iterable, Mapping

import pygltflib as gltf
import numpy as np

from gltf_builder.element import (
    BBufferViewProtocol, BuilderProtocol, EMPTY_SET,
    BufferViewTarget, ComponentType, ElementType,
)
from gltf_builder.buffer import BBuffer
from gltf_builder.accessor import BAccessor
from gltf_builder.holder import Holder


class BBufferView(BBufferViewProtocol):
    __offset: int = -1 # -1 means not yet set
    
    def __init__(self, name: str='',
                 buffer: Optional[BBuffer]=None,
                 data: Optional[bytes]=None,
                 byteStride: int=0,
                 target: BufferViewTarget = BufferViewTarget.ARRAY_BUFFER,
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
                 ):
        super().__init__(name, extras, extensions)
        self.buffer = buffer
        self.target = target
        buffer.views.add(self)
        self.data = data or bytes()
        self.byteStride = byteStride
        self.accessors = Holder()
        
    @property
    def offset(self):
        if self.__offset == -1:
            raise ValueError(f'Offset has not been set for {self}')
        return self.__offset
    
    @offset.setter
    def offset(self, offset: int):
        if self.__offset != -1 and self.__offset != offset:
            raise ValueError(f'Offset has already been set for {self} old={self.__offset} new={offset}')
        self.__offset = offset
        
    def add_accessor(self,
                    type: ElementType,
                    componentType: ComponentType,
                    data: np.ndarray[tuple[int, ...], Any]|Iterable[Any],
                    normalized: bool=False,
                    min: Optional[list[float]]=None,
                    max: Optional[list[float]]=None,
                    extras: Mapping[str, Any]=EMPTY_SET,
                    extensions: Mapping[str, Any]=EMPTY_SET,
            ):
        offset = len(self.data)
        count = len(data)
        componentSize: int = 0
        if not isinstance(data, np.ndarray):
            match componentType:
                case ComponentType.BYTE:
                    data = np.array(data, np.int8)
                    componentSize = 1
                case ComponentType.UNSIGNED_BYTE:
                    data = np.array(data, np.uint8)
                    componentSize = 1
                case ComponentType.SHORT:
                    data = np.array(data, np.int16)
                    componentSize = 2
                case ComponentType.UNSIGNED_SHORT:
                    data = np.array(data, np.uint16)
                    componentSize = 2
                case ComponentType.UNSIGNED_INT:
                    data = np.array(data, np.uint32)
                    componentSize = 4
                case ComponentType.FLOAT:
                    data = np.array(data, np.float32)
                    componentSize = 4
                case _:
                    raise ValueError(f'Invalid {componentType=}')
        match type:
            case ElementType.SCALAR:
                componentCount = 1
            case ElementType.VEC2:
                componentCount = 2
            case ElementType.VEC3:
                componentCount = 3
            case ElementType.VEC4|ElementType.MAT2:
                componentCount = 4
            case ElementType.MAT3:
                componentCount = 9
            case ElementType.MAT4:
                componentCount = 16
            case _:
                raise ValueError(f'Invalid {type=}')
        stride = componentSize * componentCount
        if self.byteStride == 0:
            self.byteStride = stride
        elif self.byteStride == stride:
            pass
        else:
            raise ValueError(f'Inconsistent byteStride. old={self.byteStride}, new={stride}')
        encoded = data.flatten().tobytes()
        self.data = self.data + encoded
        accessor = BAccessor(
            view=self,
            byteOffset=offset,
            count=count,
            type=type,
            componentType=componentType,
            data=data,
            normalized=normalized,
            max=max,
            min=min,
            extras=extras,
            extensions=extensions,
        )
        self.accessors.add(accessor)
        return accessor
        
    def do_compile(self, builder: BuilderProtocol):
        for acc in self.accessors:
            acc.compile(builder)
        byteStride = (
            self.byteStride or 4
            if self.target ==  BufferViewTarget.ARRAY_BUFFER
            else None
        )
        return gltf.BufferView(
            name=self.name,
            buffer=self.buffer.index,
            byteOffset=self.offset,
            byteLength=len(self.data),
            byteStride=byteStride,
            target=self.target,
        )