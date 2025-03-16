'''
Builder description that compiles to a BufferView
'''

from typing import Optional, Any
from collections.abc import Iterable, Mapping
import array # type: ignore

import pygltflib as gltf
import numpy as np

from gltf_builder.element import (
    BBufferView, BuilderProtocol, EMPTY_SET,
    BufferViewTarget, ComponentType, ElementType,
)
from gltf_builder.buffer import _Buffer
from gltf_builder.accessor import _Accessor
from gltf_builder.holder import Holder


class _BufferView(BBufferView):
    __array: array.array
    __blob: bytes|None = None
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = self.__array.tobytes()
        return self.__blob
    
    __offset: int = -1
    @property
    def offset(self) -> int:
        if self.__offset < 0:
            self.__offset = len(self.buffer)
        return self.__offset

    def __init__(self, name: str='',
                 buffer: Optional[_Buffer]=None,
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
        self.__array = array.array('B', data or ())
        self.byteStride = byteStride
        self.accessors = Holder()
        
    
    def add_accessor(self,
                    type: ElementType,
                    componentType: ComponentType,
                    data: np.ndarray[tuple[int, ...], Any]|Iterable[Any],
                    normalized: bool=False,
                    min: Optional[list[float]]=None,
                    max: Optional[list[float]]=None,
                    extras: Mapping[str, Any]=EMPTY_SET,
                    extensions: Mapping[str, Any]=EMPTY_SET,
            ) -> gltf.Accessor:
        offset = len(self)
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
        self.extend(data.flatten().tobytes())
        accessor = _Accessor(
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
        self.buffer.extend(self.blob)
        return gltf.BufferView(
            name=self.name,
            buffer=self.buffer.index,
            byteOffset=self.offset,
            byteLength=len(self),
            byteStride=byteStride,
            target=self.target,
        )

    def extend(self, data: bytes|np.typing.NDArray) -> None:
        self.__array.extend(data)
        self.__blob = None

    def __len__(self):
        return len(self.__array)
