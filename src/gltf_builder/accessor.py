'''
Builder representation of a glTF Accessor
'''

from typing import Optional, Any
from collections.abc import Mapping, Iterable

import pygltflib as gltf
import numpy as np

from gltf_builder.element import (
    BAccessor, BuilderProtocol, BBufferView, EMPTY_SET,
    ElementType,
)


class _Accessor(BAccessor):    
    def __init__(self, /,
                 view: BBufferView,
                 count: int,
                 type: ElementType,
                 data: np.ndarray[tuple[int, ...], Any],
                 name: str='',
                 byteOffset: int=0,
                 componentType: int=0,
                 normalized: bool=False,
                 max: Optional[list[float]]=None,
                 min: Optional[list[float]]=None,
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
    ):
        super().__init__(name, extras, extensions)
        self.view = view
        self.count = count
        self.type = type
        self.data = data
        self.byteOffset = byteOffset
        self.componentType = componentType
        self.normalized = normalized
        self.max = max
        self.min = min
    
    def do_compile(self, builder: BuilderProtocol):
        builder.accessors.add(self)
        min_axis = self.min or self.data.min(axis=0)
        max_axis = self.max or self.data.max(axis=0)
        if isinstance(min_axis, Iterable):
            min_axis = [float(v) for v in min_axis]
        else:
            min_axis = [float(min_axis)]
        if isinstance(max_axis, Iterable):
            max_axis = [float(v) for v in max_axis]
        else:
            max_axis = [float(max_axis)]
        return gltf.Accessor(
            bufferView=self.view.index,
            count=self.count,
            type=self.type,
            componentType=self.componentType,
            byteOffset=self.byteOffset,
            normalized=self.normalized,
            max=max_axis,
            min=min_axis,
        )
                 