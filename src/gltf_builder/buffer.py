'''
Builder representation of a glTF Buffer
'''

from collections.abc import Iterable, Mapping
from typing import Any
import array # type: ignore

import pygltflib as gltf
import numpy as np

from gltf_builder.element import (
    BBuffer, BBufferView, BuilderProtocol, EMPTY_MAP,
)
from gltf_builder.holder import Holder


class _Buffer(BBuffer):
    __array: np.array
    __blob: bytes|None = None
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = self.__array.tobytes()
        return self.__blob
    
    views: Holder[BBufferView]
    
    def __init__(self,
                 name: str='',
                 views: Iterable[BBufferView]=(),
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
                 ):
        super().__init__(name, extras, extensions)
        self.__array = array.array('B')
        self.views = Holder(*views)

    def extend(self, data: bytes|np.typing.NDArray) -> None:
        self.__array.extend(data)
        self.__blob = None

    def do_compile(self, builder: BuilderProtocol):
        for view in self.views:
            view.compile(builder)
        namespec = {
            'gltf_builder:name': self.name,
        } if self.name else {}
        extras = self.extras or {}
        b = gltf.Buffer(
            byteLength=len(self.blob),
            extras={
                **extras,
                **namespec,
            },
            extensions=self.extensions,
            )
        return b
    
    def __len__(self) -> int:
        return len(self.__array)
    
    