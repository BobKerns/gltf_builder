'''
Builder representation of a glTF Buffer
'''

from collections.abc import Iterable, Mapping
from typing import Any

import pygltflib as gltf

from gltf_builder.element import (
    BBuffer, BBufferView, BuilderProtocol, EMPTY_SET,
)
from gltf_builder.holder import Holder


class _Buffer(BBuffer):
    __data: bytes
    views: Holder[BBufferView]
    
    @property
    def data(self):
        return self.__data
    
    def __init__(self,
                 name: str='',
                 views: Iterable[BBufferView]=(),
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
                 ):
        super().__init__(name, extras, extensions)
        self.__data = bytes(())
        self.views = Holder(*views)
    
    def do_compile(self, builder: BuilderProtocol):
        for view in self.views:
            view.offset = len(self.__data)
            view.compile(builder)
            self.__data = self.__data + view.data
        return gltf.Buffer(
            byteLength=len(self.__data),
            extras=self.extras,
            extensions=self.extensions,
            )
    
    