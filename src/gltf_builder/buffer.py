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
    __blob: bytes
    @property
    def blob(self):
        return self.__blob
    
    views: Holder[BBufferView]
    
    def __init__(self,
                 name: str='',
                 views: Iterable[BBufferView]=(),
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
                 ):
        super().__init__(name, extras, extensions)
        self.__blob = bytes(())
        self.views = Holder(*views)
    
    def do_compile(self, builder: BuilderProtocol):
        for view in self.views:
            view.offset = len(self.__blob)
            view.compile(builder)
            self.__blob = self.__blob + view.data
        return gltf.Buffer(
            byteLength=len(self.__blob),
            extras=self.extras,
            extensions=self.extensions,
            )
    
    