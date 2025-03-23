'''
Builder representation of a glTF Buffer
'''

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import pygltflib as gltf

from gltf_builder.element import (
    BBuffer, BBufferView, BufferViewTarget, BuilderProtocol,
    EMPTY_MAP, Phase, _Scope,
)
from gltf_builder.holder import Holder
from gltf_builder.view import _BufferView


class ViewKey(NamedTuple):
    target: BufferViewTarget
    byteStride: int
    name: str


class _Buffer(BBuffer):
    __buffer: bytearray
    @property
    def buffer(self):
        return self.__buffer

    __blob: bytes|None = None
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = self.__buffer
        return self.__blob
    
    views: Holder[BBufferView]

    _views: dict[ViewKey, BBufferView]
    
    def __init__(self,
                 name: str='',
                 views: Iterable[BBufferView]=(),
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
                 ):
        super().__init__(name, extras, extensions)
        self.__buffer = bytearray()
        self.views = Holder(*views)
        self._views = {}

    def _do_compile(self, builder: BuilderProtocol, scope: _Scope, phase: Phase):
        match phase:
            case Phase.COLLECT:
                builder._views.add(*self.views)
                return [view.compile(builder, scope, phase)
                        for view in self.views]
            case Phase.SIZES:
                bytelen = sum(view.compile(builder,scope, phase)
                                for view in self.views)
                self.__buffer = self.__buffer.zfill(bytelen)
                return bytelen
            case Phase.OFFSETS:
                offset = 0
                for view in self.views:
                    view.byteOffset = offset
                    view.compile(builder, scope, phase)
                    offset += len(view)
            case Phase.BUILD:
                for view in self.views:
                    view.compile(builder, self, phase)
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
            case _:
                for view in self.views:
                    view.compile(builder, self, phase)

    def _get_view(self, target: BufferViewTarget, byteStride: int, name='') -> BBufferView:
        '''
        Get a compatible buffer view. Specifying a name permits the use of distinct views
        for the same target and byteStride for possible optimizations.
        '''
        key = ViewKey(target, byteStride, name)
        view = self._views.get(key)
        if view is None:
            view = _BufferView(name, self, byteStride, target)
            self._views[key] = view
            self.views.add(view)
        return view
    
    def __len__(self) -> int:
        return len(self.__buffer)
    