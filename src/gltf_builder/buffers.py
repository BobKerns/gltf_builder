'''
Builder representation of a glTF Buffer
'''

from collections.abc import Iterable
from typing import Literal, Optional, overload

import pygltflib as gltf

from gltf_builder.compiler import _GLTF, _STATE, _Collected, _CompileState, _DoCompileReturn
from gltf_builder.core_types import (
    JsonObject, Phase, BufferViewTarget, ScopeName,
)
from gltf_builder.protocols import _BufferViewKey, _BuilderProtocol
from gltf_builder.elements import (
    BBuffer, BBufferView,
    _Scope, Element,
)
from gltf_builder.holders import _Holder
from gltf_builder.views import _BufferView

class _BufferState(_CompileState[gltf.Buffer, '_BufferState']):
    '''
    State for the compilation of a buffer.
    '''
    bytearray: bytearray
    __blob: bytes|None = None
    _views: dict[_BufferViewKey, BBufferView]
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = bytes(self.bytearray)
        return self.__blob


    def __init__(self,
                 buffer: '_Buffer',
                 name: str='',
                 /,
                 ) -> None:
        super().__init__(name, buffer)
        self._views = {}


class _Buffer(BBuffer):
    '''
    Implementation class for `BBuffer`.
    '''
    
    @classmethod
    def state_type(cls):
        return _BufferState
    
    _scope_name = ScopeName.BUFFER
    __buffer: bytearray
    @property
    def bytearray(self) -> bytearray:
        return self.__buffer

    __blob: bytes|None = None
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = bytes(self.bytearray)
        return self.__blob
    
    views: _Holder[BBufferView]

    _views: dict[_BufferViewKey, BBufferView]
    
    def __init__(self,
                 builder: _BuilderProtocol,
                 /,
                 name: str='',
                 views: Iterable[BBufferView]=(),
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 is_accessor_scope: bool=False,
                 is_view_scope: bool=False,
                 ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions)
        _Scope.__init__(self,
                        builder=builder,
                        buffer=self,
                        is_accessor_scope=is_accessor_scope,
                        is_view_scope=is_view_scope,
                    )
        self.__buffer = bytearray()
        self.views = _Holder(BBufferView, *views)
        self._views = {}

    @overload
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Literal[Phase.COLLECT],
                    state: _CompileState,
                    /
                ) -> Iterable[_Collected]: ...
    @overload
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState,
                    /
                ) -> _DoCompileReturn[gltf.Buffer]: ...
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState[gltf.Buffer, _BufferState],
                    /
                ) -> _DoCompileReturn[gltf.Buffer]:
        def _compile1(elt: Element[_GLTF, _STATE]):
            return elt.compile(builder, scope, phase)
        def _compile_views():
            for view in self.views:
                _compile1(view)
        match phase:
            case Phase.COLLECT:
                builder._views.add(*self.views)
                return (
                    view.compile(builder, scope, Phase.COLLECT)
                    for view in self.views
                )
            case Phase.SIZES:
                bytelen = sum(
                    view.compile(builder, scope, Phase.SIZES)
                    for view in self.views
                )
                self.__buffer = self.__buffer.zfill(bytelen)
                return bytelen
            case Phase.OFFSETS:
                offset = 0
                for view in self.views:
                    view.byteOffset = offset
                    _compile1(view)
                    offset += len(view)
            case Phase.BUILD:
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
                _compile_views()

    def create_view(self,
                  target: BufferViewTarget,
                  /, *,
                  byteStride: int=0,
                  name: str='',
                  extras:  Optional[JsonObject]=None,
                  extensions:  Optional[JsonObject]=None,
                ) -> BBufferView:
        '''
        Get a compatible buffer view. Specifying a name permits the use of distinct views
        for the same target and byteStride for possible optimizations.
        '''
        key = _BufferViewKey(self, target, byteStride, name)
        view = self._views.get(key)
        if view is None:
            view = _BufferView(self, name, byteStride, target,
                                    extras=extras,
                                    extensions=extensions)
            self._views[key] = view
            self.views.add(view)
            return view
        else:
            return view
    
    def __len__(self) -> int:
        return len(self.__buffer)
    