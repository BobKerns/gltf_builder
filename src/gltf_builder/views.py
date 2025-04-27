'''
Builder description that compiles to a BufferView
'''

from typing import Optional

import pygltflib as gltf

from gltf_builder.attribute_types import AttributeData
from gltf_builder.core_types import (
    BufferViewTarget, JsonObject, NPTypes, Phase,
)
from gltf_builder.elements import (
    BAccessor, BBuffer, BBufferView, _Scope,
)
from gltf_builder.protocols import _BuilderProtocol
from gltf_builder.compiler import _CompileState
from gltf_builder.holders import _Holder
from gltf_builder.utils import std_repr


class _BufferViewState(_CompileState[gltf.BufferView, '_BufferViewState']):
    '''
    State for the compilation of a buffer view.
    '''
    pass

class _BufferView(BBufferView):
    '''
    Implementation class for `BBufferView`.
    '''

    @classmethod
    def state_type(cls):
        return _BufferViewState

    __memory: memoryview
    
    __blob: bytes|None = None
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = self.__memory.tobytes()
        return self.__blob

    def __init__(self,
                 buffer: BBuffer,
                 name: str='',
                 byteStride: int=0,
                 target: BufferViewTarget = BufferViewTarget.ARRAY_BUFFER,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 ):
        super().__init__(name, extras, extensions)
        self.buffer = buffer
        self.target = target
        buffer.views.add(self)
        self.byteStride = byteStride
        
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState[gltf.BufferView, _BufferViewState],
                    /):
        match phase:
            case Phase.COLLECT:
                builder._accessors.add(*self.accessors)
                return [acc.compile(builder, scope, phase)
                        for acc in self.accessors]
            case Phase.SIZES:
                self.byteStride = (
                    int(self.byteStride or 4)
                    if self.target ==  BufferViewTarget.ARRAY_BUFFER
                    else 0
                )
                return sum(
                    accessor.compile(builder, scope, phase)
                    for accessor in self.accessors
                )
            case Phase.OFFSETS:
                end = self.byteOffset + len(self)
                buf_memview = memoryview(self.buffer.bytearray) 
                self.__memory = buf_memview[self.byteOffset:end]
                offset = 0
                for acc in self.accessors:
                    acc.byteOffset = offset
                    offset +=  len(acc)
                    acc.compile(builder, scope, phase)
                return end
            case Phase.BUILD:
                for acc in self.accessors:
                    acc.compile(builder, scope, Phase.BUILD)
                return gltf.BufferView(
                    name=self.name,
                    buffer=self.buffer._index,
                    byteOffset=self.byteOffset,
                    byteLength=len(self),
                    byteStride=self.byteStride or None,
                    target=self.target,
                )
            case _: pass

    def __repr__(self):
        return std_repr(self, (
                'name',
                ('buffer', self.buffer.name or id(self.buffer)),
                'byteStride',
                'target',
            ),
            cls='view')