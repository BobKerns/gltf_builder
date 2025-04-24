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
from gltf_builder.compile import _CompileStates
from gltf_builder.holders import _Holder


class _BufferView(BBufferView):
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
        self.accessors = _Holder(type_=BAccessor[NPTypes, AttributeData],)        
    
    def _add_accessor(self, acc: BAccessor[NPTypes, AttributeData]) -> None:
        '''
        Add an accessor to the buffer view
        '''
        self.accessors.add(acc)
    
    def memoryview(self, offset: int, size: int) -> memoryview:
        '''
        Return a memoryview of the buffer view.se
        '''
        end = offset + size
        return self.__memory[offset:end]
        
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    states: _CompileStates,
                    /):
        match phase:
            case Phase.COLLECT:
                builder._accessors.add(*self.accessors)
                return [acc.compile(builder, scope, phase, states)
                        for acc in self.accessors]
            case Phase.SIZES:
                self.byteStride = (
                    int(self.byteStride or 4)
                    if self.target ==  BufferViewTarget.ARRAY_BUFFER
                    else 0
                )
                return sum(
                    accessor.compile(builder, scope, phase, states)
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
                    acc.compile(builder, scope, phase, states)
                return end
            case Phase.BUILD:
                for acc in self.accessors:
                    acc.compile(builder, scope, Phase.BUILD, states)
                return gltf.BufferView(
                    name=self.name,
                    buffer=self.buffer._index,
                    byteOffset=self.byteOffset,
                    byteLength=len(self),
                    byteStride=self.byteStride or None,
                    target=self.target,
                )
            case _: pass
