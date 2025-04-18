'''
Builder description that compiles to a BufferView
'''

from typing import Optional

import pygltflib as gltf

from gltf_builder.core_types import (
    BufferViewTarget, JsonObject, NPTypes, Phase,
)
from gltf_builder.element import (
    BAccessor, BBuffer, BBufferView, Scope_,
)
from gltf_builder.protocols import BType, BuilderProtocol
from gltf_builder.holder import Holder_


class BaseBufferVieW_(BBufferView):
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
        self.accessors = Holder_(type_=BAccessor[NPTypes, BType],)        
    
    def add_accessor(self, acc: BAccessor[NPTypes, BType]) -> None:
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
        
    def _do_compile(self, builder: BuilderProtocol, scope: Scope_, phase: Phase):
        match phase:
            case Phase.COLLECT:
                builder.accessors_.add(*self.accessors)
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
                    buffer=self.buffer.index,
                    byteOffset=self.byteOffset,
                    byteLength=len(self),
                    byteStride=self.byteStride,
                    target=self.target,
                )
            case _: pass
