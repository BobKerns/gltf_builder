'''
Builder description that compiles to a BufferView
'''

from typing import Optional, Any
from collections.abc import Mapping

import pygltflib as gltf

from gltf_builder.element import (
    BBuffer, BBufferView, BuilderProtocol, EMPTY_MAP, _Scope,
    BufferViewTarget, Phase,
)
from gltf_builder.accessor import _Accessor
from gltf_builder.holder import Holder


class _BufferView(BBufferView):
    __memory: memoryview
    @property
    def memory(self):
        return self.__memory
    
    __blob: bytes|None = None
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = self.__array.tobytes()
        return self.__blob

    def __init__(self, name: str='',
                 buffer: Optional[BBuffer]=None,
                 byteStride: int=0,
                 target: BufferViewTarget = BufferViewTarget.ARRAY_BUFFER,
                 extras: Mapping[str, Any]=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
                 ):
        super().__init__(name, extras, extensions)
        self.buffer = buffer
        self.target = target
        buffer.views.add(self)
        self.byteStride = byteStride
        self.accessors = Holder()        
    
    def _add_accessor(self, BAccessor: _Accessor) -> None:
        '''
        Add an accessor to the buffer view
        '''
        self.accessors.add(BAccessor)
    
    def _memory(self, offset: int, size: int) -> memoryview:
        '''
        Return a memoryview of the buffer view.se
        '''
        end = offset + size
        return self.__memory[offset:end]
        
    def _do_compile(self, builder: BuilderProtocol, scope: _Scope, phase: Phase):
        match phase:
            case Phase.COLLECT:
                builder._accessors.add(*self.accessors)
                return [acc.compile(builder, scope, phase)
                        for acc in self.accessors]
            case Phase.SIZES:
                self.byteStride = (
                    self.byteStride or 4
                    if self.target ==  BufferViewTarget.ARRAY_BUFFER
                    else None
                )
                return sum(
                    accessor.compile(builder, scope, phase)
                    for accessor in self.accessors
                )
            case Phase.OFFSETS:
                end = self.byteOffset + len(self)
                buf_memview = memoryview(self.buffer.buffer) 
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
