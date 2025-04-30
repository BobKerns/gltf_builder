'''
Builder description that compiles to a BufferView
'''

from typing import Optional, TYPE_CHECKING

import pygltflib as gltf

from gltf_builder.accessors import _AccessorState
from gltf_builder.attribute_types import AttributeData
from gltf_builder.core_types import (
    BufferViewTarget, ExtensionsData, ExtrasData, NPTypes, Phase,
)
from gltf_builder.elements import (
    BAccessor, BBuffer, BBufferView,
)
from gltf_builder.compiler import _CompileStateBinary
from gltf_builder.holders import _Holder
from gltf_builder.utils import std_repr
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState


class _BufferViewState(_CompileStateBinary[gltf.BufferView, '_BufferViewState', '_BufferView']):
    '''
    State for the compilation of a buffer view.
    '''
    memory: memoryview
    accessors: _Holder['BAccessor[NPTypes, AttributeData]']

    __blob: bytes|None = None
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = self.memory.tobytes()
        return self.__blob

    def __init__(self,
                 view: '_BufferView',
                 name: str='',
                 /,
                 ) -> None:
        super().__init__(view, name,
                         byteOffset=None,)
        self.memory = memoryview(bytearray())
        self.accessors = _Holder(type_=BAccessor[NPTypes, AttributeData],)


    def add_accessor(self, acc: BAccessor[NPTypes, AttributeData]) -> None:
        '''
        Add an accessor to the buffer view
        '''
        self.accessors.add(acc)


class _BufferView(BBufferView):
    '''
    Implementation class for `BBufferView`.
    '''

    @classmethod
    def state_type(cls):
        return _BufferViewState


    def __init__(self,
                 buffer: BBuffer,
                 name: str='',
                 byteStride: int=0,
                 target: BufferViewTarget = BufferViewTarget.ARRAY_BUFFER,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 ):
        super().__init__(name, extras, extensions)
        self.buffer = buffer
        self.target = target
        self.byteStride = byteStride

    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _BufferViewState,
                    /):
        match phase:
            case Phase.COLLECT:
                globl.accessors.add(*state.accessors)
                bstate = globl.state(self.buffer)
                bstate.add_view(self)
                return [acc.compile(globl, phase)
                        for acc in state.accessors]
            case Phase.SIZES:
                self.byteStride = (
                    int(self.byteStride or 4)
                    if self.target ==  BufferViewTarget.ARRAY_BUFFER
                    else 0
                )
                return sum(
                    accessor.compile(globl, phase)
                    for accessor in state.accessors
                )
            case Phase.OFFSETS:
                end = state.byteOffset + len(state)
                bstate = globl.state(self.buffer)
                buf_memview = memoryview(bstate._bytearray)
                state.memory = buf_memview[state.byteOffset:end]
                offset = 0
                for acc in state.accessors:
                    astate = globl.state(acc)
                    assert isinstance(astate, _AccessorState )
                    astate.byteOffset = offset
                    a_state = globl.state(acc)
                    offset +=  len(a_state)
                    acc.compile(globl, phase)
                return end
            case Phase.BUILD:
                for acc in state.accessors:
                    acc.compile(globl, Phase.BUILD)
                return gltf.BufferView(
                    name=self.name,
                    buffer=globl.idx(self.buffer),
                    byteOffset=state.byteOffset,
                    byteLength=len(state),
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