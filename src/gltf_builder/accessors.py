'''
Builder representation of a glTF Accessor
'''

from typing import Optional, cast, TYPE_CHECKING
from collections.abc import Iterable, Sequence
from pathlib import Path

import pygltflib as gltf
import numpy as np
from gltf_builder.core_types import (
    ComponentType, ElementType, ExtensionsData, ExtrasData,
    Phase, BufferViewTarget, EntityType,
)
from gltf_builder.attribute_types import (
    BTYPE, AttributeData, BTYPE_co, BType,
)
from gltf_builder.entities import (
    BAccessor, BBuffer, NP, BBufferView
)
from gltf_builder.compiler import _GlobalCompileState, _DoCompileReturn
from gltf_builder.utils import (
    decode_dtype, decode_stride, decode_type, std_repr,
)
from gltf_builder.log import GLTF_LOG
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState


LOG = GLTF_LOG.getChild(Path(__name__).stem)

class _AccessorState(_GlobalCompileState[gltf.Accessor, '_AccessorState', '_Accessor']):
    '''
    State for the compilation of an accessor.
    '''
    view: Optional['BBufferView'] = None
    memory: memoryview
    data: list[AttributeData]

    def __init__(self,
                 accessor: '_Accessor',
                 name: str='',
                 /,
                 ) -> None:
        super().__init__(accessor, name,
                         byteOffset=None,
                         )
        self.data = []

    def add_data(self, data: Sequence[BTYPE]):
        self.data.extend(data)

    def add_data_item(self, data: AttributeData):
        self.data.append(data)

    def __repr__(self):
        return std_repr(self, (
                'name',
                ('index', self._index),
                ('byteOffset', self._byteOffset),
                ('len', self._len),
                ('data', len(self.data)),
                'phase',
            ),
            id=id(self),
            )


class _Accessor(BAccessor[NP, BTYPE]):
    '''
    Implementation class for `BAccessor`.
    '''

    @classmethod
    def state_type(cls):
        return _AccessorState

    dtype: type[NP]
    btype: BType
    target: BufferViewTarget

    def __init__(self, /,
                 buffer: BBuffer,
                 count: int,
                 elementType: ElementType,
                 componentType: ComponentType,
                 dtype: type[NP],
                 btype: type[BTYPE_co],
                 name: str='',
                 normalized: bool=False,
                 max: Optional[list[float]]=None,
                 min: Optional[list[float]]=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 target: BufferViewTarget = BufferViewTarget.ARRAY_BUFFER,
    ):
        super().__init__(name=name,
                         extras=extras,
                         extensions=extensions,
                    )
        self.byteStride = decode_stride(elementType, componentType)
        self.dtype = cast(type[NP], decode_dtype(elementType, componentType))
        self.count = count
        self.elementType = elementType
        self.name = name
        self.componentType = componentType
        self.normalized = normalized
        self.max = max
        self.min = min
        self.dtype = dtype
        self.btype = btype
        self.target = target

    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _AccessorState,
                    /
                    ) -> _DoCompileReturn[gltf.Accessor]:
        match phase:
            case Phase.COLLECT:
                buffer = globl.buffer
                vname = globl._gen_name(self,
                                          scope=EntityType.BUFFER_VIEW,
                                          suffix='/view')
                state.view = globl.get_view(buffer,
                                              self.target,
                                              byteStride=self.byteStride,
                                              name=vname)
                globl.add(state.view)
                vstate = globl.state(state.view)
                vstate.add_accessor(self)
                globl.add(self)
                return [(self,())]
            case Phase.SIZES:
                (
                    self.componentCount,
                    self.componentSize,
                    self.byteStride,
                    self.dtype, # type: ignore
                    self.bufferType
                ) = decode_type(self.elementType, self.componentType)
                ldata = sum(
                    len(d) if isinstance(d, (Sequence, np.ndarray)) else 1
                    for d in state.data
                )
                return ldata * self.componentSize
            case Phase.OFFSETS:
                assert state.view is not None
                v_state = globl.state(state.view)
                start = state.byteOffset
                end = state.byteOffset + len(state)
                state.memory = v_state.memory[start:end]
                return end
            case Phase.BUILD:
                data = np.array(state.data, self.dtype)
                if len(data) == 0:
                    min_axis = max_axis = [0]
                else:
                    min_axis = data.min(axis=0)
                    max_axis = data.max(axis=0)
                if isinstance(min_axis, Iterable):
                    min_axis = [float(v) for v in min_axis]
                else:
                    min_axis = [float(min_axis)] * self.componentCount
                if isinstance(max_axis, Iterable):
                    max_axis = [float(v) for v in max_axis]
                else:
                    max_axis = [float(max_axis)] * self.componentCount
                state.memory[:] = data.tobytes()
                assert state.view is not None
                return gltf.Accessor(
                    bufferView=globl.idx(state.view),
                    count=self.count,
                    type=self.elementType,
                    componentType=self.componentType,
                    name=self.name,
                    byteOffset=state.byteOffset,
                    normalized=self.normalized,
                    max=max_axis,
                    min=min_axis,
                )
            case _: pass

    def __repr__(self):
        return std_repr(self, (
            'name',
            'byteStride',
            'normalized',
            'elementType',
            'componentType',
            'target'
        ),
        id=id(self))