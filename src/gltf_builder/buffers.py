'''
Builder representation of a glTF Buffer
'''

from collections.abc import Iterable
from typing import Optional, TYPE_CHECKING

import pygltflib as gltf

from gltf_builder.compiler import (
    _GLTF, _STATE, _GlobalCompileState, _DoCompileReturn,
    ExtensionsData, ExtrasData,
)
from gltf_builder.core_types import (
    Phase, ScopeName,
)
from gltf_builder.protocols import _BufferViewKey
from gltf_builder.elements import (
    BBuffer, BBufferView, Element,
)
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState

class _BufferState(_GlobalCompileState[gltf.Buffer, _STATE, '_Buffer']):
    '''
    State for the compilation of a buffer.
    '''
    _bytearray: bytearray
    __blob: bytes|None = None
    _byteOffset: int|None = 0
    @property
    def blob(self):
        if self.__blob is None:
            self.__blob = bytes(self._bytearray)
        return self.__blob

    _views: dict[_BufferViewKey, BBufferView]
    @property
    def views(self) -> Iterable[BBufferView]:
        return self._views.values()

    def add_view(self, view: BBufferView) -> None:
        '''
        Add a view to the buffer
        '''
        key = _BufferViewKey(
            view.buffer,
            view.target,
            view.byteStride,
            view.name,
        )
        self._views[key] = view

    def __init__(self,
                 buffer: '_Buffer',
                 name: str='',
                 /,
                 ) -> None:
        self._bytearray = bytearray()
        super().__init__(buffer, name)
        self._views = {}


class _Buffer(BBuffer):
    '''
    Implementation class for `BBuffer`.
    '''
    _scope_name = ScopeName.BUFFER

    @classmethod
    def state_type(cls):
        return _BufferState

    def __init__(self,
                 name: str='',
                 /,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 is_accessor_scope: bool=False,
                 is_view_scope: bool=False,
                 ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions)

    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _BufferState,
                    /
                ) -> _DoCompileReturn[gltf.Buffer]:
        def _compile1(elt: Element[_GLTF, _STATE]):
            return elt.compile(globl, phase)
        def _compile_views():
            for view in state.views:
                _compile1(view)
        match phase:
            case Phase.COLLECT:
                globl.views.add(*state.views)
                return (
                    view.compile(globl, Phase.COLLECT)
                    for view in state.views
                )
            case Phase.SIZES:
                bytelen = sum(
                    view.compile(globl, Phase.SIZES)
                    for view in state.views
                )
                state._bytearray = state._bytearray.zfill(bytelen)
                return bytelen
            case Phase.OFFSETS:
                offset = 0
                for view in state.views:
                    vstate = globl.state(view)
                    vstate.byteOffset = offset
                    _compile1(view)
                    offset += len(vstate.memory)
            case Phase.BUILD:
                namespec = {
                    'gltf_builder:name': self.name,
                } if self.name else {}
                extras = self.extras or {}
                b = gltf.Buffer(
                    byteLength=len(state.blob),
                    extras={
                        **extras,
                        **namespec,
                    },
                    extensions=self.extensions,
                    )
                return b
            case _:
                _compile_views()

def buffer(name: str='',
           /,
           extras: Optional[ExtrasData]=None,
           extensions: Optional[ExtensionsData]=None,


           ) -> BBuffer:
    '''
    Create a new buffer.

    Parameters
    ----------
    name : str, optional
        Name of the buffer.
    extras : dict, optional
        Extra data to be stored in the buffer.
    extensions : dict, optional
        Extensions to be stored in the buffer.
    is_accessor_scope : bool, optional
        Whether the buffer is an accessor scope.
    is_view_scope : bool, optional
        Whether the buffer is a view scope.

    Returns
    -------
    BBuffer
        The created buffer.
    '''
    return _Buffer(
        name,
        extras=extras,
        extensions=extensions,
    )
