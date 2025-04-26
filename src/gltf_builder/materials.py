'''
Implementation of the Material class and its related classes for glTF Builder.
'''

from typing import Optional

import pygltflib as gltf

from gltf_builder.compiler import _CompileState
from gltf_builder.core_types import JsonObject
from gltf_builder.elements import BMaterial


class _MaterialState(_CompileState[gltf.Material, '_MaterialState']):
    '''
    State for the compilation of a material.
    '''
    pass

class _Material(BMaterial):
    '''
    Implementation class for `BMaterial`.
    '''

    @classmethod
    def state_type(cls):
        return _MaterialState

    def __init__(self,
                 name: str='',
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                ):
        super().__init__(name, extras, extensions)