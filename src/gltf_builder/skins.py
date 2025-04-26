'''
Implementation of the _Skin class and its related classes for glTF Builder.
'''

from collections.abc import Iterable
from typing import Optional

import pygltflib as gltf

from gltf_builder.compiler import _CompileState
from gltf_builder.core_types import JsonObject
from gltf_builder.elements import BNode, BSkin


class _SkinState(_CompileState[gltf.Skin, '_SkinState']):
    '''
    State for the compilation of a skin.
    '''
    pass


class _Skin(BSkin):
    '''
    Implementation class for `BSkin`.
    '''

    @classmethod
    def state_type(cls):
        return _SkinState

    def __init__(self,
                 skeleton: BNode,
                 name: str='',
                 joints: Iterable[BNode]=(),
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                ):
        super().__init__(name, extras, extensions)
        self.skeleton = skeleton
        self.joints = list(joints)