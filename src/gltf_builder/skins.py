'''
Implementation of the _Skin class and its related classes for glTF Builder.
'''

from collections.abc import Iterable
from typing import Optional, TYPE_CHECKING

import pygltflib as gltf

from gltf_builder.compiler import _CompileState, _DoCompileReturn
from gltf_builder.core_types import ExtensionsData, ExtrasData, Phase
from gltf_builder.elements import BNode, BSkin
from gltf_builder.matrix import Matrix4
from gltf_builder.utils import std_repr
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState


class _SkinState(_CompileState[gltf.Skin, '_SkinState', '_Skin']):
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
                 /,
                 joints: Iterable[BNode]=(),
                 inverseBindMatrices: Optional[Matrix4]=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                ):
        super().__init__(name, extras, extensions)
        self.skeleton = skeleton
        self.joints = list(joints)
        self.inverseBindMatrices = inverseBindMatrices

    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _SkinState,
                    /) -> _DoCompileReturn[gltf.Skin]:
        match phase:
            case Phase.COLLECT:
                globl.nodes.add(self.skeleton)
                globl.nodes.add_from(self.joints)
                return [self.skeleton.compile(globl, phase)] + \
                       [j.compile(globl, phase) for j in self.joints]
            case Phase.BUILD:
                return gltf.Skin(
                    name=self.name,
                    skeleton=globl.idx(self.skeleton),
                    joints=[globl.idx(j) for j in self.joints],
                    extras=self.extras,
                    extensions=self.extensions,
                )

    def __repr__(self):
        return std_repr(self, (
            'name',
            ('skeleton', self.skeleton.name or id(self.skeleton)),
            ('joints', [j.name or id(j) for j in self.joints]),
        ))

def skin(
        skeleton: BNode,
        name: str='',
        /,
        joints: Iterable[BNode]=(),
        inverseBindMatrices: Optional[Matrix4]=None,
        extras: Optional[ExtrasData]=None,
        extensions: Optional[ExtensionsData]=None,
    ) -> BSkin:
    '''
    Create a skin object for a given node with the given attributes.

    Parameters
    ----------
    skeleton : BNode
        Skeleton of the skin.
    name : str
        Name of the skin.
    joints : Iterable[BNode]
        List of joints of the skin.
    extras : Optional[JsonObject]
        Extra data to be added to the skin.
    extensions : Optional[JsonObject]
        Extensions to be added to the skin.
    Returns
    -------
    BSkin
        A skin object with the given attributes.
    '''
    return _Skin(
        skeleton,
        name,
        joints=joints,
        inverseBindMatrices=inverseBindMatrices,
        extras=extras,
        extensions=extensions,
    )

