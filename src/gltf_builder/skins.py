'''
Implementation of the _Skin class and its related classes for glTF Builder.
'''

from collections.abc import Iterable
from typing import Optional

import pygltflib as gltf

from gltf_builder.compiler import _CompileState, _Scope, _DoCompileReturn
from gltf_builder.core_types import JsonObject, Phase
from gltf_builder.elements import BNode, BSkin
from gltf_builder.matrix import Matrix4
from gltf_builder.protocols import _BuilderProtocol


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
                 /,
                 joints: Iterable[BNode]=(),
                 inverseBindMatrices: Optional[Matrix4]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                ):
        super().__init__(name, extras, extensions)
        self.skeleton = skeleton
        self.joints = list(joints)
        self.inverseBindMatrices = inverseBindMatrices

    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _SkinState,
                    /) -> _DoCompileReturn[gltf.Skin]:
        match phase:
            case Phase.COLLECT:
                builder.nodes.add(self.skeleton)
                for j in self.joints:
                    builder.nodes.add(j)
                return [self.skeleton.compile(builder, scope, phase)] + \
                       [j.compile(builder, scope, phase) for j in self.joints]
            case Phase.BUILD:
                return gltf.Skin(
                    name=self.name,
                    skeleton=self.skeleton._index,
                    joints=[j._index for j in self.joints],
                    extras=self.extras,
                    extensions=self.extensions,
                )

def skin(
        skeleton: BNode,
        name: str='',
        /,
        joints: Iterable[BNode]=(),
        inverseBindMatrices: Optional[Matrix4]=None,
        extras: Optional[JsonObject]=None,
        extensions: Optional[JsonObject]=None,
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

