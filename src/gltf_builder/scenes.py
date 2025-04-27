'''
Scene definitions for glTF
'''

from collections.abc import Iterable
from typing import Any, Optional, TYPE_CHECKING

import pygltflib as gltf

from gltf_builder.compiler import _Scope, _CompileState
from gltf_builder.core_types import JsonObject, Phase
from gltf_builder.elements import BNode, BScene
from gltf_builder.utils import std_repr
if TYPE_CHECKING:
    from gltf_builder.global_state import _GlobalState


class _SceneState(_CompileState[gltf.Scene, '_SceneState']):
    '''
    State for the compilation of a scene.
    '''
    pass


class _Scene(BScene):
    '''
    Implementation class for `BScene`.
    '''

    @classmethod
    def state_type(cls):
        return _SceneState

    def __init__(self,
                 name: str='', /,
                 nodes: Iterable[BNode]=(),
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                ):
        super().__init__(name,
                       extras=extras,
                       extensions=extensions,
                       )
        self.nodes = list(nodes)

    def _clone_attributes(self) -> dict[str, Any]:
        return dict(
            nodes=self.nodes,
        )

    def _do_compile(self,
                    gbl: '_GlobalState',
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState[gltf.Scene, _SceneState],
                    /):
        match phase:
            case Phase.COLLECT:
                for n in self.nodes:
                    gbl.nodes.add(n)
                return [n.compile(gbl, scope, phase) for n in self.nodes]
            case Phase.BUILD:
                return gltf.Scene(
                    nodes=[gbl.idx(n) for n in self.nodes],
                    extras=self.extras,
                    extensions=self.extensions,
                )
            case _:
                for n in self.nodes:
                    n.compile(gbl, scope, phase)

    def __repr__(self):
        return std_repr(self, (
            'name',
            ('nodes', [n.name or id(n) for n in self.nodes]),
        ))

def scene(name: str='', /,
          *nodes: BNode,
          extras: Optional[JsonObject]=None,
          extensions: Optional[JsonObject]=None,
          ):
    '''
    Create a scene with the given name and nodes.

    Parameters
    ----------
    name : str
        Name of the scene.
    nodes : Iterable[BNode]
        List of nodes in the scene.
    extras : Optional[JsonObject]
        Optional dictionary of extra data to be attached to the scene.
    extensions : Optional[JsonObject]
        Optional dictionary of extensions to be attached to the scene.
    Returns
    -------
    BScene
        The scene object.
    '''
    return _Scene(name,
                  nodes=nodes,
                  extras=extras,
                  extensions=extensions,
                  )