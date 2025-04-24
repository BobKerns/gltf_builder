'''
Scene definitions for glTF
'''

from collections.abc import Iterable
from typing import Any, Optional

import pygltflib as gltf

from gltf_builder.compile import _Scope
from gltf_builder.core_types import JsonObject, Phase
from gltf_builder.elements import BNode, BScene
from gltf_builder.protocols import _BuilderProtocol


class _Scene(BScene):
    '''
    Implementation class for `BScene`.
    '''

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
    
    def _do_compile(self, builder: _BuilderProtocol, scope: _Scope, phase: Phase):
        match phase:
            case Phase.COLLECT:
                for n in self.nodes:
                    builder.nodes.add(n)
                return [n.compile(builder, scope, phase) for n in self.nodes]
            case Phase.BUILD:
                return gltf.Scene(
                    nodes=[n._index for n in self.nodes],
                    extras=self.extras,
                    extensions=self.extensions,
                )
            case _:
                for n in self.nodes:
                    n.compile(builder, scope, phase)

def scene(name: str='', /,
          *nodes: BNode,
          extras: Optional[JsonObject]=None,
          extensions: Optional[JsonObject]=None,
          ):
    return _Scene(name,
                  nodes=nodes,
                  extras=extras,
                  extensions=extensions,
                  )