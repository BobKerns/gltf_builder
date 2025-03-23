'''
Builder representation of a mesh to be compiled.
'''

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import pygltflib as gltf

from gltf_builder.types import (
    Phase, PrimitiveMode, EMPTY_MAP,
    Point, Vector3, Vector4,
)
from gltf_builder.element import (
    BuilderProtocol, BMesh, _Scope,
)
from gltf_builder.primitives import _Primitive


class _Mesh(BMesh):
    __detatched: bool
    @property
    def detached(self):
        return self.__detatched
    def __init__(self, /,
                 name='',
                 primitives: Iterable[_Primitive]=(),
                 weights: Iterable[float]|None=(),
                 extras: Mapping[str, Any]=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
                 detached: bool=False,
            ):
        super().__init__(name, extras, extensions)
        self.primitives = list(primitives)
        self.weights = list(weights)
        self.__detached = detached
        
    def add_primitive(self, mode: PrimitiveMode,
                      *points: Point,
                      NORMAL: Optional[Iterable[Vector3]]=None,
                      TANGENT: Optional[Iterable[Vector4]]=None,
                      TEXCOORD_0: Optional[Iterable[Vector3]]=None,
                      TEXCOORD_1: Optional[Iterable[Vector3]]=None,
                      COLOR_0: Optional[Iterable[Vector4]]=None,
                      JOINTS_0: Optional[Iterable[Vector4]]=None,
                      WEIGHTS_0: Optional[Iterable[Vector4]]=None,
                      extras: Mapping[str, Any]|None=EMPTY_MAP,
                      extensions: Mapping[str, Any]|None=EMPTY_MAP,
                      **attribs: Iterable[tuple[int|float,...]]
                    ) -> _Primitive:
        prim = _Primitive(mode, points,
                          NORMAL=NORMAL,
                          TANGENT=TANGENT,
                          TEXCOORD_0=TEXCOORD_0,
                          TEXCOORD_1=TEXCOORD_1,
                          COLOR_0=COLOR_0,
                          JOINTS_0=JOINTS_0,
                          WEIGHTS_0=WEIGHTS_0,
                          extras=extras,
                          extensions=extensions,
                          mesh=self,
                          **attribs)
        self.primitives.append(prim)
        return prim
    
    def _do_compile(self, builder: BuilderProtocol, scope: _Scope, phase: Phase):
        match phase:
            case Phase.PRIMITIVES:
                builder.meshes.add(self)
                for i, prim in enumerate(self.primitives):
                    prim.index = i
                    prim.compile(builder, scope, phase)
            case Phase.COLLECT:
                builder.meshes.add(self)
                return [prim.compile(builder, scope, phase)
                        for prim in self.primitives]
            case Phase.SIZES:
                return sum(
                    prim.compile(builder, scope, phase)
                    for prim in self.primitives
                )
            case Phase.BUILD:
                return gltf.Mesh(
                    name=self.name,
                    primitives=[
                        p.compile(builder, scope, phase)
                        for p in self.primitives
                    ]
                )
            case _:
                for prim in self.primitives:
                    prim.compile(builder, scope, phase)
