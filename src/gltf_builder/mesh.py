'''
Builder representation of a mesh to be compiled.
'''

from collections.abc import Iterable
from typing import Optional

import pygltflib as gltf

from gltf_builder.compile import  DoCompileReturn
from gltf_builder.core_types import (
    JsonObject, Phase, PrimitiveMode,
)
from gltf_builder.attribute_types import (
    PointSpec, Vector3Spec, JointSpec, WeightSpec,
    TangentSpec, ColorSpec, UvSpec, AttributeDataItem,
)
from gltf_builder.protocols import BuilderProtocol
from gltf_builder.element import BMesh, BPrimitive, Scope_
from gltf_builder.primitives import Primitive_


class Mesh_(BMesh):
    indicies: Optional[int]
    __detatched: bool
    @property
    def detached(self):
        return self.__detatched
    def __init__(self, /,
                 name: str='',
                 primitives: Iterable[BPrimitive]=(),
                 weights: Iterable[float]|None=(),
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 detached: bool=False,
            ):
        super().__init__(name, extras, extensions)
        self.primitives = list(primitives)
        if weights is None:
            self.weights = []
        else:
            self.weights = list(weights)
        self.__detached = detached
        
    def add_primitive(self, mode: PrimitiveMode,
                      *points: PointSpec,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      JOINTS_0: Optional[Iterable[JointSpec]]=None,
                      WEIGHTS_0: Optional[Iterable[WeightSpec]]=None,
                      extras: Optional[JsonObject]=None,
                      extensions: Optional[JsonObject]=None,
                      **attribs: Iterable[AttributeDataItem]
                    ) -> Primitive_:
        prim = Primitive_(mode, points,
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
    
    def _do_compile(self,
                    builder: BuilderProtocol,
                    scope: Scope_,
                    phase: Phase
                ) -> DoCompileReturn[gltf.Mesh]:
        match phase:
            case Phase.PRIMITIVES:
                builder.meshes.add(self)
                for i, prim in enumerate(self.primitives):
                    prim.index = i
                    prim.compile(builder, scope, phase)
            case Phase.COLLECT:
                builder.meshes.add(self)
                return (
                    prim.compile(builder, scope, Phase.COLLECT)
                        for prim in self.primitives
                    )
            case Phase.SIZES:
                return sum(
                    prim.compile(builder, scope, Phase.SIZES)
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
