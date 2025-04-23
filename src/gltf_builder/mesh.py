'''
Builder representation of a mesh to be compiled.
'''

from collections.abc import Iterable, Sequence
from typing import Optional, cast, overload

import pygltflib as gltf

from gltf_builder.compile import  DoCompileReturn
from gltf_builder.core_types import (
    JsonObject, Phase, PrimitiveMode,
)
from gltf_builder.attribute_types import (
    AttributeDataIterable, PointSpec, Vector3Spec,
    TangentSpec, ColorSpec, UvSpec, color, tangent, uv, vector3,
)
from gltf_builder.protocols import _BuilderProtocol
from gltf_builder.element import BMesh, BPrimitive, _Scope
from gltf_builder.primitives import _Primitive
from gltf_builder.vertices import Vertex


class _Mesh(BMesh):
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
    
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *points: PointSpec,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      extras: Optional[JsonObject]=None,
                      extensions: Optional[JsonObject]=None,
                      **attribs: AttributeDataIterable
                    ) -> _Primitive: ...
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *vertices: Vertex,
                      extras: Optional[JsonObject]=None,
                      extensions: Optional[JsonObject]=None,
                    ) -> _Primitive: ...
    def add_primitive(self, mode: PrimitiveMode, /,
                      *points: PointSpec|Vertex,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      extras: Optional[JsonObject]=None,
                      extensions: Optional[JsonObject]=None,
                      **attribs: AttributeDataIterable|None
                    ) -> _Primitive:
        match points[0]:
            case Vertex():
                vertices = cast(Sequence[Vertex], points)
                attrs = {
                    k: [v[k] for v in vertices]
                    for k in vertices[0]
                }
                prim = _Primitive(mode,
                                    mesh=self,
                                    extras=extras,
                                    extensions=extensions,
                                    **cast(dict[str, AttributeDataIterable],attrs),
                                )
            case _:
                prim = _Primitive(mode, cast(Sequence[PointSpec], points),
                                NORMAL=[vector3(n) for n in NORMAL or ()],
                                TANGENT=[tangent(t) for t in TANGENT or ()],
                                TEXCOORD_0=[uv(p) for p in TEXCOORD_0 or ()],
                                TEXCOORD_1=[uv(p) for p in TEXCOORD_1 or ()],
                                COLOR_0=[color(c) for c in COLOR_0 or ()],
                                extras=extras,
                                extensions=extensions,
                                mesh=self,
                                **attribs)
        self.primitives.append(prim)
        return prim
    
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase
                ) -> DoCompileReturn[gltf.Mesh]:
        match phase:
            case Phase.PRIMITIVES:
                builder.meshes.add(self)
                for i, prim in enumerate(self.primitives):
                    prim._index = i
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
