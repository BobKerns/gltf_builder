'''
Builder representation of a mesh to be compiled.
'''

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import pygltflib as gltf

from gltf_builder.element import (
    BuilderProtocol, BMesh, EMPTY_SET,
    Point, Vector3, Vector4,
)
from gltf_builder.primitives import _Primitive, PrimitiveMode


class _Mesh(BMesh):
    def __init__(self, /,
                 name='',
                 primitives: Iterable[_Primitive]=(),
                 weights: Iterable[float]=(),
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
            ):
        super().__init__(name, extras, extensions)
        self.primitives = list(primitives)
        self.weights = list(weights)
        
    def add_primitive(self, mode: PrimitiveMode,
                      *points: Point,
                      NORMAL: Optional[Iterable[Vector3]]=None,
                      TANGENT: Optional[Iterable[Vector4]]=None,
                      TEXCOORD_0: Optional[Iterable[Vector3]]=None,
                      TEXCOORD_1: Optional[Iterable[Vector3]]=None,
                      COLOR_0: Optional[Iterable[Vector4]]=None,
                      JOINTS_0: Optional[Iterable[Vector4]]=None,
                      WEIGHTS_0: Optional[Iterable[Vector4]]=None,
                      extras: Mapping[str, Any]=EMPTY_SET,
                      extensions: Mapping[str, Any]=EMPTY_SET,
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
                          **attribs)
        self.primitives.append(prim)
        return prim
    
    def do_compile(self, builder: BuilderProtocol):
        builder.meshes.add(self)
        return gltf.Mesh(
            name=self.name,
            primitives=[
                p.compile(builder)
                for p in self.primitives
            ]
        )
