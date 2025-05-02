'''
Builder representation of a mesh to be compiled.
'''

from collections.abc import Iterable, Sequence
from typing import Optional, Self, cast, overload, TYPE_CHECKING

import pygltflib as gltf

from gltf_builder.compiler import  _CompileState, _DoCompileReturn
from gltf_builder.core_types import (
    ExtensionsData, ExtrasData, Phase, PrimitiveMode,
)
from gltf_builder.attribute_types import (
    AttributeDataIterable, PointSpec, Vector3Spec,
    TangentSpec, ColorSpec, UvSpec, color, point, tangent, uv, vector3,
)
from gltf_builder.elements import BMesh, BPrimitive
from gltf_builder.primitives import _Primitive
from gltf_builder.utils import std_repr
from gltf_builder.vertices import Vertex
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState

class _MeshState(_CompileState[gltf.Mesh, '_MeshState', '_Mesh']):
    '''
    State for the compilation of a mesh.
    '''
    pass

class _Mesh(BMesh):
    '''
    Implementation class for `BMesh`.
    '''

    @classmethod
    def state_type(cls):
        return _MeshState


    def __init__(self,
                 name: str='', /,
                 primitives: Iterable[BPrimitive]=(),
                 weights: Iterable[float]=(),
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
            ):
        super().__init__(name, extras, extensions)
        self.primitives = list(primitives)
        self.weights = list(weights or ())


    def clone(self,
                name: str='', /,
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
                **_,
            ) -> Self:
        return self.__class__(
            name or self.name,
            primitives=self.primitives,
            weights=self.weights,
            extras={**self.extras, **(extras or {})},
            extensions={**self.extensions, **(extensions or {})},
        )

    @overload
    def add_primitive(self, primitive: BPrimitive, /, *,
                      extras: Optional[ExtrasData]=None,
                      extensions: Optional[ExtensionsData]=None,) -> BPrimitive: ...
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *points: PointSpec,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      extras: Optional[ExtrasData]=None,
                      extensions: Optional[ExtensionsData]=None,
                      **attribs: AttributeDataIterable
                    ) -> BPrimitive: ...
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *vertices: Vertex,
                      extras: Optional[ExtrasData]=None,
                      extensions: Optional[ExtensionsData]=None,
                    ) -> BPrimitive: ...
    def add_primitive(self, mode: PrimitiveMode|BPrimitive, /,
                      *points: PointSpec|Vertex,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      extras: Optional[ExtrasData]=None,
                      extensions: Optional[ExtensionsData]=None,
                      **attribs: AttributeDataIterable|None
                    ) -> BPrimitive:
        if isinstance(mode, BPrimitive):
            prim = mode
            prim.mesh = self
            prim.extras = dict(extras or ())
            prim.extensions = dict(extensions or ())
            return prim
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
                pointspecs =  cast(Sequence[PointSpec], points)
                prim = _Primitive(mode, (point(p) for p in pointspecs),
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
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _MeshState,
                    /
                ) -> _DoCompileReturn[gltf.Mesh]:
        match phase:
            case Phase.PRIMITIVES:
                globl.add(self)
                for i, prim in enumerate(self.primitives):
                    p_state = globl.state(prim)
                    p_state.index = i
                    prim.compile(globl, phase)
            case Phase.COLLECT:
                globl.add(self)
                return (
                    prim.compile(globl, Phase.COLLECT)
                    for prim in self.primitives
                )
            case Phase.SIZES:
                return sum(
                    prim.compile(globl, Phase.SIZES)
                    for prim in self.primitives
                )
            case Phase.BUILD:
                return gltf.Mesh(
                    name=self.name,
                    primitives=[
                        p.compile(globl, phase)
                        for p in self.primitives
                    ],
                    weights=self.weights,
                    extras=self.extras,
                    extensions=self.extensions,
                )
            case _:
                for prim in self.primitives:
                    prim.compile(globl, phase)

    def __repr__(self):
        return std_repr(self, (
            'name',
            ('primitives', len(self.primitives)),
            'weights',
        ), id=id(self))

def mesh(
    name: str='',
    primitives: Optional[Iterable[BPrimitive]]=None,
    weights: Optional[Iterable[float]]=None,
    extras: Optional[ExtrasData]=None,
    extensions: Optional[ExtensionsData]=None
) -> BMesh:
    '''
    Create a mesh with the given name and primitives.

    Additional primitives can be added to the mesh using the
    :meth:`BMesh.add_primitive`
    method.

    The mesh can be attached to one or more nodes, or added to the Builder.

    Parameters
    ----------
    name : str
        Name of the mesh.
    primitives : Iterable[BPrimitive]
        List of primitives to be added to the mesh.
    weights : Iterable[float]|None
        List of weights for the mesh.
    extras : Optional[JsonObject]
        Extra data to be added to the mesh.
    extensions : Optional[JsonObject]
        Extensions to be added to the mesh.

    Returns
    -------
    BMesh
        A mesh object with the given name and primitives.
    '''
    return _Mesh(name,
                 primitives=primitives or (),
                 weights=weights or (),
                 extras=extras,
                 extensions=extensions,
                )
