'''
Share and track vertex data.
'''

from collections.abc import Iterable, Sequence
from typing import Any, Optional, cast, overload

import numpy as np

from gltf_builder.attribute_types import (
    AttributeData, AttributeDataSpec,
    ColorSpec, JointMap, JointSpec, NormalSpec, PointSpec, TangentSpec, UvSpec,
    Vector3, Point, Tangent_, UvPoint, Color, Joint, Weight, WeightSpec,
    color, joint, joints, point, tangent, uv, vector3, weight,
)
from gltf_builder.core_types import Scalar

class Vertex:
    '''
    A vertex with position, normal, and texture coordinates and other attributes.
    '''
   # __slots__ = (
   #     'POSITION', 'NORMAL', 'TEXCOORD_0', 'TEXCOORD_1',
   #     'TANGENT', 'COLOR_0', 'attributes',
   #)
    POSITION: Point
    NORMAL: Optional[Vector3]
    TEXCOORD_0: Optional[UvPoint]
    TEXCOORD_1: Optional[UvPoint]
    TANGENT: Optional[Tangent_]
    COLOR_0: Optional[Color]
    attributes: dict[str, AttributeData]

    def __init__(self,
                 POSITION: Point,
                 NORMAL: Optional[Vector3]=None,
                 TEXCOORD_0: Optional[UvPoint]=None,
                 TEXCOORD_1: Optional[UvPoint]=None,
                 TANGENT: Optional[Tangent_]=None,
                 COLOR_0: Optional[Color]=None,
                 **attribs: AttributeData):
        self.POSITION = POSITION
        if NORMAL or True:
            self.NORMAL = NORMAL
        if TEXCOORD_0 or True:
            self.TEXCOORD_0 = TEXCOORD_0
        if TEXCOORD_1 or True:
            self.TEXCOORD_1 = TEXCOORD_1
        if TANGENT or True:
           self.TANGENT = TANGENT
        if COLOR_0 or True:
            self.COLOR_0 = COLOR_0
        if attribs or True:
            self.attributes = attribs

    def __iter__(self):
        for k in ('POSITION', 'NORMAL', 'TEXCOORD_0', 'TEXCOORD_1',
                    'TANGENT', 'COLOR_0'):
            v = getattr(self, k, None)
            if v is not None:
                yield k
        yield from  self.attributes

    def __getitem__(self, key: str) -> AttributeData:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is None:
                raise KeyError(f'{key} not found in vertex attributes')
            return val
        return self.attributes[key]

    def __repr__(self):
        x, y, z = self.POSITION
        def val(v: AttributeData) -> str:
            def s(x: Any):
                if isinstance(x, (float, np.floating)):
                    return f'{x:.3f}'
                return str(x)
            if isinstance(v, (tuple, np.ndarray)):
                return f"({','.join(s(x) for x in v)})"
            return s(v)
        return (f'@<{x:.3f}, {y:.3f}, {z:.3f}>(' +
                ', '.join(
                        f'{k}={val(self[k])}'
                        for k in self
                        if k != 'POSITION'
                ) + ')')
    

@overload
def vertex(*,
            POSITION: Optional[PointSpec]=None,
           NORMAL: Optional[NormalSpec]=None,
           TEXCOORD_0: Optional[UvSpec]=None,
           TEXCOORD_1: Optional[UvSpec]=None,
           TANGENT: Optional[TangentSpec]=None,
           COLOR_0: Optional[ColorSpec]=None,
           JOINTS: Optional[JointSpec|JointMap]=None,
           WEIGHTS: Optional[WeightSpec]=None,
           **attribs: AttributeData) -> Vertex: ...
@overload
def vertex(x: PointSpec, /, *,
           NORMAL: Optional[NormalSpec]=None,
           TEXCOORD_0: Optional[UvSpec]=None,
           TEXCOORD_1: Optional[UvSpec]=None,
           TANGENT: Optional[TangentSpec]=None,
           COLOR_0: Optional[ColorSpec]=None,
           JOINTS: Optional[JointSpec|JointMap]=None,
           WEIGHTS: Optional[WeightSpec]=None,
           **attribs: AttributeData) -> Vertex: ...
@overload
def vertex(x: Optional[Scalar]=None, y: Optional[Scalar]=None, z: Optional[Scalar]=None, /, *,
           NORMAL: Optional[NormalSpec]=None,
           TEXCOORD_0: Optional[UvSpec]=None,
           TEXCOORD_1: Optional[UvSpec]=None,
           TANGENT: Optional[TangentSpec]=None,
           COLOR_0: Optional[ColorSpec]=None,
           JOINTS: Optional[JointSpec|JointMap]=None,
           WEIGHTS: Optional[WeightSpec]=None,
           **attribs: AttributeData) -> Vertex: ...
def vertex(x: Optional[Scalar|PointSpec]=None, y: Optional[Scalar]=None, z: Optional[Scalar]=None, /, *,
            NORMAL: Optional[NormalSpec]=None,
            TEXCOORD_0: Optional[UvSpec]=None,
            TEXCOORD_1: Optional[UvSpec]=None,
            TANGENT: Optional[TangentSpec]=None,
            COLOR_0: Optional[ColorSpec]=None,
            JOINTS: Optional[JointSpec|JointMap]=None,
            WEIGHTS: Optional[WeightSpec]=None,
            **attribs: AttributeDataSpec|AttributeData|None) -> Vertex:
    '''
    Create a vertex with position, normal, and texture coordinates and other attributes.

    Parameters
    ----------
    x, y, z: float
        The position of the vertex. If `x` is a tuple or `Point`, it is used as the position.
    NORMAL: Vector3Spec
        The normal vector of the vertex.
    TEXCOORD_0: UvSpec
        The first texture coordinate of the vertex.
    TEXCOORD_1: UvSpec
        The second texture coordinate of the vertex.
    TANGENT: TangentSpec
        The tangent vector of the vertex.
    COLOR_0: ColorSpec
        The color of the vertex.
    JOINTS: JointSpec|JointMap
        The joint indices of the vertex, or a mapping of joint indices to weights.
    WEIGHTS: WeightSpec
        The weights of the vertex.
    attribs: AttributeDataItem
        Additional attributes of the vertex.
    Returns
    -------
    Vertex
        A vertex with the given attributes.
    '''
    POSITION = cast(PointSpec|None, attribs.get('POSITION', None))
    if POSITION is not None:
        POSITION = point(POSITION)
    elif isinstance(x, (float, int, np.floating, np.integer)):
        if y is None or z is None:
            raise ValueError('x, y, z must be given as a tuple')
        POSITION = point(x, y, z)
    elif x:
        POSITION = point(x)
    else:
        raise TypeError('POSITION must be given')
    NORMAL = vector3(NORMAL) if NORMAL else None
    TEXCOORD_0 = uv(TEXCOORD_0) if TEXCOORD_0 else None
    TEXCOORD_1 = uv(TEXCOORD_1) if TEXCOORD_1 else None
    TANGENT = tangent(TANGENT) if TANGENT else None
    COLOR_0 = color(COLOR_0) if COLOR_0 else None
    def decompose(j: JointSpec, w: WeightSpec):
        joint_map = {
            f'JOINTS_{i}': j
            for i,j in enumerate(joint(*j) if JOINTS else ())
        }
        weight_map = {
            f'WEIGHTS_{i}': w
            for i,w in enumerate(weight(*w) if WEIGHTS else ())
        }
        return joint_map, weight_map
    def compose(j: Sequence[Joint], w: Sequence[Weight]):
        joint_map = {
            f'JOINTS_{i}': ji
            for i,ji in enumerate(j)
        }
        weight_map = {
            f'WEIGHTS_{i}': wi
            for i,wi in enumerate(w)
        }
        return joint_map, weight_map
    
    match JOINTS:
        case dict():
            if WEIGHTS is not None:
                raise ValueError('JOINTS and WEIGHTS cannot be given together if JOINTS is a mapping')
            joint_list, weight_list = joints(JOINTS)
            joint_map, weight_map = compose(joint_list, weight_list)
        case Sequence():
            if WEIGHTS is None:
                joint_list, weight_list = joints(cast(JointMap, JOINTS))
                joint_map, weight_map = compose(joint_list, weight_list)
            else:
                joint_map, weight_map = decompose(cast(JointSpec,JOINTS), WEIGHTS)
        case Iterable():
            if WEIGHTS is not None:
                raise ValueError('JOINTS and WEIGHTS cannot be given together if JOINTS is a mapping')
            joint_list, weight_list = joints(JOINTS)
            joint_map, weight_map = compose(joint_list, weight_list)
        case None:
            joint_map, weight_map = {}, {}
        case _:
            raise TypeError('JOINTS must be a JointMap or a Sequence of JointSpec')
    return Vertex(POSITION,
                  NORMAL=NORMAL,
                  TEXCOORD_0=TEXCOORD_0,
                  TEXCOORD_1=TEXCOORD_1,
                  TANGENT=TANGENT,
                  COLOR_0=COLOR_0,
                  **{k:cast(AttributeData, v) for k, v in attribs.items() if v},
                  **joint_map,
                  **weight_map,
    )
