'''
Share and track vertex data.
'''

from typing import Any, Optional

import numpy as np

from gltf_builder.attribute_types import (
    AttributeDataItem, Point, Vector3, Tangent_, UvPoint, Color, Joint, Weight,
)

class Vertex:
    '''
    A vertex with position, normal, and texture coordinate.
    '''
    POSITION: Point
    NORMAL: Optional[Vector3]
    TEXCOORD_0: Optional[UvPoint]
    TEXCOORD_1: Optional[UvPoint]
    TANGENT: Optional[Tangent_]
    COLOR_0: Optional[Color]
    JOINTS_0: Optional[Joint]
    WEIGHTS_0: Optional[Weight]
    attributes: dict[str, AttributeDataItem]

    def __init__(self,
                 POSITION: Point,
                 NORMAL: Optional[Vector3]=None,
                 TEXCOORD_0: Optional[UvPoint]=None,
                 TEXCOORD_1: Optional[UvPoint]=None,
                 TANGENT: Optional[Tangent_]=None,
                 COLOR_0: Optional[Color]=None,
                 JOINTS_0: Optional[Joint]=None,
                 WEIGHTS_0: Optional[Weight]=None,
                 **attribs: AttributeDataItem):
        self.POSITION = POSITION # type: ignore
        self.NORMAL = NORMAL # type: ignore
        self.TEXCOORD_0 = TEXCOORD_0 # type: ignore
        self.TEXCOORD_1 = TEXCOORD_1 # type: ignore
        self.TANGENT = TANGENT # type: ignore
        self.COLOR_0 = COLOR_0 # type: ignore
        self.JOINTS_0 = JOINTS_0    # type: ignore
        self.WEIGHTS_0 = WEIGHTS_0 # type: ignore
        self.attributes = attribs

    def __iter__(self): # type: ignore
        for k in ('POSITION', 'NORMAL', 'TEXCOORD_0', 'TEXCOORD_1',
                    'TANGENT', 'COLOR_0', 'JOINTS_0', 'WEIGHTS_0'):
            v = getattr(self, k, None)
            if v is not None:
                yield k, v
        yield from  self.attributes.items()

    def __repr__(self):
        x, y, z = self.POSITION
        def val(v: AttributeDataItem) -> str:
            def s(x: Any):
                if isinstance(x, (float, np.floating)):
                    return f'{x:.3f}'
                return str(x)
            if isinstance(v, (tuple, np.ndarray)):
                return f"({','.join(s(x) for x in v)})"
            return s(v)
        return (f'@<{x:.3f}, {y:.3f}, {z:.3f}>(' +
                ', '.join(
                        f'{k}={val(v)}'
                        for k,v in self
                        if k != 'POSITION'
                ) + ')')

def vertex(x: float, y: float, z: float,
           NORMAL: Optional[tuple[float, float, float]]=None,
           TEXCOORD_0: Optional[tuple[float, float]]=None,
           TEXCOORD_1: Optional[tuple[float, float]]=None,
           TANGENT: Optional[tuple[float, float, float, float]]=None,
           COLOR_0: Optional[tuple[float, float, float, float]]=None,
           JOINTS_0: Optional[tuple[int, int, int, int]]=None,
           WEIGHTS_0: Optional[tuple[float, float, float, float]]=None,
           **attribs: AttributeDataItem):
           ...
