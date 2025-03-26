'''
Share and track vertex data.
'''

from typing import NamedTuple, Optional

from gltf_builder.core_types import (
 #   ElementType, ComponentType, BufferType,
    EMPTY_MAP, _Point, _Vector3, _Tangent, _Uv, _Color, _Joint, _Weight, AttributeDataItem,
)
#from gltf_builder.utils import decode_dtype, decode_stride, decode_type

class Vertex(NamedTuple):
    '''
    A vertex with position, normal, and texture coordinate.
    '''
    POSITION: _Point
    NORMAL: Optional[_Vector3] = None
    TEXCOORD_0: Optional[_Uv] = None
    TEXCOORD_1: Optional[_Uv] = None
    TANGENT: Optional[_Tangent] = None
    COLOR_0: Optional[_Color] = None
    JOINTS_0: Optional[_Joint] = None
    WEIGHTS_0: Optional[_Weight] = None
    attributes: dict[str, tuple[int, ...]|tuple[float, ...]] = EMPTY_MAP

    def __iter__(self):
        for k in ('POSITION', 'NORMAL', 'TEXCOORD_0', 'TEXCOORD_1',
                    'TANGENT', 'COLOR_0', 'JOINTS_0', 'WEIGHTS_0'):
            v = getattr(self, k, None)
            if v is not None:
                yield k, v
        yield from  self.attributes.items()

    def __repr__(self):
        x, y, z = self.POSITION
        def val(v):
            def s(x):
                if isinstance(x, float):
                    return f'{x:.3f}'
                return str(x)
            return f"({','.join(s(x) for x in v)})"
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
