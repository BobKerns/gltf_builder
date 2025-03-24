'''
Share and track vertex data.
'''

from typing import NamedTuple, Optional

#import numpy as np

from gltf_builder.types import (
 #   ElementType, ComponentType, BufferType,
    _Point, _Vector3, _Tangent, _Uv, _Color, _Joint, _Weight
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
