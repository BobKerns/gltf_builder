'''
Definitions for GLTF primitives
'''

from typing import TypeAlias
from enum import IntEnum
from collections.abc import Sequence

import pygltflib as gltf


class PrimitiveType(IntEnum):
    POINTS = gltf.POINTS
    LINES = gltf.LINES
    LINE_LOOP = gltf.LINE_LOOP
    LINE_STRIP = gltf.LINE_STRIP
    TRIANGLES = gltf.TRIANGLES
    TRIANGLE_STRIP = gltf.TRIANGLE_STRIP
    TRIANGLE_FAN = gltf.TRIANGLE_FAN
    

Point: TypeAlias = tuple[float, float, float]


class Primitive:
    '''
    Base class for primitives
    '''
    type: PrimitiveType
    points: list[Point]
    indicies: list[int]
    
    def __init__(self, type: PrimitiveType,
                 points: Sequence[Point] = ()):
        self.type = type
        self.points = list(points)

        