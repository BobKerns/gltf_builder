'''
Internal utilities for the glTF builder.
'''

from typing import Literal, TypeAlias

import numpy as np

from gltf_builder.element import (
    ElementType, ComponentType
)

BufferType: TypeAlias = Literal['b', 'B', 'h', 'H', 'l', 'L', 'f']

COMPONENT_SIZES: dict[ComponentType, tuple[int, np.dtype, BufferType]] = {
    ComponentType.BYTE: (1, np.int8, 'b'),
    ComponentType.UNSIGNED_BYTE: (1, np.uint8, 'B'),
    ComponentType.SHORT: (2, np.int16, 'h'),
    ComponentType.UNSIGNED_SHORT: (2, np.uint16, 'H'),
    ComponentType.UNSIGNED_INT: (4, np.uint32, 'L'),
    ComponentType.FLOAT: (4, np.float32, 'f'),
}


ELEMENT_TYPE_SIZES = {
    ElementType.SCALAR: 1,
    ElementType.VEC2: 2,
    ElementType.VEC3: 3,
    ElementType.VEC4: 4,
    ElementType.MAT2: 4,
    ElementType.MAT3: 9,
    ElementType.MAT4: 16,
}

def decode_component_type(componentType: ComponentType) -> tuple[int, np.dtype, BufferType]:
    '''
    Decode the component type into a tuple of the component size, numpy dtype, and buffer type.
    '''
    return COMPONENT_SIZES[componentType]


def decode_type(type: ElementType, componentType: ComponentType) -> tuple[int, int, int, np.dtype, BufferType]:
    '''
    Decode the `ElementType` and `ComponenType` into a tuple of:
    - the component count per element
    - bytes per component
    - stride (total bytes per element)
    - numpy dtype
    - buffer type char (as for `memoryview.cast()`)
    '''
    componentSize, dt, bt = decode_component_type(componentType)
    componentCount = decode_element_type(type)
    stride = componentSize * componentCount
    return componentCount, componentSize, stride, dt, bt

def decode_stride(type: ElementType, componentType: ComponentType) -> int:
    '''
    Decode the `ElementType` and `ComponenType` into the stride (total bytes per element).
    '''
    return decode_type(type, componentType)[2]

def decode_dtype(type: ElementType, componentType: ComponentType) -> np.dtype:
    '''
    Decode the `ElementType` and `ComponenType` into the numpy dtype.
    '''
    return decode_type(type, componentType)[3]


def decode_element_type(type: ElementType) -> int:
    '''
    Decode the `ElementType` into the number of components per element.

    For example, ElementType.VEC3 -> 3
    '''
    return ELEMENT_TYPE_SIZES[type]
