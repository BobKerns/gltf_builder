'''
Internal utilities for the glTF builder.
'''

from math import floor
import os
import sys
import pwd
import ctypes
import ctypes.wintypes
import subprocess
import getpass
from itertools import chain, repeat
from typing import  overload

import numpy as np

from gltf_builder.core_types import (
    ElementType, ComponentType, BufferType,
)
from gltf_builder.attribute_types import (
    Vector, Vector4, Vector3, Vector2, _Vector, _Vector4, _Vector3, _Vector2, VectorLike,
    _Tangent,
)


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
    Decode the `ElementType` and `ComponentType` into a tuple of:
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
    Decode the `ElementType` and `ComponentType` into the stride (total bytes per element).
    '''
    return decode_type(type, componentType)[2]

def decode_dtype(type: ElementType, componentType: ComponentType) -> np.dtype:
    '''
    Decode the `ElementType` and `ComponentType` into the numpy dtype.
    '''
    return decode_type(type, componentType)[3]


def decode_element_type(type: ElementType) -> int:
    '''
    Decode the `ElementType` into the number of components per element.

    For example, ElementType.VEC3 -> 3
    '''
    return ELEMENT_TYPE_SIZES[type]


def distribute_floats(*values: float,
                     lower: float=0.0,
                     upper: float=1.0) -> tuple[float, ...]:
    '''
    Distribute the floats to a tuple of floats. By default, the target range is 0.0 to 1.0.
    The input values are interpreted as weights, and the result values lie within the target
    range, with the weights determining the distribution. The sum of positions within the
    target range is equal to the total size of the range.

    For example, if the target range is 0.0 to 1.0, and the input values are `(1, 3, 4)`,
    the result will be `(0.125, 0.375, 0.5)`, which sum to 1.0, as for probabilities.
    '''
    c = len(values)
    d = float(upper - lower)
    if c == 0:
        return ()
    total = sum(float(v) - lower for v in values)
    if abs(total) < 0.00001:
        return tuple(d / c + lower for _ in values)
    return tuple(d * (v - lower) / total + lower for v in values)

    
def distribute_ints(*values: int|float, lower: int=0, upper: int=255) -> tuple[int, ...]:
    '''
    Distribute the ints to a tuple of ints. By default, the target range is 0 to 255.
    The input values are interpreted as weights, and the result values lie within the target
    range, with the weights determining the distribution. The sum of positions within the
    target range is equal to the total size of the range.

    For example, if the target range is 0 to 255, and the input values are `(1, 3, 4)`,
    the result will approximate `(0.125*255, 0.375*255, 0.5*255)`, which the additional
    constraint that the values are adjusted to sum to 255, as for probabilities scaled to
    8-bit integer values.
    '''
    c = len(values)
    if c == 0:
        return ()
    delta = upper - lower
    total = sum(float(v) - lower for v in values)
    d = float(delta)
    if abs(total) < 0.000001:
        r = (round(d / c),) * c
    else:
        r = tuple(floor(d * ((v - lower) / total)) for v in values)
    tt = sum(r)
    fixes = delta - tt
    if fixes > 0:
        f = chain(repeat(1, fixes), repeat(0, c - fixes))
    elif fixes < 0:
        f = chain(repeat(-1, -fixes), repeat(0, c + fixes))
    else:
        return tuple(lower + v for v in r)
    return tuple(a + b + lower for a, b in zip(r, f))



@overload
def normalize(vec: Vector2) -> _Vector2: ...
@overload
def normalize(vec: Vector3) -> _Vector3: ...
@overload
def normalize(vec: Vector4) -> _Vector4: ...
def normalize(vec: Vector) -> _Vector:
    '''
    Normalize the vector to unit length.
    '''
    match vec:
        case _Tangent():
            tlen = vec.length
            return _Tangent(vec.x/tlen, vec.y/tlen, vec.z/tlen, vec.w)
        case VectorLike():
            cls = type(vec)
        case tuple() if type(vec) is not tuple:
            # A NamedTuple
            raise ValueError(f'{type(vec).name} is not a vector-like value.')
    match len(vec):
        case 2:
            cls = _Vector2
        case 3:
            cls = _Vector3
        case 4:
            cls = _Vector4
        case _:
            raise ValueError(f'Unsupported vector length: {len(vec)}')

    length = sum(v*v for v in vec) ** 0.5
    if length < 0.0000001:
        return cls(*repeat(0.0, len(vec)))
    return cls(*(v / length for v in vec))


@overload
def map_range(value: int,
              from_range: tuple[int, int],
              to_range: tuple[int, int],
              ) -> int: ...
@overload
def map_range(value: int,
              from_range: tuple[int, int],
              to_range: tuple[float, float],
              ) -> float: ...
@overload
def map_range(value: float,
              from_range: tuple[float, float],
              to_range: tuple[int, int],
              ) -> int: ...
@overload
def map_range(value: float,
              from_range: tuple[float, float],
              to_range: tuple[float, float],
              ) -> float: ...
def map_range(value: float|int,
              from_range: tuple[float, float]|tuple[int, int],
              to_range: tuple[float, float]|tuple[int, int],
              ) -> float|int:
    '''
    Map a value from one range to another. The value is clamped to the input range.

    If the to_range is ints, the result will be an int, otherwise a float.
    '''
    from_min, from_max = from_range
    to_min, to_max = to_range
    from_delta = from_max - from_min
    to_delta = to_max - to_min
    if from_delta == 0:
        return to_min
    value = max(from_min, min(from_max, value))
    new_value = (float(value - from_min) / from_delta) * to_delta + to_min
    if isinstance(to_min, int) and isinstance(to_max, int):
        return round(new_value)
    return new_value


def _get_human_name():
    """Returns the full name of the current user, falling back to the username if necessary."""
    
    full_name = None

    if sys.platform.startswith("linux") or sys.platform == "darwin":  # macOS and Linux
        try:
            full_name = pwd.getpwuid(os.getuid()).pw_gecos.split(',')[0].strip()
        except KeyError:
            pass
        
        # Try getent as a fallback
        if not full_name:
            try:
                result = subprocess.check_output(["getent", "passwd", os.getlogin()], text=True)
                full_name = result.split(":")[4].split(",")[0].strip()
            except (subprocess.CalledProcessError, IndexError, FileNotFoundError, OSError):
                pass

    elif sys.platform.startswith("win"):  # Windows
        try:
            size = ctypes.wintypes.DWORD(0)
            ctypes.windll.advapi32.GetUserNameExW(3, None, ctypes.byref(size))  # Get required buffer size
            buffer = ctypes.create_unicode_buffer(size.value)
            if ctypes.windll.advapi32.GetUserNameExW(3, buffer, ctypes.byref(size)):
                full_name = buffer.value.strip()
        except Exception:
            pass

    # If full name is not found, fall back to the username
    if not full_name:
        full_name = getpass.getuser()

    return full_name

USERNAME: str
'''
The current user's login name, included by default in the glTF `Asset` metadata.
'''
try:
    USERNAME = getpass.getuser()
except Exception:
    USERNAME = ''

USER: str
'''
The current user's full name, included by default in the glTF `Asset` metadata.
'''
try:
    USER = _get_human_name()
except Exception:
    USER = ''
