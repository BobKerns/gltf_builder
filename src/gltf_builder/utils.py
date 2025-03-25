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

import numpy as np

from gltf_builder.core_types import (
    ElementType, ComponentType, BufferType,
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
    if abs(total) < 0.0000001:
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
    if abs(total) < 0.0000001:
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

try:
    USERNAME = getpass.getuser()
except Exception:
    USERNAME = ''
try:
    USER = _get_human_name()
except Exception:
    USER = ''
