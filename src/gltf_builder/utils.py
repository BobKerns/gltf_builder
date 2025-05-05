'''
Internal utilities for the glTF builder.
'''

from collections.abc import Callable, Iterable
from contextlib import suppress
from enum import Enum
from math import floor
import os
import sys
import pwd
import ctypes
import ctypes.wintypes
import subprocess
import getpass
from itertools import chain, repeat
from typing import  Any, Optional, TypeAlias, TypeVar, overload

import numpy as np

from gltf_builder.core_types import (
    ElementType, ComponentType, BufferType,































































































































































    ComponentSize, ElementSize, IndexSize, NPTypes,
)
from gltf_builder.attribute_types import (
    AttributeData, VectorSpec, Vector4Spec, Vector3Spec, Vector2Spec, Vector4, Vector3, Vector2, VectorLike,
    Tangent,
)


COMPONENT_SIZES: dict[ComponentType|IndexSize, tuple[ComponentSize, type[NPTypes], BufferType]] = {
    ComponentType.BYTE: (1, np.int8, 'b'),
    ComponentType.UNSIGNED_BYTE: (1, np.uint8, 'B'),
    IndexSize.UNSIGNED_BYTE: (1, np.uint8, 'B'),
    ComponentType.SHORT: (2, np.int16, 'h'),
    ComponentType.UNSIGNED_SHORT: (2, np.uint16, 'H'),
    IndexSize.UNSIGNED_SHORT: (2, np.uint16, 'H'),
    ComponentType.UNSIGNED_INT: (4, np.uint32, 'L'),
    IndexSize.UNSIGNED_INT: (4, np.uint32, 'L'),
    ComponentType.FLOAT: (4, np.float32, 'f'),
}

ELEMENT_TYPE_SIZES: dict[ElementType, ElementSize] = {
    ElementType.SCALAR: 1,
    ElementType.VEC2: 2,
    ElementType.VEC3: 3,
    ElementType.VEC4: 4,
    ElementType.MAT2: 4,
    ElementType.MAT3: 9,
    ElementType.MAT4: 16,
}

def decode_component_type(componentType: ComponentType|IndexSize) -> tuple[int, type[NPTypes], BufferType]:
    '''
    Decode the component type into a tuple of the component size, numpy dtype, and buffer type.
    '''
    return COMPONENT_SIZES[componentType]


def decode_type(type: ElementType, componentType: ComponentType|IndexSize) -> tuple[int, int, int, type[NPTypes], BufferType]:
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


def decode_dtype(type: ElementType, componentType: ComponentType|IndexSize) -> type[NPTypes]:
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
def normalize(vec: Tangent ,/) -> Tangent: ... # type: ignore
@overload
def normalize(vec: Vector2Spec, /) -> Vector2: ...
@overload
def normalize(vec: Vector3Spec, /) -> Vector3: ...
@overload
def normalize(vec: Vector4Spec, /) -> Vector4: ...
def normalize(vec: VectorSpec|Tangent, /) -> Vector2|Vector3|Vector4|Tangent:
    '''
    Normalize the vector to unit length.
    '''
    match vec:
        case Tangent():
            tlen = vec.length
            return Tangent(float(vec.x/tlen), float(vec.y/tlen), float(vec.z/tlen), vec.w)
        case VectorLike():
            cls = type(vec)
        case tuple():
            match len(vec):
                case 2:
                    cls = Vector2
                case 3:
                    cls = Vector3
                case 4:
                    cls = Vector4
                case _:
                    raise ValueError(f'Unsupported vector length: {len(vec)}')
        case np.ndarray():
            total = sum(v*v for v in vec) ** 0.5
            match len(vec):
                case 2:
                    return Vector2(vec[0] / total, vec[1] / total)
                case 3:
                    return Vector3(vec[0] / total, vec[1] / total, vec[2] / total)
                case 4:
                    return Vector4(vec[0] / total, vec[1] / total, vec[2] / total, vec[3] / total)
                case _:
                    raise ValueError(f'Unsupported vector length: {len(vec)}')
            raise ValueError(f'{type(vec).__name__} is not a vector-like value.')
        case _:
            raise ValueError(f'{type(vec).__name__} is not a vector-like value.')

    length = sum(float(v)*float(v) for v in vec) ** 0.5
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


def count_iter(iterable: Iterable[Any]) -> int:
    '''
    Count the number of items in an iterable.
    '''
    return sum(1 for _ in iterable)


_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')

@overload
def first(iterable: Iterable[_T1], /) -> _T1: ...
@overload
def first(iterable: Iterable[_T1], default: _T2, /) -> _T1|_T2: ...
def first(iterable: Iterable[_T1], /, *args) -> _T1|Any:
    '''
    Return the first item in an iterable, or a default value if the iterable is empty.
    '''
    if len(args) > 1:
        raise TypeError('Too many arguments for first(iter, [default])')
    try:
        return next(iter(iterable))
    except StopIteration:
        if len(args) == 0:
            raise ValueError('Iterable is empty and no default value was provided.')
        return args[0]
    except TypeError:
        raise TypeError(f'Expected an iterable, got {type(iterable).__name__}')

@overload
def last(iterable: Iterable[_T1], /) -> _T1: ...
@overload
def last(iterable: Iterable[_T1], default: _T2, /) -> _T1|_T2: ...
def last(iterable: Iterable[_T1], *args) -> _T1|Any:
    '''
    Return the last item in an iterable.
    '''
    item = last
    match len(args):
        case 0:
            for item in iterable:
                pass
            if item is last:
                raise ValueError('Iterable is empty and no default value was provided.')
        case 1:
            for item in iterable:
                pass
            if item is last:
                return args[0]
        case _:
            raise TypeError('Too many arguments for last(iter, [default])')
    return item


@overload
def index_of(obj: Iterable[_T1], /, fn: Callable[[_T1], bool], *,
             field: Optional[Any]=None,
             attribute: Optional[str]=None,
             ): ...
@overload
def index_of(obj: Iterable[_T1], /, *,
             value_is: Any,
             field: Optional[Any]=None,
             attribute: Optional[str]=None,
             ): ...
@overload
def index_of(obj: Iterable[_T1], /, *,
             value_is_not: Any,
             field: Optional[Any]=None,
             attribute: Optional[str]=None,
             ): ...
@overload
def index_of(obj: Iterable[_T1], /, *,
             value_eq: Any,
             field: Optional[Any]=None,
             attribute: Optional[str]=None,
             ): ...
@overload
def index_of(obj: Iterable[_T1], /, *,
             value_ne: Any,
             field: Optional[Any]=None,
             attribute: Optional[str]=None,
             ): ...
def index_of(obj: Iterable[_T1], /, fn: Optional[Callable[[_T1], bool]]=None,
             **kwargs) -> int:
    '''
    Return the index of the first item in an iterable that matches the given predicate.
    If no item matches, return -1.

    See also `value_is`, `value_is_not`, `value_eq`, `value_ne`, which are
    convenience functions for common predicates. The corresponding keyword arguments
    use the functions to create the predicate function if no predicate function is given.

    The keyword arguments `field` and `attribute` can be used to extract a value from
    the item before applying the predicate function. If the item is a dictionary,
    `field` is the key to extract, and if the item is an object, `attribute` is the
    attribute to extract. If both are given, the value is extracted from the item
    in the order they are given.

    These correspond to the `value_field` and `value_attribute` functions, respectively.

    PARAMETERS
    ----------
    obj: Iterable[_T1]
        The iterable to search.
    fn: Callable[[_T1], bool]
        The predicate function to match the items against.
    value_is: Any
        A value to match the items against. If the item is equal to this value, it is considered a match.
    value_is_not: Any
        A value to match the items against. If the item is not equal to this value, it is considered a match.
    value_eq: Any
        A value to match the items against. If the item is == to this value, it is considered a match.
    value_ne: Any
        A value to match the items against. If the item is != to this value, it is considered a match.
    field: Optional[Any]
        A field to extract from the item. If the item is a dictionary, this is the key to extract.
    attribute: Optional[str]
        An attribute to extract from the item. If the item is an object, this is the attribute to extract.


    RETURNS
    -------
    int
        The index of the first item that matches the predicate, or -1 if no item matches.
    '''
    extract = None
    if 'value_is' in kwargs:
        if fn:
            raise TypeError('Can specify one of fn=, value_is=, or value_is_not=.')
        fn = value_eq(kwargs['value_is'])
    if 'value_is_not' in kwargs:
        if fn:
            raise TypeError('Can specify one of fn=, value_is=, or value_is_not=.')
        fn = value_ne(kwargs['value_is_not'])
    if 'value_eq' in kwargs:
        if fn:
            raise TypeError('Can specify one of fn=, value_eq=, or value_ne=.')
        fn = value_eq(kwargs['value_eq'])
    if 'value_ne' in kwargs:
        if fn:
            raise TypeError('Can specify one of fn=, value_eq=, or value_ne=.')
        fn = value_ne(kwargs['value_ne'])
    if fn is None:
        fn = bool
    for key, val in kwargs.items():
        match key:
            case 'attribute':
                if extract is None:
                    extract = value_attribute(val)
                else:
                    extract = chain_fns(extract, value_attribute(val))
            case 'field':
                if extract is None:
                    extract = value_field(val)
                else:
                    extract = chain_fns(extract, value_field(val))
            case 'value_is'|'value_is_not'|'value_eq'|'value_ne'|'fn':
                pass
            case _:
                raise TypeError(f'Unknown keyword argument {key!r}.')
    for i, item in enumerate(obj):
        with suppress(Exception):
            if extract is not None:
                item = extract(item)
            if fn(item):
                return i
    return -1

def value_is(item: Any) -> Callable[[Any], bool]:
    '''
    Return a predicate function that checks if an item is equal to the given item.
    '''
    def value_is(x: Any) -> bool:
        return x is item
    return value_is

def value_is_not(item: Any) -> Callable[[Any], bool]:
    '''
    Return a predicate function that checks if an item is not equal to the given item.
    '''
    def value_is_not(x: Any) -> bool:
        return x is not item
    return value_is_not

def value_eq(item: Any) -> Callable[[Any], bool]:
    '''
    Return a predicate function that checks if an item is equal to the given item.
    '''
    def value_eq(x: Any) -> bool:
        return x == item
    return value_eq

def value_ne(item: Any) -> Callable[[Any], bool]:
    '''
    Return a predicate function that checks if an item is not equal to the given item.
    '''
    def item_ne(x: Any) -> bool:
        return x != item
    return item_ne


def value_field(field: Any, /, *args) -> Callable[[Any], Any]:
    '''
    Return a function that obtains the the value of the given field from the object.
    If the field is not found, return None.

    A field is either accessed with subscript notation.
    '''
    match len(args):
        case 0:
            def value_field(obj: Any, /) -> Any:
                try:
                    return obj[field] # type: ignore
                except KeyError:
                    raise ValueError(f'Field {field} not found in object {obj}.')
        case 1:
            def value_field(obj: Any, /) -> Any:
                try:
                    return obj[field] # type: ignore
                except KeyError:
                    return args[0]
        case _:
            raise TypeError('Too many arguments for field_of(obj, field, [default])')
    return value_field

def value_attribute(attr: str, /, *args) -> Callable[[Any], Any]:
    '''
    Return the value of the given attribute from the object.

    An attribute is either accessed with dot notation or subscript notation.
    '''
    match len(args):
        case 0:
            def value_attribute(obj: Any, /) -> Any:
                return getattr(obj, attr)
        case 1:
            def value_attribute(obj: Any, /) -> Any:
                return getattr(obj, attr, args[0])
        case _:
            raise TypeError('Too many arguments for attribute_of(obj, attr, [default])')
    return value_attribute

def chain_fns(fn1: Callable[[_T1], _T2],
                fn2: Callable[[_T2], _T3],
                /,
                ) -> Callable[[_T1], _T3]:
    '''
    Chain two functions together. The first function is called with the input value,
    and the result is passed to the second function.
    '''
    def chain_fns(x: _T1) -> _T3:
        return fn2(fn1(x))
    return chain_fns


def simple_num(x: Any, /) -> str:
    '''
    Convert a number to a string with a maximum of 3 decimal places.
    If the value is an Enum, return the name of the Enum member.
    '''
    if isinstance(x, Enum):
        return x.name
    if isinstance(x, (float, np.floating)):
        # Eliminate trailing zeros
        if x == int(x):
            return f'{int(x)}'
        if x * 10 == int(x * 10):
            return f'{x:.1f}'
        if x * 100 == int(x * 100):
            return f'{x:.2f}'
        return f'{x:.3f}'
    return str(x)

AttrSpec: TypeAlias = str|tuple[str, Any]|tuple[str, Any, str]

def std_repr(self: object,
             keys: Iterable[AttrSpec],
             /,
             id: Optional[str|int]=None,
             cls: Optional[str]=None,
             ) -> str:
    '''
    Generic tool for `__repr__` methods.
    Generates a string representation of the object with the given keys.

    The format is:
    <ClassName@id key1=value1, key2=value2, ...>

    where `id` is optional and can be None.

    PARAMETERS
    ----------
    self: object
        The object to generate a representation for.
    keys: Iterable[str|tuple[str, Any]|tuple[str, Any, str]]
        The keys to include in the representation.
        If a tuple, the first element is the key and the second element is the value.
        An optional third element can be used to specify the display name.
    id: Optional[str]
        The id to include in the representation. If None, no id is included.
    cls: Optional[str]
        The class name to use in the representation. If None, the class name is used.
    '''
    def val(v: AttributeData) -> str:
        if isinstance(v, (tuple, np.ndarray)):
            return f"({','.join(simple_num(x) for x in v)})"
        return simple_num(v)
    cls = cls or type(self).__name__.lstrip('_')
    def get(key: AttrSpec) -> Any:
        if isinstance(key, tuple):
            return key[1]
        if hasattr(self, key):
            val = getattr(self, key)
            return val
        if hasattr(self, 'attributes'):
            attrs = getattr(self, 'attributes', {})
            if key in attrs:
                return attrs[key]
        return self[key] # type: ignore
    def key(key: AttrSpec) -> str:
        if isinstance(key, tuple):
            if len(key) == 3:
                return key[2]
            return key[0]
        return key
    if id is not None:
        if isinstance(id, int):
            id = f'00000000{hex(id)}'
            id = f'#{id[-8:-4]}-{id[-4:]}'
        else:
            id = f'@{id}'
    else:
        id = ''
    return f'''<{cls}{id} {
            ", ".join(
                    f"{k}={val(v)}"
                    for k, v in (
                        (key(kv),get(kv))
                        for kv in keys
                    )
                    if v not in (None, "")
                )
            }>'''

def _get_human_name():
    """
    Returns the full name of the current user, falling back to the username if necessary.
    """

    try:
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
    except Exception:
        return ''


def _get_username():
    """
    Returns the username of the current user.
    """
    try:
        return getpass.getuser()
    except Exception:
        return ''

USERNAME: str = _get_username()


USER: str = _get_human_name()
