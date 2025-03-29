'''
Tests for type constructors and types in attribute_types.py.
'''

from collections.abc import Callable
from typing import Literal, Any
from inspect import signature


import numpy as np

import pytest
from pytest import approx

from gltf_builder.core_types import ByteSize
from gltf_builder.attribute_types import (
    vector2, vector3, vector4, tangent, scale, point, uv, joint,
    weight, weight8, weight16,
    color, rgb8,  rgb16, RGB, RGBA, RGB8, RGBA8, RGB16, RGBA16, 
    _Vector2, _Vector3, _Vector4,
    _Weightf, _Weight8, _Weight16,
    _Tangent, _Scale, _Point, _Uvf,
    _Joint, _Joint8, _Joint16,
    EPSILON,
)

def case_tuple(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a tuple of data.
    '''
    return tuple(data)


def case_obj(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a tuple of data.
    '''
    return cnst(*data)


def case_weight(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a tuple of data.
    '''
    total = sum(data)
    if abs(total) < EPSILON:
        return cnst(*(0.0 for _ in data))
    return cnst(*(float(d)/total for d in data))

def case_numpy(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a numpy array.
    '''
    return np.array(data, np.float32)


# Input cases

@pytest.fixture(params=[case_tuple, case_obj, case_numpy])
def validator(request):
    '''
    Validator for types and constructor functions.

    '''
    tcase = request.param
    def validator(
            t: type,
            cnst: Callable[..., Any],
            min_data: int,
            max_data: int,
            ndata: int,
            data: tuple,
            exc: type[Exception] | None,
            size: ByteSize|Literal['inf']=4,
            to_int: bool=False,
    ):
        '''
        Validate the type constructor against the supplied data.
        The data is passed to the constructor and the result is compare
        to the expected type and value.

        The expected data is constructed by one of the following functions:
        - case_tuple: The data is passed as a tuple to the constructor.
        - case_obj: The data is passed as an object of type _t_.
        - case_numpy: The data is passed as a numpy array to the constructor.

        If the constructor in cnst takes a _size_ keyword argument, it is passed.

        An exception is expected if ndata is outside the range of
        min_data and max_data, or if exc is supplied.
        
        Parameters
        ----------
        t : type
            The expected type of the result.
        cnst : Callable[..., Any]
            The constructor function to test.
        min_data : int
            The minimum number of data elements accepted by the constructor.
        max_data : int
            The maximum number of data elements accepted by the constructor.
        ndata : int
            The number of data elements to pass to the constructor.
        data : tuple
            The data to pass to the constructor.
        exc : type[Exception] | None
            The expected exception type, or None if no exception is expected.
        size : ByteSize|Literal['inf'], optional keyword argument
            The size of the data in bytes, by default 4.
            The values are:
            - 1: 1 byte integer
            - 2: 2 bytes integer
            - 4: 4 bytes float32 between 0 and 1
            - 'inf': unlimited range float
        to_int : bool, optional keyword argument
            If True, the data is converted to int, by default False.
            This is used for the color types, where the data is
            converted to int in the range of 0 to 255 or 0 to 65535, based
            on the _size_ parameter.
        '''
        match size:
            case 1:
                def elt_type(x):
                    return round(x*255)
            case 2:
                def elt_type(x):
                    return round(x*65535)
            case 4|'inf':
                elt_type = float
            case _:
                raise ValueError(f"Invalid size: {size}")
        kwargs = {}
        if signature(cnst).parameters.get('size') is not None:
            kwargs['size'] = size
        if isinstance(exc, type) and issubclass(exc, Exception):
            with pytest.raises((exc, TypeError)):
                r = cnst(*data, **kwargs)
            return
        data = data[0:ndata]
        if 0 < len(data) < min_data or len(data) > max_data:
            with pytest.raises((ValueError, TypeError)):
                r = cnst(*data, **kwargs)
            return
        expected = tcase(t, [elt_type(n) for n in data][0:ndata])
        if to_int:
            data = tuple(elt_type(n) for n in data)
        r = cnst(*data, **kwargs)
        assert tuple(r) == approx(tuple(expected))
        assert type(r) is t
        match size:
            case 1:
                assert all(isinstance(v, int) for v in r)
                assert all(0 <= v <= 255 for v in r)
            case 2:
                assert all(isinstance(v, int) for v in r)
                assert all(0 <= v <= 65535 for v in r)
            case 4:
                assert all(isinstance(v, float) for v in r)
                assert all(0 <= v <= 1.0 for v in r)
            case 'inf':
                assert all(isinstance(v, float) for v in r)
            case _:
                raise ValueError(f"Invalid size: {size}")
    return validator
    

# Constructors and constructed types, floating point unlimited range.
@pytest.mark.parametrize('cnst, ndata, t', [
    (vector2, 2, _Vector2),
    (vector3, 3, _Vector3),
    (vector4, 4, _Vector4),
    (scale, 3, _Scale),
    (point, 3, _Point),
    (uv, 2, _Uvf),
])
# Data and expected exceptions.
@pytest.mark.parametrize('data, exc', [
    ((1.0, 2.0, 3.0, 4.0), None),
    ((0.4, 0.3, 0.2, 0.1), None),
    ((0.0, 0.0, 0.0, 0.0), None),
    ((0, "foo", 0, 0,), ValueError),
    ((0, 0, 0, 0, 0), ValueError),
])
def test_type_constructors(
                        validator,
                        t ,
                        ndata,
                        cnst,
                        data,
                        exc,
                    ):
    '''
    Test the type constructors.
    '''
    validator(t, cnst, ndata, ndata, ndata, data, exc, size='inf')

@pytest.mark.parametrize('cnst, min_data, max_data, ndata, t, size, to_int', [
    (color, 3, 4, 3, RGB, 4, False),
    (color, 3, 4, 4, RGBA, 4, False),
    (rgb8, 3, 4, 3, RGB8, 1, True),
    (rgb8, 3, 4, 4, RGBA8, 1, True),
    (rgb16, 3, 4, 3, RGB16, 2, True),
    (rgb16, 3, 4, 4, RGBA16, 2, True),
])
# Data and expected exceptions.
@pytest.mark.parametrize('data, exc', [
    ((1.0, 0.9, 0.3, 0.1), None),
    ((1, "foo", 1, 1,), ValueError),
    ((1, 1, 1, 1, 1), ValueError),
])
def test_color(
                validator,
                t,
                min_data,
                max_data,
                ndata,
                cnst,
                data,
                exc,
                size,
                to_int,
            ):
    validator(t, cnst, min_data, max_data, ndata, data, exc,
               size=size,
               to_int=to_int,
            )


@pytest.mark.parametrize('data', [
    (1, 2, 3, 1),
    (0.4, 0.3, 0.2, 1),
    (1, 2, 3, -1),
    (0.4, 0.3, 0.2, -1),
])
def test_tangent(data):
    '''
    Test the tangent type constructor.
    '''
    expected = _Tangent(*(float(n) for n in data))
    r = tangent(*data)
    assert tuple(r) == approx(tuple(expected))
    assert type(r) is _Tangent
    assert all(isinstance(v, float) for v in r)

@pytest.mark.parametrize('size', [0, 1, 2])
@pytest.mark.parametrize('data', [
    (1,),
    (1, 2,),
    (1, 2, 3,),
    (1, 2, 3, 4,),
    (1, 2, 3, 4, 5,),
    (1, 2, 3, 4, 5, 6,),
    (1, 2, 3, 4, 5, 6, 7,),
])
def test_joint(data, size):
    '''
    Test the joint type constructor.
    '''
    jtype = [_Joint, _Joint8, _Joint16][size]
    def extend(d):
        return d + (0,) * (4 - len(d))
    expected = [jtype(*extend(data[i*4:i*4+4])) for i in range((len(data)+3)//4)]
    r = joint(*data, size=size)
    for v, e in zip(r, expected):
        assert isinstance(v, jtype)
        assert tuple(v) == approx(tuple(e))

@pytest.mark.parametrize('data, size, expected', [
    ((0.4, 0.3, 0.2, 0.1), 0, ((0.4, 0.3, 0.2, 0.1),)),
    ((0.2, 0.2, 0.3, 0.2, 0.1), 0, ((0.2, 0.2, 0.3, 0.2), (0.1, 0.0, 0.0, 0.0))),
    ((0.1, 0.2, 0.3, 0.2, 0.1, 0.1,), 0, ((0.1, 0.2, 0.3, 0.2), (0.1, 0.1, 0.0, 0.0))),
    ((1, 2, 3, 4), 0, ((0.1, 0.2, 0.3, 0.4),)),
    ((0.2, 0, 0, 0), 0, ((1.0, 0, 0, 0),)),
    ((0.5, 0.25, 0.125, 0.125), 8, ((128, 64, 31, 32),)),
    ((2, 2, 4, 8), 8, ((31, 32, 64, 128),)),
    ((0.2, 0, 0, 0), 8, ((255, 0, 0, 0),)),
    ((0.5, 0.25, 0.125, 0.125), 16, ((32768, 16384, 8191, 8192),)),
    ((2, 2, 4, 8), 16, ((8191, 8192, 16384, 32768),)),
    ((0.2, 0, 0, 0), 16, ((65535, 0, 0, 0),)),
    ((0.2,), 16, ((65535, 0, 0, 0),)),
    ((), 0, ValueError),
    ((), 8, ValueError),
    ((), 16, ValueError),
])
@pytest.mark.parametrize('tcase, zeropad', [
    (case_tuple, False),
    (case_weight, True),
    (case_numpy, False)   
])
def test_weight(tcase,
                zeropad,
                data,
                size,
                expected):
    '''
    Test the weight type constructor.
    '''
    c, t = {
        0: (weight, _Weightf),
        8: (weight8, _Weight8),
        16: (weight16, _Weight16),
    }[size]
    argdata = data
    if zeropad:
        # Zeropad means we pad with zeros and construct the expected data
        # from the input data, and ignore the declared expected data.
        if len(data) != 4:
            argdata = (*(float(d) for d in data), 0.0, 0.0, 0.0, 0.0)[:4]
    elif isinstance(expected, type) and issubclass(expected, Exception):
        arg = tcase(t, argdata)
        with pytest.raises(expected):
            c(arg)
        return
    arg =  tcase(t, argdata)
    expected = (arg, ) if zeropad else tuple(t(*e) for e in expected)
    r = c(arg)
    for v, e in zip(r, expected):
        assert type(v) is type(e)
        assert tuple(v) == approx(tuple(e))

@pytest.mark.parametrize('data, expected', [
    ((), (1.0, 1.0, 1.0)),
    ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    ((1.0, 2.0), ValueError),
    ((1.0,), (1.0, 1.0, 1.0)),
    ((1,), (1.0, 1.0, 1.0)),
    ((-1,), (-1.0, -1.0, -1.0)),
])
def test_scale(data, expected):
    '''
    Test the scale type constructor.
    '''
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            scale(*data)
        return
    expected = _Scale(*(float(n) for n in expected))
    r = scale(*data)
    assert tuple(r) == approx(tuple(expected))
    assert type(r) is _Scale
    assert all(isinstance(v, float) for v in r)
