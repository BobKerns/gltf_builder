'''
Tests for type constructors and types in attribute_types.py.
'''

from collections.abc import Callable
from typing import Any

import numpy as np

import pytest
from pytest import approx

from gltf_builder.attribute_types import (
    vector2, vector3, vector4, tangent, scale, point, uv, color,
    weight, weight8, weight16,
    RGB, RGBA,
    _Vector2, _Vector3, _Vector4,
    _Weightf, _Weight8, _Weight16,
    _Tangent, _Scale, _Point, _Uvf,
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
    ):        
        if isinstance(exc, type) and issubclass(exc, Exception):
            with pytest.raises((exc, TypeError)):
                r = cnst(*data)
            return
        if 0 < len(data) < min_data or len(data) > max_data:
            with pytest.raises((ValueError, TypeError)):
                r = cnst(*data)
            return
        data = data[0:ndata]
        r = cnst(*data)
        expected = tcase(t, [float(n) for n in data])
        assert tuple(r) == approx(tuple(expected))
        assert type(r) is t
        assert all(isinstance(v, float) for v in r)
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
    validator(t, cnst, ndata, ndata, ndata, data, exc)

@pytest.mark.parametrize('cnst, min_data, max_data, ndata, t', [
    (color, 3, 4, 3, RGB),
    (color, 3, 4, 4, RGBA),
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
            ):
    validator(t, cnst, min_data, max_data, ndata, data, exc)


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
