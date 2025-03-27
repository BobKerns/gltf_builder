'''
Tests for type constructors and types in attribute_types.py.
'''

from collections.abc import Callable
from typing import Any

import numpy as np

import pytest
from pytest import approx

from gltf_builder.attribute_types import (
    vector2, vector3, vector4, tangent, scale,
    weight, weight8, weight16,
    _Vector2, _Vector3, _Vector4,
    _Weightf, _Weight8, _Weight16,
    _Tangent, _Scale,
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


@pytest.mark.parametrize('tcase', [
    (case_tuple),
    (case_obj),
    (case_numpy)
])
@pytest.mark.parametrize('cnst, ndata, t', [
    (vector2, 2, _Vector2),
    (vector3, 3, _Vector3),
    (vector4, 4, _Vector4),
    (scale, 3, _Scale)
])
@pytest.mark.parametrize('data', [
    ((1.0, 2.0, 3.0, 4.0)),
    ((0.4, 0.3, 0.2, 0.1)),
    ((0.0, 0.0, 0.0, 0.0)),
    ((0, 0, 0, 0)),
])
def test_type_constructors(tcase, t , ndata, cnst, data):
    '''
    Test the type constructors.
    '''
    data = data[:ndata]
    expected = tcase(t, [float(n) for n in data])
    r = cnst(*data)
    assert tuple(r) == approx(tuple(expected))
    assert type(r) is t
    assert all(isinstance(v, float) for v in r)

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
])
@pytest.mark.parametrize('tcase, exact', [
    (case_tuple, False),
    (case_weight, True),
    (case_numpy, False)   
])
def test_weight(tcase,
                exact,
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
    if exact and len(data) != 4:
        return
    expected = tuple(t(*e) for e in expected)
    arg = expected[0] if exact else tcase(t, data)
    r = c(arg)
    for v, e in zip(r, expected):
        assert type(v) is type(e)
        assert tuple(v) == approx(tuple(e))