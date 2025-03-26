'''
Test cases for functions in the utils module.
'''

from math import sqrt

import numpy as np

import pytest
from pytest import approx

from gltf_builder.utils import distribute_ints, distribute_floats, normalize, map_range

from gltf_builder.core_types import (
    _Vector2, _Vector3, _Vector4, _Tangent, EPSILON,
)


@pytest.mark.parametrize('lower, upper, values, expected', [
    (0, 1, (), ()),
    (0, 1, (0,), (1.0,)),
    (0, 1, (0.5,), (1.0,)),
    (0, 1, (0.1, 0.2, 0.3), (0.16666666666666666, 0.3333333333333333, 0.5)),
    (0, 1, (0, 0, 0, 0), (0.25, 0.25, 0.25, 0.25)),
    (0, 1, (0, 0, 0, 0, 0), (0.2, 0.2, 0.2, 0.2, 0.2)),
    (-1, 1, (0,), (1.0,)),
    (-1, 1, (0, 0), (0.0, 0.0)),
    (0, 1, (1, 3, 4), (0.125, 0.375, 0.5)),
    (1, 2, (2, 4, 5), (1.125, 1.375, 1.5)),
])
def test_distribute_floats(lower, upper, values, expected):
    assert lower < upper, f'BAD TEST: {lower} < {upper}'
    assert len(values) == len(expected), f'BAD TEST: {len(values)} ≠ {len(expected)}'
    expected_sum = sum((v - lower) for v in (expected or (upper,)))
    assert expected_sum == approx(upper - lower), (
        f'BAD TEST: expected values do not sum to {upper - lower}'
    )
    r = distribute_floats(*values,
                            lower=lower,
                            upper=upper,
                        )
    assert len(r) == len(values), "Wrong number of values"
    assert sum((v - lower) for v in (r or (upper,))) == approx(upper - lower), "total is wrong"
    assert r == approx(expected)  

@pytest.mark.parametrize('lower, upper, values, expected', [
    (0, 255, (), ()),
    (0, 255, (0, 0, 0), (85, 85, 85)),
    (0, 255, (0.5,), (255,)),
    (0, 255, (10, 20, 30), (43, 85, 127)),
    (3, 258, (13, 23, 33), (46, 88, 130)),
    (0, 255, (0, 0, 0, 0), (63, 64, 64, 64)),
    (0, 255, (0, 0, 0, 0, 0), (51, 51, 51, 51, 51)),
    (0, 255, (0, 0, 0, 0, 0, 0), (43, 43, 43, 42, 42, 42)),
    (0, 255, (0, 0, 0, 0, 0, 0, 0), (37, 37, 37, 36, 36, 36, 36)),
    (0, 255, (0, 0, 0, 0, 0, 0, 0, 0), (31, 32, 32, 32, 32, 32, 32, 32)),
    (0, 255, (0, 0, 0, 0, 0, 0, 0, 0, 0), (29, 29, 29, 28, 28, 28, 28, 28, 28)),
    (0, 255, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (25, 25, 25, 25, 25, 26, 26, 26, 26, 26)),
    (0, 255, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (24, 24, 23, 23, 23, 23, 23, 23, 23, 23, 23)),
    (0, 65535, (0, 0, 0), (21845, 21845, 21845)),
    (0, 65535, (0, 0, 0, 0), (16383, 16384, 16384, 16384)),
    (0, 65535, (10, 20, 30), (10923, 21845, 32767)),
    (10, 65545, (20, 30, 40), (10933, 21855, 32777)),

])
def test_distribute_ints(lower, upper,
                        values,
                        expected):
    assert lower < upper, f'BAD TEST: {lower} < {upper}'
    assert len(values) == len(expected), f'BAD TEST: {len(values)} ≠ {len(expected)}'
    expected_sum = sum(v - lower for v in (expected or (upper,)))
    assert expected_sum == upper - lower, (
        f'BAD TEST: expected values do not sum to {upper - lower}'
    )
    r = distribute_ints(*values,
        lower=lower,
        upper=upper,
        )
    assert len(r) == len(values), "Wrong number of values"
    if len(r) == 0:
        return
    total = sum(v - lower for v in r)
    assert total == upper - lower, "total is wrong"
    assert r == expected


@pytest.mark.parametrize('input,vtype,expected', [
    ((1, 1), _Vector2, (sqrt(2)/2, sqrt(2)/2)),
    ((1, 0), _Vector2, (1.0, 0.0)),
    ((0, 1), _Vector2, (0.0, 1.0)),
    ((0, 0), _Vector2, (0, 0)),
    (_Vector2(1, 1), _Vector2, (sqrt(2)/2, sqrt(2)/2)),
    (np.array((1, 1), np.float32), _Vector2, (sqrt(2)/2, sqrt(2)/2)),
    ((1, 1, 1), _Vector3, (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)),
    (_Vector3(1, 1, 1), _Vector3, (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)),
    (np.array((1, 1, 1), np.float32), _Vector3, (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)),
    ((1, 1, 1, 1), _Vector4, (0.5, 0.5, 0.5, 0.5)),
    (_Vector4(1, 1, 1, 1), _Vector4, (0.5, 0.5, 0.5, 0.5)),
    (_Tangent(1, 1, 1, 1), _Tangent, (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3, 1.0)),
    (np.array((1, 1, 1, 1), np.float32), _Vector4, (0.5, 0.5, 0.5, 0.5)),
])
def test_normalize(input, vtype, expected):
    r = normalize(input)
    assert type(r) is vtype
    assert r == approx(expected)
    if r.length > EPSILON:
        assert r.length == approx(1.0)


@pytest.mark.parametrize('input,from_range,to_range,expected', [
    (0, (0, 1), (0, 1), 0),
    (0, (0, 1), (0, 2), 0),
    (0.5, (0, 1), (0, 2), 1),
    (0.25, (0, 1), (0, 2), 0),
    (0, (0, 1), (1, 2), 1),
    (0.25, (0, 1), (1.0, 2.0), 1.25),
    (0.25, (0, 1), (2.0, 1.0), 1.75),
    (0, (0, 1), (0.0, 1.0), 0.0),
    (0, (0, 1), (0.0, 2.0), 0.0),
    (0, (0, 1), (1.0, 2.0), 1.0),
    (0.0, (0.0, 1.0), (0, 255), 0),
    (0.5, (0.0, 1.0), (0, 255), 128),
    (1.0, (0.0, 1.0), (0, 255), 255),
    (0.0, (0.0, 1.0), (0, 65535), 0),
    (0.5, (0.0, 1.0), (0, 65535), 32768),
    (1.0, (0.0, 1.0), (0, 65535), 65535),
])
def test_map_range(input, from_range, to_range, expected):
    r = map_range(input,
                  from_range=from_range,
                  to_range=to_range,
                )
    assert r == approx(expected)
    assert type(r) is type(expected)
