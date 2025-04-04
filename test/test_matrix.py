'''
Tests of the matrix.py module
This file contains unit tests for the functions in the matrix.py module.
These tests cover matrix multiplication, matrix inversion, and matrix
decomposition to translation, rotation, and scale.
'''

import numpy as np
from pytest import approx

from gltf_builder.matrix import (
    matrix,
    IDENTITY,
)
from gltf_builder.attribute_types import (
    vector3, point,
)

def test_identity_matrix_preserves_vector_and_point():
    m = IDENTITY
    v = vector3(1, 2, 3)
    p = point(4, 5, 6)

    assert m @ v == approx(v)
    assert m @ p == approx(p)

def test_translation_affects_point_not_vector():
    m = matrix((
        (1, 0, 0, 10),
        (0, 1, 0, 20),
        (0, 0, 1, 30),
        (0, 0, 0,  1)
    ))

    v = vector3(1, 2, 3)
    p = point(1, 2, 3)

    # Point is translated
    assert m @ p == approx(point(11, 22, 33))

    # Vector is unaffected by translation
    assert m @ v == approx(vector3(1, 2, 3))

def test_rotation_z_affects_vector_and_point():
    # 90° rotation around Z axis
    theta = np.pi / 2
    c, s = np.cos(theta), np.sin(theta)

    m = matrix((
        (c, -s, 0, 0),
        (s,  c, 0, 0),
        (0,  0, 1, 0),
        (0,  0, 0, 1)
    ))

    v = vector3(1, 0, 0)
    p = point(1, 0, 0)

    rotated = (0, 1, 0)
    assert m @ v == approx(vector3(*rotated))
    assert m @ p == approx(point(*rotated))

def test_combined_rotation_translation():
    # Rotate 180° around Z and translate (5, 0, 0)
    m = matrix((
        (-1,  0, 0, 5),
        (0, -1, 0, 0),
        ( 0,  0, 1, 0),
        ( 0,  0, 0, 1)
    ))

    v = vector3(2, 0, 0)
    p = point(2, 0, 0)

    # Vector rotates but isn't translated
    assert m @ v == approx(vector3(-2, 0, 0))

    # Point rotates and then translates
    assert m @ p == approx(point(3, 0, 0))


def test_copy_equal():
    m1 = matrix((
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16)
    ))
    m2 = m1.copy()
    assert m1 == m2

def test_scalar_mul():
    m = matrix((
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16)
    ))
    scalar = 2
    result = m * scalar
    expected = matrix((
        (2, 4, 6, 8),
        (10, 12, 14, 16),
        (18, 20, 22, 24),
        (26, 28, 30, 32)
    ))
    assert result == expected


def test_matrix_rmul():
    m = matrix((
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16)
    ))
    scalar = 2
    result = scalar * m
    expected = matrix((
        (2, 4, 6, 8),
        (10, 12, 14, 16),
        (18, 20, 22, 24),
        (26, 28, 30, 32)
    ))
    assert result == expected