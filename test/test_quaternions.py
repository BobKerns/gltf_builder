'''
Tests for quaternions.py
This file contains unit tests for the functions in the quaternions.py module.
These tests cover quaternion multiplication, quaternion normalization,
quaternion conversion to matrix, and quaternion interpolation.
'''

import math

import numpy as np
import pytest
from pytest import approx, mark

from gltf_builder.quaternions import (
    I, IDENTITY, J, K, MINUS_ONE, quaternion, Quaternion as Q, Quaternion,
)
from gltf_builder.attribute_types import (
    vector3, scale,
    _Vector3, _Scale,
)
from gltf_builder.matrix import matrix


def test_quaternian():
    'Test quaternion initialization'
    q = quaternion(1, 2, 3, 4)
    assert q.x == 1
    assert q.y == 2
    assert q.z == 3
    assert q.w == 4


def rotate_vector(v, q):
    """Rotate 3D vector v = (x, y, z) by unit quaternion q"""
    v_q = quaternion(*v, 0.0)
    q_inv = Q.conjugate(q)  # if q is unit-length, conjugate = inverse
    return Q.multiply(Q.multiply(q, v_q), q_inv)[:3]  # drop scalar part

def test_identity_multiplication():
    q = quaternion(0.5, 1, 2, 3)
    assert q * IDENTITY == q
    assert IDENTITY * q == q

@pytest.mark.parametrize("a, b, result", [
    (I, J, K),
    (J, K, I),
    (K, I, J),
    (J, I, -K),
    (K, J, -I),
    (I, K, -J),
    (I, I, MINUS_ONE),
    (J, J, MINUS_ONE),
    (K, K, MINUS_ONE),
])
def test_imaginary_unit_products(a, b, result):
    assert a * b == result


def test_full_formula_check():
    q1 = quaternion(1, 2, 3, 0)
    q2 = quaternion(4, 5, 6, 0)
    expected = quaternion(-3, 6, -3, -32)
    assert q1 * q2 == expected


def test_full_formula_check_operator():
    q1 = quaternion(1, 2, 3, 0)
    q2 = quaternion(4, 5, 6, 0)
    expected = quaternion(-3, 6, -3, -32)
    assert q1 * q2 == expected


def test_mul_operator_float():
    q = quaternion(1, 2, 3, 4)
    expected = quaternion(2, 4, 6, 8)
    assert q * 2 == expected


def test_inverse_multiplication():
    theta = math.pi / 2
    cos_t = math.cos(theta / 2)
    sin_t = math.sin(theta / 2)
    q = quaternion(sin_t, 0, 0, cos_t)
    q_inv = quaternion(-sin_t, 0, 0, cos_t)

    assert q * q_inv == approx(IDENTITY)


def test_non_commutativity():
    q1 = quaternion(1, 2, 3, 0)
    q2 = quaternion(4, 5, 6, 0)
    prod1 = q1 * q2
    prod2 = q2 * q1
    assert prod1 != approx(prod2)

def test_rotate_vector_90deg_z():
    # Rotate (1, 0, 0) by 90° around Z axis → should become (0, 1, 0)
    theta = math.pi / 2
    cos_t = math.cos(theta / 2)
    sin_t = math.sin(theta / 2)
    q = quaternion(0, 0, sin_t, cos_t)  # unit quaternion for 90° around Z

    v = vector3(1, 0, 0)
    v_rotated = q.rotate_vector(v)

    assert v_rotated == approx(vector3(0, 1, 0), abs=1e-6)

def test_rotate_vector_180deg_y():
    # Rotate (1, 0, 0) by 180° around Y axis → should become (-1, 0, 0)
    theta = math.pi
    q = quaternion(math.cos(theta/2), 0, math.sin(theta/2), 0)

    v = vector3(1, 0, 0)
    v_rotated = q.rotate_vector(v)

    assert v_rotated == approx(vector3(-1, 0, 0), abs=1e-6)


def test_q_norm():
    'Test quaternion normalization'
    q = quaternion(1, 2, 3, 4)
    normalized_q = q.normalize()
    norm = (q.w**2 + q.x**2 + q.y**2 + q.z**2) ** 0.5
    assert normalized_q.w == approx(q.w / norm)
    assert normalized_q.x == approx(q.x / norm)
    assert normalized_q.y == approx(q.y / norm)
    assert normalized_q.z == approx(q.z / norm)


@mark.parametrize("matrix_input, expected_translation, expected_rotation, expected_scale", [
    # Translation only
    (
        np.array([
            [1, 0, 0, 5],
            [0, 1, 0, 6],
            [0, 0, 1, 7],
            [0, 0, 0, 1]
        ]),
        vector3(5, 6, 7),
        quaternion(0, 0, 0, 1),
        scale(1, 1, 1)
    ),

    # Scaling only
    (
        np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 1]
        ]),
        vector3(0, 0, 0),
        quaternion(0, 0, 0, 1),
        scale(2, 3, 4)
    ),

    # Rotation only: 90° around Z
    (
        np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ]),
        vector3(0, 0, 0),
        quaternion(0, 0, np.sqrt(0.5), np.sqrt(0.5)),
        scale(1, 1, 1)
    ),

    # Rotation only: 180° around Y
    (
        np.array([
            [-1,  0, 0, 0],
            [ 0,  1, 0, 0],
            [ 0,  0, -1, 0],
            [ 0,  0,  0, 1]
        ]),
        vector3(0, 0, 0),
        quaternion(0, 1, 0, 0),
        scale(1, 1, 1)
    ),

    # Composed case: translation + uniform scale + identity rotation
    (
        np.array([
            [2, 0, 0, 9],
            [0, 2, 0, 8],
            [0, 0, 2, 7],
            [0, 0, 0, 1]
        ]),
        vector3(9, 8, 7),
        quaternion(0, 0, 0, 1),
        scale(2, 2, 2)
    ),

    # Composed case: rotation (90° Z) + translation
    (
        np.array([
            [0, -1, 0, 3],
            [1,  0, 0, 4],
            [0,  0, 1, 5],
            [0,  0, 0, 1]
        ]),
        vector3(3, 4, 5),
        quaternion(0, 0, np.sqrt(0.5), np.sqrt(0.5)),
        scale(1, 1, 1)
    ),

    # Composed case: rotation (90° X), uniform scale (3), and translation
    (
        np.array([
            [3, 0,  0, 10],
            [0, 0, -3, 20],
            [0, 3,  0, 30],
            [0, 0,  0, 1]
        ]),
        vector3(10, 20, 30),
        quaternion(np.sqrt(0.5), 0, 0, np.sqrt(0.5)),
        scale(3, 3, 3)
    ),
])
def test_decompose_trs(matrix_input, expected_translation, expected_rotation, expected_scale):
    t, r, s = Q.decompose_trs(matrix(matrix_input))

    assert isinstance(t, _Vector3)
    assert isinstance(r, Quaternion)
    assert isinstance(s, _Scale)

    assert t == approx(expected_translation)
    assert r == approx(expected_rotation)
    assert s == approx(expected_scale)


def angle_axis_quaternion(axis, angle_rad):
    """Constructs quaternion (x, y, z, w) for rotation around axis."""
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    half_angle = angle_rad / 2
    s = np.sin(half_angle)
    x, y, z = axis * s
    w = np.cos(half_angle)
    return quaternion(x, y, z, w)


def test_log_of_identity_is_zero():
    q = quaternion(0, 0, 0, 1)
    log = q.log()
    assert (log.x, log.y, log.z) == approx((0, 0, 0))


def test_exp_of_zero_is_identity():
    zero = quaternion(0, 0, 0, 0)
    result = Q.exp(zero)
    assert (result.x, result.y, result.z, result.w) == approx((0, 0, 0, 1))


def test_log_exp_round_trip():
    q = Q.from_axis_angle((0, 0, 1), np.pi / 2)
    q2 = Q.exp(q.log())
    assert q2 == approx(q)

def test_exp_log_round_trip():
    q = angle_axis_quaternion((1, 2, 3), np.pi * 0.75)
    q2 = Q.exp(q.log())
    assert q2 == approx((q))


def test_log_magnitude_is_half_angle():
    q = angle_axis_quaternion((0, 1, 0), np.pi)
    log_q = q.log()
    # Expected: log should have magnitude π/2 in direction (0,1,0)
    assert (log_q.x, log_q.y, log_q.z) == approx((0, np.pi/2, 0), abs=1e-6)


def test_halfway_rotation_via_scaled_log():
    full = angle_axis_quaternion((0, 0, 1), np.pi)
    log_half = 0.5 * full.log()
    half = Q.exp(log_half)
    expected = angle_axis_quaternion((0, 0, 1), np.pi / 2)
    assert (half.x, half.y, half.z, half.w) == approx(tuple(expected))
