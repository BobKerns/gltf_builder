'''
Tests for quaternian.py
This file contains unit tests for the functions in the quaternian.py module.
These tests cover quaternion multiplication, quaternion normalization,
quaternion conversion to matrix, and quaternion interpolation.
'''

import math

import gltf_builder.quaternion as Q
from gltf_builder.quaternion import (
    I, IDENTITY, J, K, MINUS_ONE, quaternion,
)
from gltf_builder.attribute_types import vector3

import pytest
from pytest import approx



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
    assert Q.multiply(q, IDENTITY) == q
    assert Q.multiply(IDENTITY, q) == q

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
    assert Q.multiply(a, b) == result


def test_full_formula_check():
    q1 = quaternion(1, 2, 3, 0)
    q2 = quaternion(4, 5, 6, 0)
    expected = quaternion(-3, 6, -3, -32)
    assert Q.multiply(q1, q2) == expected


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

    assert Q.multiply(q, q_inv) == approx(IDENTITY)


def test_non_commutativity():
    q1 = quaternion(1, 2, 3, 0)
    q2 = quaternion(4, 5, 6, 0)
    prod1 = Q.multiply(q1, q2)
    prod2 = Q.multiply(q2, q1)
    assert prod1 != approx(prod2)

def test_rotate_vector_90deg_z():
    # Rotate (1, 0, 0) by 90° around Z axis → should become (0, 1, 0)
    theta = math.pi / 2
    cos_t = math.cos(theta / 2)
    sin_t = math.sin(theta / 2)
    q = quaternion(0, 0, sin_t, cos_t)  # unit quaternion for 90° around Z

    v = vector3(1, 0, 0)
    v_rotated = rotate_vector(v, q)

    assert v_rotated == approx(vector3(0, 1, 0), abs=1e-6)

def test_rotate_vector_180deg_y():
    # Rotate (1, 0, 0) by 180° around Y axis → should become (-1, 0, 0)
    theta = math.pi
    q = (math.cos(theta/2), 0, math.sin(theta/2), 0)

    v = vector3(1, 0, 0)
    v_rotated = rotate_vector(v, q)

    assert v_rotated == approx(vector3(-1, 0, 0), abs=1e-6)


def test_q_norm():
    'Test quaternion normalization'
    q = quaternion(1, 2, 3, 4)
    normalized_q = Q.normalize(q)
    norm = (q.w**2 + q.x**2 + q.y**2 + q.z**2) ** 0.5
    assert normalized_q.w == approx(q.w / norm)
    assert normalized_q.x == approx(q.x / norm)
    assert normalized_q.y == approx(q.y / norm)
    assert normalized_q.z == approx(q.z / norm)