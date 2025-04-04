'''
Quaternion utilities, per ChatGPT.
'''

import math
from typing import NamedTuple, TypeAlias, overload

import numpy as np

from gltf_builder.attribute_types import (
    EPSILON, Vector3, vector3,
    _Vector3, _Scale, Scale
)
from gltf_builder.matrix import matrix, Matrix

dtype = np.dtype([('x', np.float32),
                       ('y', np.float32),
                       ('z', np.float32),
                       ('w', np.float32),])
'''
Numpy dtype for a quaternion.
'''


class _Quaternion(NamedTuple):
    '''
    A quaterhion with x, y, z, and w components. Used here to
    represent a rotation or orientation.
    '''
    x: float
    y: float
    z: float
    w: float

    def __neg__(self):
        return _Quaternion(-self.x, -self.y, -self.z, -self.w)
    
    def __mul__(self, other):
        if isinstance(other, _Quaternion):
            return multiply(self, other)
        elif isinstance(other, (int, float)):
            return _Quaternion(self.x * other,
                               self.y * other,
                               self.z * other,
                               self.w * other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other).__name__}'")

    def __rmul__(self, other):
        if isinstance(other, _Quaternion):
            return multiply(other, self)
        if isinstance(other, (int, float)):
            return _Quaternion(self.x * other,
                               self.y * other,
                               self.z * other,
                               self.w * other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other).__name__}'")
        
        def __truediv__(self, other):
            if isinstance(other, (int, float)):
                return _Quaternion(self.x / other,
                                   self.y / other,
                                   self.z / other,
                                   self.w / other)
            else:
                raise TypeError(f"Unsupported operand type(s) for /: 'Quaternion' and '{type(other).__name__}'")

IDENTITY = _Quaternion(0.0, 0.0, 0.0, 1.0)
MINUS_ONE = _Quaternion(0, 0, 0, -1)

I = _Quaternion(1.0, 0.0, 0.0, 0.0) # noqa
J = _Quaternion(0.0, 1.0, 0.0, 0.0)
K = _Quaternion(0.0, 0.0, 1.0, 0.0)
W = _Quaternion(0.0, 0.0, 0.0, 1.0)

@overload
def quaternion(qx: _Quaternion|tuple[float, float, float, float]) -> _Quaternion:
    ...
@overload
def quaternion(qx: float,
               y: float|None=None,
               z: float|None=None,
               w: float|None=None) -> _Quaternion:
    ...
def quaternion(qx: _Quaternion|float,
               y: float|None=None,
               z: float|None=None,
               w: float|None=None) -> _Quaternion:
    '''
    Create a quaternion from a Quaternion object or components.

    Parameters
    ----------
    qx : Quaternion or float
        A Quaternion object or the x component of the quaternion.
    y : float, optional
        The y component of the quaternion.
    z : float, optional
        The z component of the quaternion.
    w : float, optional
        The w component of the quaternion.

    Returns
    -------
    Quaternion
        A quaternion with x, y, z, and w components.
    '''
    if isinstance(qx, _Quaternion):
        return qx
    if y is None:
        return _Quaternion(*qx)
    return _Quaternion(qx, y, z, w)


Quaternion: TypeAlias = (
    _Quaternion|tuple[float, float, float, float]
    |np.ndarray[tuple[int], np.float32]
)

def from_axis_angle(axis: tuple[float, float, float], angle: float) -> _Quaternion:
    """
    Convert a rotation about an arbitrary axis to a quaternion in (x, y, z, w) order.

    Parameters
    ----------
    axis : tuple[float, float, float]
        A 3-element tuple (x, y, z) representing the rotation axis. It does not need to be normalized.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    _Quaternion
        A 4-element tuple representing the quaternion (x, y, z, w).

    Raises
    ------
    ValueError
        If the axis vector is a zero vector (norm is zero).
    """
    x, y, z = axis
    norm = math.sqrt(x**2 + y**2 + z**2)
    
    if norm == 0:
        raise ValueError("Axis vector must be nonzero.")
    
    # Normalize the axis
    x /= norm
    y /= norm
    z /= norm
    
    half_angle = angle / 2
    sin_half_angle = math.sin(half_angle)
    
    qx = x * sin_half_angle
    qy = y * sin_half_angle
    qz = z * sin_half_angle
    w = math.cos(half_angle)
    
    return _Quaternion(qx, qy, qz, w)


def from_euler(yaw: float, pitch: float, roll: float) -> _Quaternion:
    """
    Convert Euler angles (yaw, pitch, roll) to a quaternion in (x, y, z, w) order.

    Parameters
    ----------
    yaw : float
        Rotation around the z-axis, in radians.
    pitch : float
        Rotation around the y-axis, in radians.
    roll : float
        Rotation around the x-axis, in radians.

    Returns
    -------
    _Quaternion
        A quaternion (x, y, z, w) representing the same rotation.

    Notes
    -----
    The Euler angles are assumed to follow the Tait-Bryan convention in the ZYX order:
    1. `yaw` (rotation around z-axis)
    2. `pitch` (rotation around y-axis)
    3. `roll` (rotation around x-axis)
    """

    half_yaw = yaw / 2
    half_pitch = pitch / 2
    half_roll = roll / 2

    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)

    # Compute quaternion components
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return _Quaternion(qx, qy, qz, w)


def to_axis_angle(q: Quaternion) -> tuple[Vector3, float]:
    """
    Convert a quaternion (x, y, z, w) to axis-angle representation.

    Parameters
    ----------
    q : Quaternion
        The quaternion (x, y, z, w).

    Returns
    -------
    tuple[tuple[float, float, float], float]
        A unit axis (x, y, z) and an angle in radians.
    """
    x, y, z, w = q
    angle = 2 * math.acos(w)
    sin_half_angle = math.sqrt(x**2 + y**2 + z**2)

    if sin_half_angle < 1e-8:  # Avoid division by zero (identity rotation case)
        return (vector3(1.0, 0.0, 0.0), 0.0)  # Default axis (arbitrary when angle is zero)

    axis = vector3(x / sin_half_angle, y / sin_half_angle, z / sin_half_angle)
    return axis, angle


def slerp(q1: Quaternion, 
        q2: Quaternion,
        t: float) -> _Quaternion:
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.

    Parameters
    ----------
    q1 : tuple[float, float, float, float]
        The starting quaternion (x, y, z, w).
    q2 : tuple[float, float, float, float]
        The target quaternion (x, y, z, w).
    t : float
        Interpolation factor (0 = q1, 1 = q2).

    Returns
    -------
    Quaternion
        The interpolated quaternion (x, y, z, w).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # Compute the dot product (cosine of the angle between quaternions)
    dot = x1*x2 + y1*y2 + z1*z2 + w1*w2

    # If the dot product is negative, negate one quaternion to take the shorter path
    if dot < 0.0:
        x2, y2, z2, w2 = -x2, -y2, -z2, -w2
        dot = -dot

    # Clamp dot product to avoid numerical errors
    dot = max(min(dot, 1.0), -1.0)

    # Compute interpolation weights
    if dot > 0.9995:  # If very close, use linear interpolation to avoid numerical instability
        qx = x1 + t * (x2 - x1)
        qy = y1 + t * (y2 - y1)
        qz = z1 + t * (z2 - z1)
        qw = w1 + t * (w2 - w1)
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        return _Quaternion(qx / norm, qy / norm, qz / norm, qw / norm)

    theta_0 = math.acos(dot)  # Initial angle
    sin_theta_0 = math.sin(theta_0)
    
    theta = theta_0 * t  # Scaled angle
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    qx = s0 * x1 + s1 * x2
    qy = s0 * y1 + s1 * y2
    qz = s0 * z1 + s1 * z2
    qw = s0 * w1 + s1 * w2

    return _Quaternion(qx, qy, qz, qw)


def from_matrix(m: Matrix):
    """
    Convert a 3x3 or 4x4 rotation matrix to a quaternion (x, y, z, w).

    Parameters
    ----------
    matrix : Matrix
        A 3x3 or 4x4 rotation matrix. Can be a NumPy array or a sequence of sequences.

    Returns
    -------
    np.ndarray
        A 1D array representing the quaternion (x, y, z, w).
    """
    # Convert input to NumPy array
    mat = matrix(m).as_array()

    # Extract the rotation part
    if mat.shape == (4, 4):
        rot_mat = mat[:3, :3]
    elif mat.shape == (3, 3):
        rot_mat = mat
    else:
        raise ValueError("Input matrix must be 3x3 or 4x4.")

    # Compute the trace of the matrix
    trace = np.trace(rot_mat)

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rot_mat[2, 1] - rot_mat[1, 2]) / s
        y = (rot_mat[0, 2] - rot_mat[2, 0]) / s
        z = (rot_mat[1, 0] - rot_mat[0, 1]) / s
    else:
        if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
            w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            x = 0.25 * s
            y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
            w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            y = 0.25 * s
            z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
            w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=dtype)
    return _Quaternion(*(q / np.linalg.norm(q)))


def to_matrix(quaternion) -> Matrix:
    """
    Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Parameters
    ----------
    quaternion : array-like
        A 1D array or sequence representing the quaternion (x, y, z, w).

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix.
    """
    # Convert input to NumPy array
    q = np.array(quaternion, dtype=float)
    q = q / np.linalg.norm(q)  # Normalize the quaternion

    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return matrix(rot_matrix)


def decompose_trs(m: Matrix) -> tuple[Vector3, Quaternion, Scale]:
    """
    Decompose a 4x4 transformation matrix into translation, rotation (as a quaternion), and scale components.

    Parameters
    ----------
    m : Matrox
        A 4x4 transformation matrix. Can be a NumPy array or a sequence of sequences.

    Returns
    -------
    tuple
        - translation : np.ndarray of shape (3,)
            Translation vector (tx, ty, tz).
        - rotation_quaternion : Quaternion
            Rotation quaternion (x, y, z, w).
        - scale : np.ndarray of shape (3,)
            Scale factors (sx, sy, sz).
    """
    # Convert input to NumPy array
    mat = matrix(m).as_array()

    if mat.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4.")

    # Extract translation
    translation = mat[:3, 3]

    # Extract rotation and scale
    rot_scale_mat = mat[:3, :3]
    scale = np.linalg.norm(rot_scale_mat, axis=0)
    rotation_matrix = rot_scale_mat / scale

    # Convert rotation matrix to quaternion
    trace = np.trace(rotation_matrix)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s

    rotation_quaternion = _Quaternion(x, y, z, w)
    return _Vector3(*translation), rotation_quaternion, _Scale(*scale)


def multiply(q1: Quaternion, q2: Quaternion):
    """
    Multiply two quaternions.

    Parameters
    ----------
    q1 : array-like
        First quaternion as a 4-element array-like (x, y, z, w).
    q2 : array-like
        Second quaternion as a 4-element array-like (x, y, z, w).

    Returns
    -------
    _Quaternion:
        The resulting quaternion (x, y, z, w) from the multiplication.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return _Quaternion(x, y, z, w)


def conjugate(q: Quaternion) -> _Quaternion:
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    q : array-like
        A 4-element array-like representing the quaternion (x, y, z, w).

    Returns
    -------
    np.ndarray
        The conjugate quaternion (-x, -y, -z, w).
    """
    return _Quaternion(-q[0], -q[1], -q[2], q[3])


def inverse(q: Quaternion) -> np.ndarray:
    """
    Compute the inverse of a quaternion.

    Parameters
    ----------
    q : array-like
        A 4-element array-like representing the quaternion (x, y, z, w).

    Returns
    -------
    np.ndarray
        The inverse quaternion.
    """
    q = np.array(q, dtype=float)
    norm_sq = np.dot(q, q)  # Equivalent to |q|^2
    if norm_sq == 0:
        raise ValueError("Cannot invert a zero-norm quaternion.")
    return _Quaternion(v / norm_sq for v in conjugate(q))


def norm(q: Quaternion) -> float:
    """
    Compute the norm (magnitude) of a quaternion.

    Parameters
    ----------
    q : array-like
        A 4-element array-like representing the quaternion (x, y, z, w).

    Returns
    -------
    float
        The norm of the quaternion.
    """
    return np.linalg.norm(q)


def normalize(q: Quaternion) -> np.ndarray:
    """
    Normalize a quaternion to unit length.

    Parameters
    ----------
    q : array-like
        A 4-element array-like representing the quaternion (x, y, z, w).

    Returns
    -------
    np.ndarray
        The normalized quaternion.
    """
    n = norm(q)
    if n < EPSILON:
        raise ValueError("Cannot normalize a zero-norm quaternion.")
    return _Quaternion(*(v / n for v in q))


def rotate_vector(q: Quaternion, v: Vector3) -> Vector3:
    """
    Rotate a 3D vector using a quaternion.

    Parameters
    ----------
    q : array-like
        A 4-element array-like representing the rotation quaternion (x, y, z, w).
    v : array-like
        A 3-element array-like representing the vector (vx, vy, vz).

    Returns
    -------
    np.ndarray
        The rotated vector.
    """
    v_quat = np.array([v[0], v[1], v[2], 0.0])
    q_inv = inverse(q)
    v_rot = multiply(multiply(q, v_quat), q_inv)
    return vector3(*v_rot[:3])  # Extract rotated vector


def log(q: Quaternion) -> np.ndarray:
    """
    Compute the logarithm of a unit quaternion.

    Parameters
    ----------
    q : array-like
        A 4-element unit quaternion (x, y, z, w).

    Returns
    -------
    np.ndarray
        The logarithm of the quaternion (axis-angle representation).
    """
    v = q[:3]
    w = q[3]
    theta = np.arccos(w)
    sin_theta = np.sin(theta)

    if sin_theta > 1e-6:  # Avoid division by zero
        return theta * v / sin_theta
    return vector3()  # If theta is zero, log is zero


def exp(v: Vector3) -> np.ndarray:
    """
    Compute the exponential map of an axis-angle representation.

    Parameters
    ----------
    v : array-like
        A 3-element array representing an axis-angle rotation.

    Returns
    -------
    np.ndarray
        The corresponding quaternion.
    """
    v = np.array(v, dtype=np.float32)
    theta = np.linalg.norm(v)
    if theta > 1e-6:
        axis = v / theta
        sin_theta = np.sin(theta)
        return _Quaternion(*np.concatenate([axis * sin_theta, [np.cos(theta)]]))
    return IDENTITY # Identity quaternion
