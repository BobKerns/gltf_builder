'''
Quaternion utilities, per ChatGPT.
'''

import math
from typing import NamedTuple, TypeAlias, overload

import numpy as np

from gltf_builder.core_types import Number, float01
from gltf_builder.attribute_types import (
    EPSILON, vector3,
    _Vector3, _Scale, scale,
)
from gltf_builder.matrix import matrix, _Matrix

dtype = np.dtype([('x', np.float32),
                       ('y', np.float32),
                       ('z', np.float32),
                       ('w', np.float32),])
'''
Numpy dtype for a quaternion.
'''


class Quaternion(NamedTuple):
    '''
    A quaterhion with x, y, z, and w components. Used here to
    represent a rotation or orientation.
    '''
    x: float
    y: float
    z: float
    w: float

    def __neg__(self):
        return Quaternion(-self.x, -self.y, -self.z, -self.w)
    
    def __mul__(self, other: 'Quaternion|Number'):
        if isinstance(other, Quaternion):
            x1, y1, z1, w1 = self
            x2, y2, z2, w2 = other

            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            return Quaternion(x, y, z, w)
        
        elif isinstance(other, (int, float, np.float32)):
            return Quaternion(self.x * other,
                               self.y * other,
                               self.z * other,
                               self.w * other)
        else:
            return NotImplemented

    def __rmul__(self, other: 'Quaternion|Number'):
        if isinstance(other, Quaternion):
            return other * self
        if isinstance(other, (int, float, np.float32)):
            return Quaternion(self.x * other,
                               self.y * other,
                               self.z * other,
                               self.w * other)
        else:
            return NotImplemented
        
    def __truediv__(self, other: 'Quaternion|Number'):
        if isinstance(other, (int, float, np.float32)):
            other = float(other)
            return Quaternion(self.x / other,
                                self.y / other,
                                self.z / other,
                                self.w / other)
        else:
            return NotImplemented
        
    def norm(self) -> float:
        return np.linalg.norm(self)
        
    def normalize(self) -> 'Quaternion':
        n = self.norm()
        if n < EPSILON:
            raise ValueError("Cannot normalize a zero-norm quaternion.")
        return Quaternion(self.x / n,
                           self.y / n,
                           self.z / n,
                           self.w / n)

    def rotate_vector(self, v: _Vector3) -> _Vector3:
        """
        Rotate a 3D vector using this quaternion.

        Parameters
        ----------
        v : Vector3
            A 3-element tuple (vx, vy, vz) representing the vector.

        Returns
        -------
        Vector3
            The rotated vector.
        """

        v_quat = Quaternion(*v, 0.0)
        q_inv = self.inverse()
        v_rot = (self * v_quat) * q_inv
        return vector3(*v_rot[:3])  # Extract rotated vector

    def inverse(self) -> 'Quaternion':
        """
        Compute the inverse of a quaternion.

        Returns
        -------
        _Quaternion
            The inverse quaternion.
        """
        norm_sq = np.dot(self, self)  # Equivalent to |q|^2
        if norm_sq < EPSILON:
            raise ValueError("Cannot invert a zero-norm quaternion.")
        return Quaternion(*(v / norm_sq for v in self.conjugate()))
    

    def conjugate(self) -> 'Quaternion':
        """
        Compute the conjugate of a quaternion.

        Returns
        -------
        Quaternion
            The conjugate quaternion (-x, -y, -z, w).
        """
        return Quaternion(-self[0], -self[1], -self[2], self[3])

    def log(self) -> 'Quaternion':
        """
        Compute the logarithm of a unit quaternion.

        Returns
        -------
        Quaternion
            The logarithm of the quaternion (axis-angle representation) as
            a pure-imaginnary Quaternion.
        """
        v = Quaternion(*self[:3], 0)
        w = self[3]
        theta = np.arccos(w)
        sin_theta = np.sin(theta)

        if sin_theta > 1e-6:  # Avoid division by zero
            return Quaternion(*(theta * v / sin_theta))
        return Quaternion(0.0, 0.0, 0.0, 0.0)  # If theta is zero, log is zero

    @staticmethod
    def exp(v: '_Vector3|Quaternion') -> 'Quaternion':
        """
        Compute the exponential map of an axis-angle representation.

        Parameters
        ----------
        v : Vector3|Quaternion
            A 3D vector representing an axis-angle rotation or a
            pure-imaginary quaternion (quaternion with w=0).
            The vector should not be normalized, as that will lose
            the angle information.

        Returns
        -------
        Quaternion
            The corresponding quaternion.
        """
        theta = np.linalg.norm(v)
        if theta > 1e-6:
            axis = vector3(*v[:3]) / theta
            sin_theta = np.sin(theta)
            return Quaternion(*np.concatenate([axis * sin_theta, [np.cos(theta)]]))
        return IDENTITY # Identity
    
    @staticmethod
    def decompose_trs(m: _Matrix) -> tuple[_Vector3, 'Quaternion', _Scale]:
        """
        Decompose a 4x4 transformation matrix into translation, rotation (as a quaternion), and scale components.

        Parameters
        ----------
        m : Matrix
            A 4x4 transformation matrix. Can be a NumPy array or a sequence of sequences.

        Returns
        -------
        tuple
            - translation : _Vector3
                Translation vector (tx, ty, tz).
            - rotation_quaternion : _Quaternion
                Rotation quaternion (x, y, z, w).
            - scale : _Scale
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
        _scale = np.linalg.norm(rot_scale_mat, axis=0)
        rotation_matrix = rot_scale_mat / _scale

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

        rotation_quaternion = Quaternion(x, y, z, w)
        return _Vector3(*translation), rotation_quaternion, scale(*_scale)

    def to_matrix(self) -> _Matrix:
        """
        Convert a quaternion (x, y, z, w) to a 4x4 rotation matrix.

        Parameters
        ----------
        quaternion : Qaternion
            A 1D array or sequence representing the quaternion (x, y, z, w).

        Returns
        -------
        Matrix
            A 4x4 rotation matrix.
        """
        q = self.normalize()

        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return matrix((
            (1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 1),
            (2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 1),
            (2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 1),
            (0, 0, 0, 1)
        ))
    
    @staticmethod
    def from_matrix(m: _Matrix) -> 'Quaternion':
        """
        Convert a 3x3 or 4x4 rotation matrix to a quaternion (x, y, z, w).

        Parameters
        ----------
        matrix : Matrix
            A 3x3 or 4x4 rotation matrix. Can be a NumPy array or a sequence of sequences.

        Returns
        -------
        Quaternion
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
        return Quaternion(*(q / np.linalg.norm(q)))
    
    def to_axis_angle(self) -> tuple[_Vector3, float]:
        """
        Convert a quaternion (x, y, z, w) to axis-angle representation.

        Parameters
        ----------
        q : Quaternion
            The quaternion (x, y, z, w).

        Returns
        -------
        tuple[_Vector3, float]
            A unit axis (x, y, z) and an angle in radians.
        """
        x, y, z, w = self
        angle = 2 * math.acos(w)
        sin_half_angle = math.sqrt(x**2 + y**2 + z**2)

        if sin_half_angle < 1e-8:  # Avoid division by zero (identity rotation case)
            return (vector3(1.0, 0.0, 0.0), 0.0)  # Default axis (arbitrary when angle is zero)

        axis = vector3(x / sin_half_angle, y / sin_half_angle, z / sin_half_angle)
        return axis, angle


    @staticmethod
    def from_axis_angle(axis: _Vector3, angle: float) -> 'Quaternion':
        """
        Convert a rotation about an arbitrary axis to a quaternion in (x, y, z, w) order.

        Parameters
        ----------
        axis : Vector3
            A 3-element tuple (x, y, z) representing the rotation axis. It does not need to be normalized.
        angle : float
            Rotation angle in radians.

        Returns
        -------
        Quaternion
            A 4-element tuple representing the quaternion (x, y, z, w).

        Raises
        ------
        ValueError
            If the axis vector is a zero vector (norm is zero).
        """
        x, y, z = vector3(axis)
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
        
        return Quaternion(qx, qy, qz, w)

    @staticmethod
    def from_euler(yaw: float, pitch: float, roll: float) -> 'Quaternion':
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
        Quaternion
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

        return Quaternion(qx, qy, qz, w)

    @staticmethod
    def slerp(
            q1: 'QuaternionSpec', 
            q2: 'QuaternionSpec',
            t: float01
        ) -> 'Quaternion':
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
            return Quaternion(qx / norm, qy / norm, qz / norm, qw / norm)

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

        return Quaternion(qx, qy, qz, qw)


IDENTITY = Quaternion(0.0, 0.0, 0.0, 1.0)
MINUS_ONE = Quaternion(0, 0, 0, -1)

I = Quaternion(1.0, 0.0, 0.0, 0.0) # noqa
J = Quaternion(0.0, 1.0, 0.0, 0.0)
K = Quaternion(0.0, 0.0, 1.0, 0.0)
W = Quaternion(0.0, 0.0, 0.0, 1.0)

@overload
def quaternion(qx: 'QuaternionSpec') -> Quaternion:
    ...
@overload
def quaternion(qx: Number,
               y: Number|None=None,
               z: Number|None=None,
               w: Number|None=None) -> Quaternion:
    ...
def quaternion(qx: 'QuaternionSpec|Number',
               y: Number|None=None,
               z: Number|None=None,
               w: Number|None=None) -> Quaternion:
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
    if isinstance(qx, Quaternion):
        return qx
    if y is None:
        return Quaternion(*qx)
    return Quaternion(qx, y, z, w)


QuaternionSpec: TypeAlias = (
    Quaternion|tuple[Number, Number, Number, Number]
    |np.ndarray[tuple[int], np.float32]
)
'''
A Quaternion, or a type convertible to a Quaternion.
This is a 4-element tuple or a 1D array of 4 floats.
'''
