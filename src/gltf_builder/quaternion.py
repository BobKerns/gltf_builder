'''
Quaternion utilities, per ChatGPT.
'''

import math

def axis_angle_to_quaternion(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
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
    tuple[float, float, float, float]
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
    
    return (qx, qy, qz, w)
