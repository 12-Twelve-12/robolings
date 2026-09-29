"""Track 01 - Rotations.

Conventions used in every track:

* Rotation matrices are 3x3, act on column vectors, and map body-frame
  coordinates to world-frame coordinates.
* Quaternions are ``[w, x, y, z]`` (scalar first), Hamilton convention.
* Everything is ``float64`` NumPy.
"""

import numpy as np


def rodrigues(axis, angle):
    """Rotation matrix for a rotation of ``angle`` radians about ``axis``.

    ``axis`` is a 3-vector that is *not* guaranteed to be unit length, so
    normalise it first. Rodrigues' formula, with ``K`` the skew-symmetric
    matrix of the unit axis::

        R = I + sin(angle) * K + (1 - cos(angle)) * K @ K

    Returns a ``(3, 3)`` array.
    """
    # >>> solution
    axis = np.asarray(axis, dtype=np.float64)
    k = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    # <<< solution


def quat_mul(q1, q2):
    """Hamilton product ``q1 * q2`` of two ``[w, x, y, z]`` quaternions.

    Rotating by ``q2`` first and then by ``q1`` is the rotation ``q1 * q2``.
    The product is not commutative.

    Returns a ``(4,)`` array. Do not normalise the result.
    """
    # >>> solution
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    # <<< solution


def quat_to_matrix(q):
    """Rotation matrix of a ``[w, x, y, z]`` quaternion.

    ``q`` may not be exactly unit length; normalise it first.

    Returns a ``(3, 3)`` array.
    """
    # >>> solution
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    # <<< solution


def matrix_to_quat(R):
    """Unit quaternion ``[w, x, y, z]`` of a rotation matrix, with ``w >= 0``.

    The textbook formula ``w = sqrt(1 + trace) / 2`` divides by ``w`` to get
    the vector part, which blows up for rotations near 180 degrees where
    ``w`` is close to zero. Branch on the largest of ``trace, R[0,0],
    R[1,1], R[2,2]`` so that you always divide by a number that is not small.

    ``q`` and ``-q`` are the same rotation; return the one with ``w >= 0``.

    Returns a ``(4,)`` array.
    """
    # >>> solution
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(1.0 + trace)
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)
    # <<< solution


def slerp(q0, q1, t):
    """Spherical linear interpolation between two unit quaternions.

    ``t = 0`` gives ``q0`` and ``t = 1`` gives ``q1`` (up to sign).

    Two details that matter on a real robot:

    * ``q1`` and ``-q1`` are the same orientation. If ``dot(q0, q1) < 0``,
      flip ``q1`` so the interpolation takes the short way round.
    * When the two quaternions are nearly identical, ``sin(theta)`` is close
      to zero. Fall back to normalised linear interpolation when
      ``dot > 0.9995``.

    Returns a unit ``(4,)`` array.
    """
    # >>> solution
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    q = (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / sin_theta
    return q / np.linalg.norm(q)
    # <<< solution
