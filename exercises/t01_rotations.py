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
    raise NotImplementedError("TODO: write this function")


def quat_mul(q1, q2):
    """Hamilton product ``q1 * q2`` of two ``[w, x, y, z]`` quaternions.

    Rotating by ``q2`` first and then by ``q1`` is the rotation ``q1 * q2``.
    The product is not commutative.

    Returns a ``(4,)`` array. Do not normalise the result.
    """
    raise NotImplementedError("TODO: write this function")


def quat_to_matrix(q):
    """Rotation matrix of a ``[w, x, y, z]`` quaternion.

    ``q`` may not be exactly unit length; normalise it first.

    Returns a ``(3, 3)`` array.
    """
    raise NotImplementedError("TODO: write this function")


def matrix_to_quat(R):
    """Unit quaternion ``[w, x, y, z]`` of a rotation matrix, with ``w >= 0``.

    The textbook formula ``w = sqrt(1 + trace) / 2`` divides by ``w`` to get
    the vector part, which blows up for rotations near 180 degrees where
    ``w`` is close to zero. Branch on the largest of ``trace, R[0,0],
    R[1,1], R[2,2]`` so that you always divide by a number that is not small.

    ``q`` and ``-q`` are the same rotation; return the one with ``w >= 0``.

    Returns a ``(4,)`` array.
    """
    raise NotImplementedError("TODO: write this function")


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
    raise NotImplementedError("TODO: write this function")
