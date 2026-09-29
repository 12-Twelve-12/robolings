"""Track 01 - Rotations.

Conventions used in every track:

* Rotation matrices are 3x3, act on column vectors, and map body-frame
  coordinates to world-frame coordinates.
* Quaternions are ``[w, x, y, z]`` (scalar first), Hamilton convention.
* Everything is ``float64`` NumPy.
"""

import numpy as np


def quat_mul(q1, q2):
    """Hamilton product ``q1 * q2`` of two ``[w, x, y, z]`` quaternions.

    Rotating by ``q2`` first and then by ``q1`` is the rotation ``q1 * q2``.
    The product is not commutative.

    Returns a ``(4,)`` array. Do not normalise the result.
    """
    raise NotImplementedError("TODO: write this function")
