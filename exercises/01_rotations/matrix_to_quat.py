"""Track 01 - Rotations.

Conventions used in every track:

* Rotation matrices are 3x3, act on column vectors, and map body-frame
  coordinates to world-frame coordinates.
* Quaternions are ``[w, x, y, z]`` (scalar first), Hamilton convention.
* Everything is ``float64`` NumPy.
"""

import numpy as np


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
