"""Track 01 - Rotations.

Conventions used in every track:

* Rotation matrices are 3x3, act on column vectors, and map body-frame
  coordinates to world-frame coordinates.
* Quaternions are ``[w, x, y, z]`` (scalar first), Hamilton convention.
* Everything is ``float64`` NumPy.
"""

import numpy as np


def quat_to_matrix(q):
    """Rotation matrix of a ``[w, x, y, z]`` quaternion.

    ``q`` may not be exactly unit length; normalise it first.

    Returns a ``(3, 3)`` array.
    """
    raise NotImplementedError("TODO: write this function")
