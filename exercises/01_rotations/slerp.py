"""Track 01 - Rotations.

Conventions used in every track:

* Rotation matrices are 3x3, act on column vectors, and map body-frame
  coordinates to world-frame coordinates.
* Quaternions are ``[w, x, y, z]`` (scalar first), Hamilton convention.
* Everything is ``float64`` NumPy.
"""

import numpy as np


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
