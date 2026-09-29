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
