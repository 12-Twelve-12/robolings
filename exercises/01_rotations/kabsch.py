"""Track 01 - Rotations.

Conventions used in every track:

* Rotation matrices are 3x3, act on column vectors, and map body-frame
  coordinates to world-frame coordinates.
* Quaternions are ``[w, x, y, z]`` (scalar first), Hamilton convention.
* Everything is ``float64`` NumPy.
"""

import numpy as np


def kabsch(P, Q):
    """Rigid transform that best maps the points ``P`` onto the points ``Q``.

    Finds the rotation ``R`` and translation ``t`` that minimise::

        sum_i || R @ P[i] + t - Q[i] ||^2

    This is how you align a motion-capture frame to a robot frame, or
    register a calibration board seen by two sensors.

    The Kabsch algorithm:

    1. Subtract the centroid from each point set.
    2. ``H = P_centered.T @ Q_centered``
    3. ``U, S, Vt = svd(H)``
    4. ``d = sign(det(Vt.T @ U.T))``
    5. ``R = Vt.T @ diag(1, 1, d) @ U.T``
    6. ``t = centroid_Q - R @ centroid_P``

    Step 4 is the one that gets left out. Without it, noisy or nearly flat
    data can produce a reflection: a matrix that is orthogonal and fits the
    points well, but has determinant -1 and is not a rotation.

    ``P`` and ``Q`` are ``(N, 3)`` with corresponding rows.

    Returns ``(R, t)`` with shapes ``(3, 3)`` and ``(3,)``.
    """
    raise NotImplementedError("TODO: write this function")
