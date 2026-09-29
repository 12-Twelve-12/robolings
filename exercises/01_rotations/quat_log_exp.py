"""Track 01 - Rotations.

Conventions used in every track:

* Rotation matrices are 3x3, act on column vectors, and map body-frame
  coordinates to world-frame coordinates.
* Quaternions are ``[w, x, y, z]`` (scalar first), Hamilton convention.
* Everything is ``float64`` NumPy.
"""

import numpy as np


def quat_log(q):
    """Rotation vector of a unit quaternion.

    The rotation vector is the axis scaled by the angle, so its norm is the
    angle in radians. This is what an orientation controller wants: the error
    ``quat_log(q_des * conj(q))`` is a vector it can multiply by a gain.

    With ``q = [w, v]`` and ``theta = 2 * atan2(norm(v), w)``::

        quat_log(q) = theta * v / norm(v)

    Two details:

    * ``q`` and ``-q`` are the same rotation, but the formula above maps them
      to vectors of length ``theta`` and ``2 * pi - theta``. Flip ``q`` when
      ``w < 0`` so the result is always the short way round.
    * Near zero rotation ``norm(v)`` goes to zero and so does ``theta``. The
      ratio has a finite limit of 2, so return ``2 * v`` when
      ``norm(v)`` is tiny rather than dividing.

    Returns a ``(3,)`` array.
    """
    raise NotImplementedError("TODO: write this function")


def quat_exp(v):
    """Unit quaternion of a rotation vector, the inverse of ``quat_log``.

    With ``theta = norm(v)``::

        quat_exp(v) = [cos(theta / 2), sin(theta / 2) * v / theta]

    Same small-angle problem as in ``quat_log``: ``sin(theta / 2) / theta``
    tends to ``1 / 2``, so return ``[1, v / 2]`` normalised when ``theta`` is
    tiny.

    Returns a unit ``(4,)`` array.
    """
    raise NotImplementedError("TODO: write this function")
