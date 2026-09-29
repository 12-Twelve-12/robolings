"""Track 02 - Kinematics of a serial chain.

A chain of ``n`` revolute joints is described by two arrays:

* ``origins``: ``(n, 4, 4)``. ``origins[i]`` is the fixed transform from the
  frame of link ``i - 1`` to the frame of joint ``i`` at zero angle. For
  ``i = 0`` the parent is the world frame. This is what a URDF ``<origin>``
  tag stores.
* ``axes``: ``(n, 3)``. ``axes[i]`` is the unit rotation axis of joint ``i``,
  expressed in the joint's own frame. This is the URDF ``<axis>`` tag.

The frame of link ``i`` in the world is therefore::

    T[i] = T[i - 1] @ origins[i] @ Rot(axes[i], q[i])
"""

import numpy as np


def ik_2link(x, y, l1, l2):
    """Both closed-form solutions of a two-link planar arm.

    ``dls_ik_step`` iterates because a general chain has no formula. A two
    link planar arm does, and having it here makes the point that iteration
    is the fallback rather than the starting point.

    With ``r2 = x**2 + y**2``, the law of cosines gives the elbow angle::

        cos(q2) = (r2 - l1**2 - l2**2) / (2 * l1 * l2)

    and the shoulder follows from::

        q1 = arctan2(y, x) - arctan2(l2 * sin(q2), l1 + l2 * cos(q2))

    Each reachable point has two solutions, one with ``q2 >= 0`` and one
    with ``q2 <= 0``, and they coincide on the boundary of the workspace.

    The target is unreachable when ``cos(q2)`` leaves ``[-1, 1]``: too far
    for ``r > l1 + l2``, and inside the hole in the middle for
    ``r < abs(l1 - l2)``. Both show up as the same test, which is why it is
    worth doing it this way rather than comparing radii.

    Returns a ``(2, 2)`` array, row 0 with ``q2 >= 0`` and row 1 with
    ``q2 <= 0``, each row being ``[q1, q2]``. Returns ``None`` when the
    target cannot be reached.
    """
    # >>> solution
    r2 = float(x) ** 2 + float(y) ** 2
    cos_q2 = (r2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    if abs(cos_q2) > 1.0 + 1e-12:
        return None
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(1.0 - cos_q2**2)
    out = np.empty((2, 2), dtype=np.float64)
    for row, sign in enumerate((1.0, -1.0)):
        q2 = np.arctan2(sign * sin_q2, cos_q2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        out[row] = (q1, q2)
    return out
    # <<< solution
