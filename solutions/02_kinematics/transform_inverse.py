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


def transform_inverse(T):
    """Inverse of a homogeneous transform, without a general matrix inverse.

    For ``T = [[R, p], [0, 1]]`` the inverse is ``[[R.T, -R.T @ p], [0, 1]]``.
    This is cheaper and numerically better than ``np.linalg.inv``, and it is
    what runs in a control loop.

    Do not call ``np.linalg.inv`` or ``np.linalg.solve``.

    Returns a ``(4, 4)`` array.
    """
    # >>> solution
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out
    # <<< solution
