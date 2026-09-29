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


def geometric_jacobian(frames, axes, tip_offset):
    """Geometric Jacobian of a point attached to the last link.

    ``frames`` is the output of ``forward_kinematics``. ``tip_offset`` is the
    position of the point of interest in the last link's frame, for example
    a fingertip.

    Column ``i`` of the result describes what joint ``i`` does to the tip::

        a_i = R_i @ axes[i]            joint axis in the world frame
        p_i = frames[i][:3, 3]         joint position in the world frame
        J[:3, i] = cross(a_i, p_tip - p_i)     linear velocity
        J[3:, i] = a_i                         angular velocity

    Returns a ``(6, n)`` array: linear rows on top, angular rows below.
    """
    # >>> solution
    frames = np.asarray(frames, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    tip_offset = np.asarray(tip_offset, dtype=np.float64)
    n = frames.shape[0]
    p_tip = frames[-1][:3, :3] @ tip_offset + frames[-1][:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        a = frames[i][:3, :3] @ (axes[i] / np.linalg.norm(axes[i]))
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(a, p_tip - p)
        J[3:, i] = a
    return J
    # <<< solution
