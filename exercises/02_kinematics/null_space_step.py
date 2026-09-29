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


def null_space_step(J, dx, q_secondary):
    """Task-space step plus a secondary objective that cannot disturb it.

    A redundant arm has more joints than the task constrains, and the spare
    motion is free to do something else: stay away from a joint limit, keep
    the elbow clear of an obstacle, drift back towards a comfortable
    posture. The standard form splits the two::

        dq = pinv(J) @ dx + (I - pinv(J) @ J) @ q_secondary

    The first term is the minimum-norm solution of ``J @ dq = dx``. The
    second projects ``q_secondary`` onto the null space of ``J``, so
    whatever it asks for, ``J @ dq`` is unchanged.

    That invariance is the property worth testing, and it is exactly what an
    implementation without the projector loses: adding ``q_secondary``
    directly still moves the tip in roughly the right direction, so a test
    that only checks the first term passes.

    ``J`` is ``(m, n)`` with ``m <= n``, ``dx`` is ``(m,)`` and
    ``q_secondary`` is ``(n,)``. Use ``np.linalg.pinv``.

    Returns an ``(n,)`` array.
    """
    raise NotImplementedError("TODO: write this function")
