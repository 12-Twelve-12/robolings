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


def dls_ik_step(J, err, damping):
    """One damped-least-squares inverse kinematics step.

    Solves for the joint update that reduces the task-space error ``err``::

        dq = J.T @ inv(J @ J.T + damping**2 * I) @ err

    The plain pseudo-inverse produces huge joint velocities near a
    singularity. The damping term trades a little accuracy for a bounded
    step, which is why this is the default on real arms and hands.

    ``J`` is ``(m, n)`` and ``err`` is ``(m,)``. Use ``np.linalg.solve``
    rather than forming the inverse.

    Returns a ``(n,)`` array.
    """
    # >>> solution
    J = np.asarray(J, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    m = J.shape[0]
    A = J @ J.T + (damping**2) * np.eye(m)
    return J.T @ np.linalg.solve(A, err)
    # <<< solution
