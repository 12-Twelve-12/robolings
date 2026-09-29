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


def manipulability(J):
    """Yoshikawa's manipulability measure.

    How much velocity the tip can produce in the worst direction, in one
    number::

        w = sqrt(det(J @ J.T))

    It is the volume of the ellipsoid that unit joint velocities map to. It
    goes to zero exactly at a singularity, which is what makes it useful as
    an early warning: the damping in ``dls_ik_step`` exists to survive the
    configurations this measure is telling you about.

    Computing it as written squares the condition number. Multiplying the
    singular values of ``J`` gives the same answer and keeps more digits::

        w = prod(svdvals(J))

    Note that ``det(J)`` is not a substitute. It is only defined for a
    square ``J``, and for a redundant arm the whole point is that ``J`` is
    not square. Nor is ``J.T @ J`` interchangeable with ``J @ J.T``: with
    more joints than task dimensions the former is singular and its
    determinant is zero everywhere.

    ``J`` is ``(m, n)``. When ``m > n`` the task has more dimensions than
    the arm has joints and the measure is 0.

    Returns a Python ``float``.
    """
    # >>> solution
    J = np.asarray(J, dtype=np.float64)
    if J.shape[0] > J.shape[1]:
        return 0.0
    return float(np.prod(np.linalg.svd(J, compute_uv=False)))
    # <<< solution
