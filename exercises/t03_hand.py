"""Track 03 - Dexterous hands.

Hands differ from arms in three ways that these exercises cover: many joints
are mechanically coupled, the task is defined by several fingertips at once,
and the joint ranges are small enough that limits are hit all the time.
"""

import numpy as np


def expand_mimic(q_active, n_joints, active_idx, mimic):
    """Full joint vector of a hand with coupled joints.

    A tendon- or linkage-driven finger has fewer motors than joints. URDF
    expresses this with ``<mimic>``: a passive joint follows an active one as
    ``q[joint] = multiplier * q[source] + offset``.

    * ``q_active``: ``(k,)`` angles of the actuated joints.
    * ``n_joints``: total number of joints in the hand.
    * ``active_idx``: ``(k,)`` index of each actuated joint in the full
      vector.
    * ``mimic``: list of ``(joint, source, multiplier, offset)``. ``source``
      is an index into the full vector and always refers to an actuated
      joint.

    Joints that are neither actuated nor mimicking stay at zero.

    Returns a ``(n_joints,)`` array.
    """
    raise NotImplementedError("TODO: write this function")


def retarget_cost(human_vecs, robot_vecs, scale, q, q_prev, beta):
    """Cost of vector-based hand retargeting.

    Retargeting maps a human hand pose onto a robot hand whose proportions
    are different. Matching joint angles does not work, so the usual choice
    is to match *vectors* such as wrist-to-fingertip::

        cost = 0.5 * sum_i || scale * human_vecs[i] - robot_vecs[i] ||^2
             + 0.5 * beta * || q - q_prev ||^2

    ``scale`` accounts for the size difference between the two hands. The
    second term penalises jumping away from the previous solution, which
    keeps the motion smooth when the first term has several minima.

    ``human_vecs`` and ``robot_vecs`` are ``(k, 3)``; ``q`` and ``q_prev``
    are ``(n,)``.

    Returns a Python ``float``.
    """
    raise NotImplementedError("TODO: write this function")


def fingertip_ik(fk_fn, jac_fn, q0, target, lower, upper, iters, damping):
    """Move one fingertip to ``target`` without leaving the joint limits.

    Repeat ``iters`` times:

    1. ``err = target - fk_fn(q)``
    2. ``J = jac_fn(q)``, a ``(3, n)`` position Jacobian
    3. ``dq = J.T @ solve(J @ J.T + damping**2 * I, err)``
    4. ``q = clip(q + dq, lower, upper)``

    Clipping inside the loop matters. If you clip only once at the end, the
    solver spends its iterations pushing a joint that is already at its
    limit and the other joints never compensate.

    * ``fk_fn(q)`` returns the fingertip position, ``(3,)``.
    * ``jac_fn(q)`` returns the position Jacobian, ``(3, n)``.

    Returns the final ``q``, a ``(n,)`` array.
    """
    raise NotImplementedError("TODO: write this function")
