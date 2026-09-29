"""Track 03 - Dexterous hands.

Hands differ from arms in three ways that these exercises cover: many joints
are mechanically coupled, the task is defined by several fingertips at once,
and the joint ranges are small enough that limits are hit all the time.
"""

import numpy as np


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
    # >>> solution
    human_vecs = np.asarray(human_vecs, dtype=np.float64)
    robot_vecs = np.asarray(robot_vecs, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q_prev = np.asarray(q_prev, dtype=np.float64)
    diff = scale * human_vecs - robot_vecs
    return float(0.5 * np.sum(diff * diff) + 0.5 * beta * np.sum((q - q_prev) ** 2))
    # <<< solution
