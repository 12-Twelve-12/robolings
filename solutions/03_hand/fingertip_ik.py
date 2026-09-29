"""Track 03 - Dexterous hands.

Hands differ from arms in three ways that these exercises cover: many joints
are mechanically coupled, the task is defined by several fingertips at once,
and the joint ranges are small enough that limits are hit all the time.
"""

import numpy as np


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
    # >>> solution
    q = np.clip(np.asarray(q0, dtype=np.float64), lower, upper)
    target = np.asarray(target, dtype=np.float64)
    for _ in range(iters):
        err = target - fk_fn(q)
        J = jac_fn(q)
        A = J @ J.T + (damping**2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(A, err)
        q = np.clip(q + dq, lower, upper)
    return q
    # <<< solution
