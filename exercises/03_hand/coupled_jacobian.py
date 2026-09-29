"""Track 03 - Dexterous hands.

Hands differ from arms in three ways that these exercises cover: many joints
are mechanically coupled, the task is defined by several fingertips at once,
and the joint ranges are small enough that limits are hit all the time.
"""

import numpy as np


def coupled_jacobian(J_full, n_joints, active_idx, mimic):
    """Jacobian of a coupled hand with respect to its actuated joints.

    ``expand_mimic`` builds the full joint vector from the actuated one.
    This is the derivative of that map. Writing it as ``q_full = C @ q_act``
    with ``C`` of shape ``(n_joints, k)``, the chain rule gives::

        J_act = J_full @ C

    Build ``C`` one column per actuated joint:

    * column ``j`` has a 1 in row ``active_idx[j]``
    * a mimic ``(joint, source, multiplier, offset)`` puts ``multiplier`` in
      row ``joint`` of the column that drives ``source``

    Two things are easy to get wrong. ``offset`` is a constant and does not
    belong in a derivative at all. And ``source`` indexes the *full* vector,
    so it has to be mapped back to a column of ``C``; using it directly as a
    column index happens to work when ``active_idx`` is ``[0, 1, 2, ...]``
    and silently stops working when it is not.

    Arguments other than ``J_full`` mean what they do in ``expand_mimic``.
    ``J_full`` is ``(m, n_joints)``.

    Returns an ``(m, k)`` array.
    """
    raise NotImplementedError("TODO: write this function")
