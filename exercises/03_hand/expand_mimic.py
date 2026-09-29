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
