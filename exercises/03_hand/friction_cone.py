"""Track 03 - Dexterous hands.

Hands differ from arms in three ways that these exercises cover: many joints
are mechanically coupled, the task is defined by several fingertips at once,
and the joint ranges are small enough that limits are hit all the time.
"""

import numpy as np


def project_to_friction_cone(force, normal, mu):
    """Project a contact force onto the Coulomb friction cone.

    A fingertip can only push, and it can only carry tangential force up to
    ``mu`` times the normal force. Everything a contact-aware controller
    commands has to be clipped to that set first.

    Split the force at the contact into its normal and tangential parts,
    where ``n`` is ``normal`` after normalising::

        f_n = dot(force, n)
        f_t = force - f_n * n

    Three cases, and all three have to come out right:

    1. ``norm(f_t) <= mu * f_n``: already inside the cone, return it
       unchanged.
    2. ``mu * norm(f_t) <= -f_n``: the force points into the surface and no
       part of it survives, return zeros.
    3. otherwise, project onto the surface of the cone::

           s   = (mu * norm(f_t) + f_n) / (mu**2 + 1)
           out = s * n + mu * s * f_t / norm(f_t)

    Case 3 is where the wrong answers live. Rescaling the whole vector until
    it fits also shrinks the normal component, which is not a projection and
    quietly weakens the grasp.

    ``force`` and ``normal`` are ``(3,)``; ``normal`` need not be unit
    length. ``mu >= 0``.

    Returns a ``(3,)`` array.
    """
    raise NotImplementedError("TODO: write this function")
