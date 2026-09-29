"""Track 03 - Dexterous hands.

Hands differ from arms in three ways that these exercises cover: many joints
are mechanically coupled, the task is defined by several fingertips at once,
and the joint ranges are small enough that limits are hit all the time.
"""

import numpy as np


def is_force_closure(contacts, normals, mu):
    """Does a planar two-finger grasp resist an arbitrary wrench?

    Two point contacts with friction hold an object if, and only if, the
    line joining them lies inside both friction cones. Then the fingers can
    squeeze along that line, and the friction available at each end covers
    every direction the object could otherwise slip.

    With ``d`` the unit vector from contact 0 to contact 1, the grasp needs
    the force at contact 0 to point along ``+d`` and the one at contact 1
    along ``-d``, each within its own cone of half-angle ``arctan(mu)``.

    Compare the components rather than the angles. Splitting a direction
    ``u`` against a unit normal ``n`` the same way ``friction_cone`` does::

        along  = dot(u, n)
        across = norm(u - along * n)

    the direction is inside the cone when ``along > 0`` and
    ``across <= mu * along``. That is the same test as
    ``angle(u, n) <= arctan(mu)``, but ``arccos`` near 1 turns a rounding
    error of 1e-16 into 1e-8 and reports a perfectly good diagonal grasp as
    slipping.

    Checking only that the normals oppose each other is not enough. Every
    antipodal pair passes that test, and a pair whose normals are tilted
    away from the line slips as soon as ``mu`` is small.

    * ``contacts``: ``(2, 2)``, the two contact points in the plane.
    * ``normals``: ``(2, 2)``, the inward surface normal at each contact,
      not necessarily unit length.
    * ``mu >= 0``, the friction coefficient.

    Returns a Python ``bool``.
    """
    raise NotImplementedError("TODO: write this function")
