"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


def min_jerk(q0, q1, duration, t):
    """Minimum-jerk trajectory from ``q0`` to ``q1``.

    With ``tau = clip(t / duration, 0, 1)``::

        s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
        pos    = q0 + (q1 - q0) * s

    Velocity and acceleration are the first and second time derivatives of
    ``pos``. Remember the chain rule: ``d tau / dt = 1 / duration``.

    Outside ``[0, duration]`` the position holds at the nearest endpoint and
    velocity and acceleration are zero.

    Returns ``(pos, vel, acc)``, each the same shape as ``q0``.
    """
    raise NotImplementedError("TODO: write this function")
