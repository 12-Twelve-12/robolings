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
    # >>> solution
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    tau = float(np.clip(t / duration, 0.0, 1.0))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    if t < 0.0 or t > duration:
        ds = 0.0
        dds = 0.0
    delta = q1 - q0
    return q0 + delta * s, delta * ds, delta * dds
    # <<< solution
