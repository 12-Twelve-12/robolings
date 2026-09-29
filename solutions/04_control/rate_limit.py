"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


def rate_limit(target, prev, max_rate, dt):
    """Limit how fast a command may change.

    A policy can output a target far from the current command. Sending it
    straight to a position-controlled joint produces a violent move. Limit
    the change per control step to ``max_rate * dt``, per joint::

        out = prev + clip(target - prev, -max_rate * dt, max_rate * dt)

    ``max_rate`` is a scalar or a per-joint array.

    Returns an array with the same shape as ``target``.
    """
    # >>> solution
    target = np.asarray(target, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    step = np.asarray(max_rate, dtype=np.float64) * dt
    return prev + np.clip(target - prev, -step, step)
    # <<< solution
