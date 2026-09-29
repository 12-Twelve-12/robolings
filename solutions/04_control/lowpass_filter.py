"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


def lowpass_filter(x, cutoff_hz, dt):
    """First-order low-pass filter over a sequence of samples.

    ::

        rc    = 1 / (2 * pi * cutoff_hz)
        alpha = dt / (rc + dt)
        y[0]  = x[0]
        y[k]  = y[k - 1] + alpha * (x[k] - y[k - 1])

    Initialising with ``y[0] = x[0]`` rather than zero avoids a start-up
    transient, which on a robot would be a jump in the command.

    ``x`` is ``(T,)`` or ``(T, D)``; filter each dimension independently.

    Returns an array with the same shape as ``x``.
    """
    # >>> solution
    x = np.asarray(x, dtype=np.float64)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    y = np.zeros_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y
    # <<< solution
