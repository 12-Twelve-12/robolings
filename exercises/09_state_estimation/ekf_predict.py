"""Track 09 - State estimation.

A robot never sees its own state; it sees sensors. The gyro is smooth but
drifts, the accelerometer is noisy but does not drift, and the Kalman filter
is the general way to weigh a prediction against a measurement. These
exercises build the pieces one at a time.
"""

import numpy as np


def ekf_predict(x, P, f, F, Q):
    """The prediction half of an extended Kalman filter.

    Push the state through the motion model and the covariance through its
    Jacobian::

        x_pred = f(x)
        P_pred = F @ P @ F.T + Q

    ``f`` is a Python callable, ``f(x)`` returns the next state ``(n,)``.
    ``F`` is its ``(n, n)`` Jacobian at ``x``, already evaluated. ``Q`` is
    the ``(n, n)`` process noise.

    The covariance grows here and only shrinks in the update. Do not modify
    ``P`` in place: the caller may still hold it.

    Returns ``(x_pred, P_pred)``.
    """
    raise NotImplementedError("TODO: write this function")
