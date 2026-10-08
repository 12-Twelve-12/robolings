"""Track 09 - State estimation.

A robot never sees its own state; it sees sensors. The gyro is smooth but
drifts, the accelerometer is noisy but does not drift, and the Kalman filter
is the general way to weigh a prediction against a measurement. These
exercises build the pieces one at a time.
"""

import numpy as np


def complementary_filter(angle, gyro, accel_angle, dt, alpha):
    """One update of a complementary filter for a tilt angle.

    The gyro gives a rate that integrates nicely over one step but drifts
    over minutes. The accelerometer gives an absolute angle that is noisy
    on every step but never drifts. Blend them::

        angle = alpha * (angle + gyro * dt) + (1 - alpha) * accel_angle

    ``alpha`` close to ``1`` trusts the gyro; ``alpha = 0`` throws it away.
    The gyro term integrates from the *previous* estimate, not from the
    accelerometer angle.

    ``angle``, ``gyro`` and ``accel_angle`` are floats, or arrays of one
    shape to filter roll and pitch at once. ``dt`` and ``alpha`` are floats,
    ``0 <= alpha <= 1``.

    Returns the new angle: a Python ``float`` for float inputs, an array of
    the same shape otherwise.
    """
    raise NotImplementedError("TODO: write this function")
