"""Track 09 - State estimation.

A robot never sees its own state; it sees sensors. The gyro is smooth but
drifts, the accelerometer is noisy but does not drift, and the Kalman filter
is the general way to weigh a prediction against a measurement. These
exercises build the pieces one at a time.
"""

import numpy as np


def gyro_integrate(q, omega, dt):
    """Propagate an attitude quaternion with a body-frame angular rate.

    ``q`` is ``[w, x, y, z]`` and maps body coordinates to world
    coordinates, as everywhere in this repo. ``omega`` is the rate the gyro
    reports, in the body frame, in rad/s. Over one step the body turns by
    the rotation vector ``omega * dt``, which as a quaternion is::

        dq = [cos(theta / 2), sin(theta / 2) * omega / norm(omega)]
        theta = norm(omega) * dt

    A body-frame rotation composes on the *right*::

        q_new = q * dq

    and the result is normalised so rounding does not accumulate. When
    ``norm(omega)`` is tiny, ``dq`` is ``[1, omega * dt / 2]`` (then
    normalised); do not divide by zero.

    ``q`` is ``(4,)``, ``omega`` is ``(3,)``, ``dt`` is a float.

    Returns a unit ``(4,)`` array.
    """
    # >>> solution
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega)) * dt
    if theta < 1e-8:
        dq = np.concatenate(([1.0], 0.5 * omega * dt))
    else:
        axis = omega / np.linalg.norm(omega)
        dq = np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * axis))
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = dq
    out = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    return out / np.linalg.norm(out)
    # <<< solution
