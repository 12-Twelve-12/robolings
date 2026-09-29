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
    raise NotImplementedError("TODO: write this function")


def mit_torque(kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit):
    """Joint torque of the "MIT mode" impedance controller.

    Most quasi-direct-drive actuators expose this control law::

        tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff

    Clip the result to ``[-tau_limit, tau_limit]``. The clip is applied to
    the *total* torque, not to the individual terms.

    All arguments broadcast against each other.

    Returns an array with the broadcast shape.
    """
    raise NotImplementedError("TODO: write this function")


def rate_limit(target, prev, max_rate, dt):
    """Limit how fast a command may change.

    A policy can output a target far from the current command. Sending it
    straight to a position-controlled joint produces a violent move. Limit
    the change per control step to ``max_rate * dt``, per joint::

        out = prev + clip(target - prev, -max_rate * dt, max_rate * dt)

    ``max_rate`` is a scalar or a per-joint array.

    Returns an array with the same shape as ``target``.
    """
    raise NotImplementedError("TODO: write this function")
