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


def mit_torque(kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit):
    """Joint torque of the "MIT mode" impedance controller.

    Most quasi-direct-drive actuators expose this control law::

        tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff

    Clip the result to ``[-tau_limit, tau_limit]``. The clip is applied to
    the *total* torque, not to the individual terms.

    All arguments broadcast against each other.

    Returns an array with the broadcast shape.
    """
    # >>> solution
    kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit = (
        np.asarray(v, dtype=np.float64) for v in (kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit)
    )
    tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
    return np.clip(tau, -tau_limit, tau_limit)
    # <<< solution


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
