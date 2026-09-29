"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""


def pid_step(state, error, dt, kp, ki, kd, u_limit):
    """One step of a PID controller with a saturated output.

    ``state`` is ``(integral, prev_error)``. On the first call ``prev_error``
    is ``None``.

    ::

        integral_new = integral + error * dt
        derivative   = (error - prev_error) / dt      0 on the first call
        u_raw        = kp * error + ki * integral_new + kd * derivative
        u            = clip(u_raw, -u_limit, u_limit)

    Anti-windup: if the output is saturated and the error pushes it further
    into saturation (``u_raw`` and ``error`` have the same sign), throw
    ``integral_new`` away and keep the old ``integral``.

    Without this, the integral keeps growing for as long as the actuator is
    at its limit. When the error finally changes sign, the controller stays
    saturated until all of that has been integrated away again, and the
    joint overshoots badly.

    On the first call there is no previous error. Using zero instead makes
    the derivative term jump, so the derivative is defined as zero.

    All values are Python floats.

    Returns ``(u, (integral, error))``, the output and the state for the
    next call.
    """
    raise NotImplementedError("TODO: write this function")
