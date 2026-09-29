"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


def trapezoid(distance, v_max, a_max, t):
    """Trapezoidal velocity profile, the one most motor drives run internally.

    Accelerate at ``a_max`` up to ``v_max``, cruise, then decelerate at
    ``a_max`` and arrive at ``distance`` with zero velocity.

    ``t_ramp = v_max / a_max`` and ``d_ramp = v_max ** 2 / (2 * a_max)``. If
    ``2 * d_ramp > distance`` the move is too short to ever reach ``v_max``:
    the profile becomes a triangle, the peak speed is
    ``sqrt(a_max * distance)``, and the ramp time shrinks with it. Leaving
    that case out is what makes a naive implementation overshoot.

    Before ``t = 0`` the output is ``(0, 0)``; after the move it is
    ``(distance, 0)``.

    ``distance`` is non-negative. ``t`` is a Python float.

    Returns ``(position, velocity)`` as Python floats.
    """
    raise NotImplementedError("TODO: write this function")
