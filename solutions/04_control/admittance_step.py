"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


def admittance_step(x, v, x_ref, f_ext, M, D, K, dt):
    """One step of admittance control: a measured force moves a virtual mass.

    Impedance control (``mit_torque``) turns a position error into a
    torque. Admittance control is the other direction: a measured contact
    force ``f_ext`` drives a virtual mass-spring-damper, and the position
    that comes out is what gets commanded. The spring is anchored at
    ``x_ref``::

        M * a + D * v + K * (x - x_ref) = f_ext

    Solve for ``a`` and integrate with the semi-implicit Euler step, which
    is the one that stays stable with a stiff spring::

        v_new = v + a * dt
        x_new = x + v_new * dt

    Note ``x_new`` uses the *new* velocity. The explicit variant with the
    old ``v`` gains energy every step.

    All of ``x``, ``v``, ``x_ref``, ``f_ext``, ``M``, ``D``, ``K`` are
    floats or arrays of one shape (one axis each). ``dt`` is a float.

    Returns ``(x_new, v_new)`` with the shape of ``x``.
    """
    # >>> solution
    x, v, x_ref, f_ext = (np.asarray(a, dtype=np.float64) for a in (x, v, x_ref, f_ext))
    M, D, K = (np.asarray(a, dtype=np.float64) for a in (M, D, K))
    a = (f_ext - D * v - K * (x - x_ref)) / M
    v_new = v + a * dt
    x_new = x + v_new * dt
    return x_new, v_new
    # <<< solution
