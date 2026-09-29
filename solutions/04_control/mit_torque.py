"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


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
