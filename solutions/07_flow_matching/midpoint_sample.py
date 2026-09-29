"""Track 07 - Flow matching.

The action head of recent vision-language-action models. Compared with
diffusion there is no noise schedule: the path from noise to data is a
straight line and the network learns the velocity along it.

Convention in this track: ``t = 0`` is pure noise and ``t = 1`` is data.
Some papers use the opposite direction. Check before porting code.
"""

import numpy as np


def midpoint_sample(v_fn, x, num_steps):
    """Integrate the velocity field with the midpoint method.

    Euler uses the velocity at the start of the step for the whole step.
    The midpoint method first takes half a step, then uses the velocity
    found there::

        x_mid = x + dt / 2 * v_fn(x, t)
        x     = x + dt * v_fn(x_mid, t + dt / 2)

    with ``dt = 1 / num_steps`` and ``t = 0, dt, ..., 1 - dt``.

    It calls the network twice per step, and its error shrinks with
    ``dt**2`` instead of ``dt``. For the same number of network calls it is
    usually the better choice when the learned path is curved.

    ``v_fn(x, t)`` takes the current sample and a Python ``float`` and
    returns an array shaped like ``x``.

    Returns an array with the same shape as ``x``.
    """
    # >>> solution
    x = np.asarray(x, dtype=np.float64).copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i * dt
        x_mid = x + 0.5 * dt * v_fn(x, t)
        x = x + dt * v_fn(x_mid, t + 0.5 * dt)
    return x
    # <<< solution
