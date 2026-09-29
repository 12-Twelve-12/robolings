"""Track 07 - Flow matching.

The action head of recent vision-language-action models. Compared with
diffusion there is no noise schedule: the path from noise to data is a
straight line and the network learns the velocity along it.

Convention in this track: ``t = 0`` is pure noise and ``t = 1`` is data.
Some papers use the opposite direction. Check before porting code.
"""

import numpy as np


def euler_sample(v_fn, x, num_steps):
    """Generate an action chunk by integrating the learned velocity field.

    Start from noise ``x`` at ``t = 0`` and take ``num_steps`` Euler steps
    of size ``dt = 1 / num_steps`` up to ``t = 1``::

        x = x + dt * v_fn(x, t)

    Evaluate the velocity at the *start* of each step, so ``t`` takes the
    values ``0, dt, 2 dt, ..., 1 - dt``. It never reaches ``1``.

    ``v_fn(x, t)`` takes the current sample and a Python ``float`` and
    returns an array shaped like ``x``.

    Returns an array with the same shape as ``x``.
    """
    # >>> solution
    x = np.asarray(x, dtype=np.float64).copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        x = x + dt * v_fn(x, i * dt)
    return x
    # <<< solution
