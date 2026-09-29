"""Track 07 - Flow matching.

The action head of recent vision-language-action models. Compared with
diffusion there is no noise schedule: the path from noise to data is a
straight line and the network learns the velocity along it.

Convention in this track: ``t = 0`` is pure noise and ``t = 1`` is data.
Some papers use the opposite direction. Check before porting code.
"""

import numpy as np


def flow_matching_target(noise, data, t):
    """Training pair for conditional flow matching.

    ::

        x_t = (1 - t) * noise + t * data
        v   = data - noise

    The network sees ``x_t`` and ``t`` and is trained to output ``v``.

    ``noise`` and ``data`` are ``(B, ...)``. ``t`` is ``(B,)`` with values in
    ``[0, 1]``, one per sample, so it has to be reshaped to broadcast over
    the trailing dimensions.

    Returns ``(x_t, v)``, each with the same shape as ``data``.
    """
    raise NotImplementedError("TODO: write this function")


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
    raise NotImplementedError("TODO: write this function")
