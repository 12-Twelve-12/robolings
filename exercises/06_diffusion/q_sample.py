"""Track 06 - Diffusion policy.

The noise schedule and the two update rules that Diffusion Policy is built
on. The network is left out on purpose: once these three functions are
right, the network is an ordinary regression problem.
"""

import numpy as np


def q_sample(x0, t, noise, alphas_cumprod):
    """Forward process: noise a clean action chunk to timestep ``t``.

    ::

        x_t = sqrt(a_t) * x0 + sqrt(1 - a_t) * noise,   a_t = alphas_cumprod[t]

    ``x0`` and ``noise`` are ``(B, ...)``. ``t`` is an integer array ``(B,)``
    with one timestep per sample, so ``a_t`` has to be reshaped to broadcast
    over the trailing dimensions.

    Returns an array with the same shape as ``x0``.
    """
    raise NotImplementedError("TODO: write this function")
