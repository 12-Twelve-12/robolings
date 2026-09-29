"""Track 06 - Diffusion policy.

The noise schedule and the two update rules that Diffusion Policy is built
on. The network is left out on purpose: once these three functions are
right, the network is an ordinary regression problem.
"""

import numpy as np


def ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    """One deterministic DDIM step from timestep ``t`` to ``t_prev``.

    ``eps_pred`` is the noise the network predicted for ``x_t``::

        x0_pred = (x_t - sqrt(1 - a_t) * eps_pred) / sqrt(a_t)
        x_prev  = sqrt(a_prev) * x0_pred + sqrt(1 - a_prev) * eps_pred

    ``t`` and ``t_prev`` are Python integers. ``t_prev < 0`` means this is
    the final step: use ``a_prev = 1``, which returns ``x0_pred`` itself.

    DDIM is what lets a policy trained with 100 steps run with 10 at
    inference time, which is the difference between 2 Hz and 20 Hz control.

    Returns an array with the same shape as ``x_t``.
    """
    raise NotImplementedError("TODO: write this function")
