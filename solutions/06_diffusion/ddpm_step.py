"""Track 06 - Diffusion policy.

The noise schedule and the two update rules that Diffusion Policy is built
on. The network is left out on purpose: once these three functions are
right, the network is an ordinary regression problem.
"""

import numpy as np


def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    """One stochastic DDPM step from timestep ``t`` to ``t - 1``.

    ``eps_pred`` is the noise the network predicted for ``x_t``. With
    ``a = alphas_cumprod``::

        mean = (x_t - betas[t] / sqrt(1 - a[t]) * eps_pred) / sqrt(1 - betas[t])
        var  = betas[t] * (1 - a[t - 1]) / (1 - a[t])
        out  = mean + sqrt(var) * noise

    ``var`` is the variance of the true posterior, not ``betas[t]``.

    At ``t = 0`` there is no noise left to add. Return ``mean``, and do not
    read ``a[-1]``: in Python that is the last element, not an error.

    ``t`` is a Python integer. ``noise`` has the shape of ``x_t`` and is
    passed in so that the function is deterministic.

    Returns an array with the same shape as ``x_t``.
    """
    # >>> solution
    x_t = np.asarray(x_t, dtype=np.float64)
    eps_pred = np.asarray(eps_pred, dtype=np.float64)
    beta = betas[t]
    a_t = alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(1.0 - beta)
    if t == 0:
        return mean
    var = beta * (1.0 - alphas_cumprod[t - 1]) / (1.0 - a_t)
    return mean + np.sqrt(var) * np.asarray(noise, dtype=np.float64)
    # <<< solution
