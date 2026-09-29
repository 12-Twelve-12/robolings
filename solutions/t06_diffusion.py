"""Track 06 - Diffusion policy.

The noise schedule and the two update rules that Diffusion Policy is built
on. The network is left out on purpose: once these three functions are
right, the network is an ordinary regression problem.
"""

import numpy as np


def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    """Cosine noise schedule of Nichol and Dhariwal.

    Diffusion Policy uses this schedule (``squaredcos_cap_v2``)::

        f(u)     = cos((u + s) / (1 + s) * pi / 2) ** 2
        betas[i] = min(1 - f((i + 1) / num_steps) / f(i / num_steps), max_beta)

    and then::

        alphas_cumprod = cumprod(1 - betas)

    Returns ``(betas, alphas_cumprod)``, each ``(num_steps,)``.
    """
    # >>> solution
    def f(u):
        return np.cos((u + s) / (1.0 + s) * np.pi / 2.0) ** 2

    i = np.arange(num_steps, dtype=np.float64)
    betas = 1.0 - f((i + 1.0) / num_steps) / f(i / num_steps)
    betas = np.minimum(betas, max_beta)
    alphas_cumprod = np.cumprod(1.0 - betas)
    return betas, alphas_cumprod
    # <<< solution


def q_sample(x0, t, noise, alphas_cumprod):
    """Forward process: noise a clean action chunk to timestep ``t``.

    ::

        x_t = sqrt(a_t) * x0 + sqrt(1 - a_t) * noise,   a_t = alphas_cumprod[t]

    ``x0`` and ``noise`` are ``(B, ...)``. ``t`` is an integer array ``(B,)``
    with one timestep per sample, so ``a_t`` has to be reshaped to broadcast
    over the trailing dimensions.

    Returns an array with the same shape as ``x0``.
    """
    # >>> solution
    x0 = np.asarray(x0, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    a = np.asarray(alphas_cumprod, dtype=np.float64)[np.asarray(t, dtype=int)]
    a = a.reshape((-1,) + (1,) * (x0.ndim - 1))
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise
    # <<< solution


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
    # >>> solution
    x_t = np.asarray(x_t, dtype=np.float64)
    eps_pred = np.asarray(eps_pred, dtype=np.float64)
    a_t = alphas_cumprod[t]
    a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
    x0_pred = (x_t - np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
    return np.sqrt(a_prev) * x0_pred + np.sqrt(1.0 - a_prev) * eps_pred
    # <<< solution
