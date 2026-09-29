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

    raise NotImplementedError("TODO: write this function")
