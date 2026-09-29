"""Track 06 - Diffusion policy.

The noise schedule and the two update rules that Diffusion Policy is built
on. The network is left out on purpose: once these three functions are
right, the network is an ordinary regression problem.
"""

import numpy as np


def cfg_noise(eps_cond, eps_uncond, scale):
    """Combine a conditional and an unconditional noise prediction.

    Classifier-free guidance runs the network twice, once with the
    observation and once with it dropped, and extrapolates away from the
    unconditional prediction::

        eps = eps_uncond + scale * (eps_cond - eps_uncond)

    ``scale = 1`` reproduces the conditional prediction, ``scale = 0`` the
    unconditional one, and anything above 1 pushes further in the direction
    the condition asks for. For a policy that is the knob between an action
    that ignores what the camera sees and one that follows it too eagerly.

    One line, and the reason it is worth an exercise is that both the sign
    and the base term are easy to swap. Pin the two anchors and the wrong
    versions have nowhere to hide: an implementation built on ``eps_cond``
    as the base gives the wrong answer at ``scale = 0``, and one with the
    difference reversed gives the wrong answer at ``scale = 1``.

    ``eps_cond`` and ``eps_uncond`` have the same shape. ``scale`` is a
    Python float.

    Returns an array with that shape.
    """
    raise NotImplementedError("TODO: write this function")
