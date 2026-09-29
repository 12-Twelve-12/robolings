"""Track 05 - Imitation learning plumbing.

The data handling around a policy network. ACT, Diffusion Policy and the
current vision-language-action models all share these pieces, and bugs here
produce a policy that trains fine and then fails on the robot.
"""

import numpy as np


def minmax_normalize(x, lo, hi):
    """Map ``x`` from ``[lo, hi]`` to ``[-1, 1]``, per dimension.

    ``lo`` and ``hi`` are the per-dimension minimum and maximum over the
    training set. A dimension that never moves in the data has
    ``hi == lo``; dividing by that range gives ``nan`` and silently poisons
    training. Map such dimensions to ``0`` instead.

    ``x`` is ``(..., D)``; ``lo`` and ``hi`` are ``(D,)``.

    Returns an array with the same shape as ``x``.
    """
    raise NotImplementedError("TODO: write this function")


def minmax_unnormalize(y, lo, hi):
    """Inverse of ``minmax_normalize``: map ``[-1, 1]`` back to ``[lo, hi]``.

    For a constant dimension (``hi == lo``) return ``lo``, whatever ``y``
    is. The policy output for that dimension carries no information.

    Returns an array with the same shape as ``y``.
    """
    raise NotImplementedError("TODO: write this function")
