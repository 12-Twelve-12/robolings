"""Track 05 - Imitation learning plumbing.

The data handling around a policy network. ACT, Diffusion Policy and the
current vision-language-action models all share these pieces, and bugs here
produce a policy that trains fine and then fails on the robot.
"""

import numpy as np


def ema_decay(step, max_decay=0.9999, warmup=10.0):
    """Decay rate of the weight average at a given step.

    Diffusion Policy evaluates an exponential moving average of the weights
    rather than the weights themselves. A constant decay close to 1 keeps the
    average pinned near its random initial value for thousands of steps, so
    the decay is ramped up instead::

        decay = min(max_decay, (1 + step) / (warmup + step))

    ``step`` counts from 0.

    Returns a Python float.
    """
    raise NotImplementedError("TODO: write this function")


def ema_update(average, weights, step, max_decay=0.9999, warmup=10.0):
    """One step of the average, over a dict of arrays.

    ::

        average[k] = decay * average[k] + (1 - decay) * weights[k]

    Return a new dict of new arrays. Updating in place would change arrays
    the caller still holds, and the training loop holds exactly those: the
    live weights it is about to take another gradient step on.

    ``average`` and ``weights`` have the same keys and matching shapes.

    Returns a new dict.
    """
    raise NotImplementedError("TODO: write this function")
