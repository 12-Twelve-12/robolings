"""Track 05 - Imitation learning plumbing.

The data handling around a policy network. ACT, Diffusion Policy and the
current vision-language-action models all share these pieces, and bugs here
produce a policy that trains fine and then fails on the robot.
"""

import numpy as np


def temporal_ensemble(preds, m):
    """Blend the overlapping predictions for the current step, as in ACT.

    With action chunking, the action for the current step was predicted
    several times: once by every recent chunk that covers it. ACT averages
    them with exponential weights::

        w[i] = exp(-m * i)
        out  = sum_i w[i] * preds[i] / sum_i w[i]

    ``preds`` is ordered oldest first, so ``i = 0`` is the *oldest*
    prediction and gets the *largest* weight. A smaller ``m`` takes new
    observations into account more slowly.

    ``preds`` is ``(K, D)``.

    Returns a ``(D,)`` array.
    """
    # >>> solution
    preds = np.asarray(preds, dtype=np.float64)
    w = np.exp(-m * np.arange(preds.shape[0]))
    w = w / w.sum()
    return (w[:, None] * preds).sum(axis=0)
    # <<< solution
