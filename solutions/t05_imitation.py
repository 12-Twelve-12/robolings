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
    # >>> solution
    x = np.asarray(x, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    span = hi - lo
    constant = span == 0.0
    safe = np.where(constant, 1.0, span)
    y = 2.0 * (x - lo) / safe - 1.0
    return np.where(constant, 0.0, y)
    # <<< solution


def minmax_unnormalize(y, lo, hi):
    """Inverse of ``minmax_normalize``: map ``[-1, 1]`` back to ``[lo, hi]``.

    For a constant dimension (``hi == lo``) return ``lo``, whatever ``y``
    is. The policy output for that dimension carries no information.

    Returns an array with the same shape as ``y``.
    """
    # >>> solution
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    span = hi - lo
    x = (y + 1.0) / 2.0 * span + lo
    return np.where(span == 0.0, lo, x)
    # <<< solution


def make_action_chunks(actions, horizon):
    """Cut an episode into overlapping action chunks.

    Chunk ``t`` holds the ``horizon`` actions starting at step ``t``::

        chunks[t, k] = actions[t + k]

    Near the end of the episode ``t + k`` runs past the last step. Fill
    those slots by repeating the final action, and mark them in ``is_pad``
    so the loss can ignore them.

    ``actions`` is ``(T, D)``.

    Returns ``(chunks, is_pad)`` with shapes ``(T, horizon, D)`` and
    ``(T, horizon)``. ``is_pad`` is boolean and ``True`` on padded slots.
    """
    # >>> solution
    actions = np.asarray(actions, dtype=np.float64)
    T = actions.shape[0]
    idx = np.arange(T)[:, None] + np.arange(horizon)[None, :]
    is_pad = idx >= T
    chunks = actions[np.minimum(idx, T - 1)]
    return chunks, is_pad
    # <<< solution


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


def stack_obs_history(obs, n):
    """Give every step its last ``n`` observations.

    ::

        out[t] = [obs[t - n + 1], ..., obs[t - 1], obs[t]]

    At the start of the episode there is no history yet. Fill the missing
    slots by repeating the first observation, which is what the robot does
    at deployment.

    ``obs`` is ``(T, D)``.

    Returns a ``(T, n, D)`` array, oldest observation first.
    """
    # >>> solution
    obs = np.asarray(obs, dtype=np.float64)
    T = obs.shape[0]
    idx = np.arange(T)[:, None] + np.arange(-n + 1, 1)[None, :]
    return obs[np.maximum(idx, 0)]
    # <<< solution
