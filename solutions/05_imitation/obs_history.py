"""Track 05 - Imitation learning plumbing.

The data handling around a policy network. ACT, Diffusion Policy and the
current vision-language-action models all share these pieces, and bugs here
produce a policy that trains fine and then fails on the robot.
"""

import numpy as np


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
