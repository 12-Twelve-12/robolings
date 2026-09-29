"""Track 05 - Imitation learning plumbing.

The data handling around a policy network. ACT, Diffusion Policy and the
current vision-language-action models all share these pieces, and bugs here
produce a policy that trains fine and then fails on the robot.
"""

import numpy as np


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
