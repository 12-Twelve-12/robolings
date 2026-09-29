"""Track 05 - Imitation learning plumbing.

The data handling around a policy network. ACT, Diffusion Policy and the
current vision-language-action models all share these pieces, and bugs here
produce a policy that trains fine and then fails on the robot.
"""

import numpy as np


def to_delta(chunk, state, absolute_mask):
    """Express an action chunk relative to the current robot state.

    Policies usually learn better from "move 2 cm up from where you are"
    than from absolute targets. Every action in the chunk is taken relative
    to the state at the *start* of the chunk::

        delta[k] = chunk[k] - state

    It is not the difference between consecutive actions. With consecutive
    differences, a small error in one step shifts every later step.

    Some dimensions should stay absolute, a gripper opening for example.
    ``absolute_mask`` is ``True`` for those, and they are copied unchanged.

    ``chunk`` is ``(H, D)``; ``state`` and ``absolute_mask`` are ``(D,)``.

    Returns a ``(H, D)`` array.
    """
    # >>> solution
    chunk = np.asarray(chunk, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    mask = np.asarray(absolute_mask, dtype=bool)
    return np.where(mask, chunk, chunk - state)
    # <<< solution


def from_delta(delta, state, absolute_mask):
    """Inverse of ``to_delta``: turn a predicted chunk into absolute targets.

    ``state`` has to be the state the policy saw when it predicted the
    chunk, not the state at the time each action is executed.

    Returns a ``(H, D)`` array.
    """
    # >>> solution
    delta = np.asarray(delta, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    mask = np.asarray(absolute_mask, dtype=bool)
    return np.where(mask, delta, delta + state)
    # <<< solution
