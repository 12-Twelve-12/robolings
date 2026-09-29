"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


def wrap_to_pi(a):
    """Wrap angles in radians to the interval from ``-pi`` to ``pi``.

    ``a`` is a scalar or an array, and may be many turns away from zero.

    Exactly half a turn can be returned as ``-pi`` or as ``pi``.

    Returns an array with the same shape as ``a``.
    """
    raise NotImplementedError("TODO: write this function")


def angle_diff(a, b):
    """Shortest signed rotation that takes angle ``b`` to angle ``a``.

    For a continuous joint or a heading, ``a - b`` is the wrong error: going
    from 179 degrees to -179 degrees is a move of 2 degrees, not of -358.
    A controller fed with the raw difference spins the joint the long way
    round.

    The result is ``a - b`` wrapped to the interval from ``-pi`` to ``pi``.

    Returns an array with the broadcast shape of ``a`` and ``b``.
    """
    raise NotImplementedError("TODO: write this function")
