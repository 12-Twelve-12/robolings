"""Track 07 - Flow matching.

The action head of recent vision-language-action models. Compared with
diffusion there is no noise schedule: the path from noise to data is a
straight line and the network learns the velocity along it.

Convention in this track: ``t = 0`` is pure noise and ``t = 1`` is data.
Some papers use the opposite direction. Check before porting code.
"""

import numpy as np


def flow_matching_target(noise, data, t):
    """Training pair for conditional flow matching.

    ::

        x_t = (1 - t) * noise + t * data
        v   = data - noise

    The network sees ``x_t`` and ``t`` and is trained to output ``v``.

    ``noise`` and ``data`` are ``(B, ...)``. ``t`` is ``(B,)`` with values in
    ``[0, 1]``, one per sample, so it has to be reshaped to broadcast over
    the trailing dimensions.

    Returns ``(x_t, v)``, each with the same shape as ``data``.
    """
    # >>> solution
    noise = np.asarray(noise, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape((-1,) + (1,) * (data.ndim - 1))
    x_t = (1.0 - t) * noise + t * data
    return x_t, data - noise
    # <<< solution
