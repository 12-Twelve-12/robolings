"""Track 04 - Low-level control.

Small functions that sit between a policy and the motors. They are simple,
and getting any of them subtly wrong is a common way to damage hardware.
"""

import numpy as np


def gravity_torque(q, lengths, masses, com, g=9.81):
    """Joint torques that hold a planar arm against gravity.

    The arm lies in the vertical plane, joint ``i`` rotates about the
    out-of-plane axis, and ``q[i]`` is measured relative to the previous link.
    Gravity points along ``-y``.

    * ``lengths[i]`` is the length of link ``i``.
    * ``masses[i]`` is its mass.
    * ``com[i]`` is the distance from joint ``i`` to the centre of mass of
      link ``i``, along the link.

    Feed the result to ``mit_torque`` as ``tau_ff`` and the arm stops sagging
    between setpoints.

    The height of the centre of mass of link ``i`` is::

        y_i = sum over j < i of lengths[j] * sin(a_j)   +   com[i] * sin(a_i)

    where ``a_i`` is the absolute angle of link ``i``, the running sum of
    ``q[0..i]``. The potential energy is ``g * sum_i masses[i] * y_i``, and
    the torque that balances it is its gradient with respect to ``q``.

    Working the gradient out by hand for every joint is fiddly. There is a
    shortcut: joint ``k`` moves every link from ``k`` onwards, so

        tau[k] = g * sum over i >= k of masses[i] * d y_i / d q[k]

    and ``d y_i / d q[k]`` is a cosine sum over the same links.

    Returns a ``(n,)`` array.
    """
    # >>> solution
    q = np.asarray(q, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    com = np.asarray(com, dtype=np.float64)
    n = len(q)
    absolute = np.cumsum(q)
    # weight[j] is the mass hanging on the far end of link j, plus the part of
    # link j itself that sits beyond its own centre of mass
    tau = np.zeros(n)
    for k in range(n):
        total = 0.0
        for i in range(k, n):
            arm = com[i] * np.cos(absolute[i])
            for j in range(k, i):
                arm += lengths[j] * np.cos(absolute[j])
            total += masses[i] * arm
        tau[k] = g * total
    return tau
    # <<< solution
