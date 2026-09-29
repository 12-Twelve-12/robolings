"""Track 02 - Kinematics of a serial chain.

A chain of ``n`` revolute joints is described by two arrays:

* ``origins``: ``(n, 4, 4)``. ``origins[i]`` is the fixed transform from the
  frame of link ``i - 1`` to the frame of joint ``i`` at zero angle. For
  ``i = 0`` the parent is the world frame. This is what a URDF ``<origin>``
  tag stores.
* ``axes``: ``(n, 3)``. ``axes[i]`` is the unit rotation axis of joint ``i``,
  expressed in the joint's own frame. This is the URDF ``<axis>`` tag.

The frame of link ``i`` in the world is therefore::

    T[i] = T[i - 1] @ origins[i] @ Rot(axes[i], q[i])
"""

import numpy as np


def forward_kinematics(origins, axes, q):
    """World-frame transform of every link in the chain.

    See the module docstring for the meaning of ``origins`` and ``axes``.

    Returns a ``(n, 4, 4)`` array where entry ``i`` is the frame of link
    ``i`` expressed in the world frame.
    """

    raise NotImplementedError("TODO: write this function")
