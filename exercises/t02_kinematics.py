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


def transform_inverse(T):
    """Inverse of a homogeneous transform, without a general matrix inverse.

    For ``T = [[R, p], [0, 1]]`` the inverse is ``[[R.T, -R.T @ p], [0, 1]]``.
    This is cheaper and numerically better than ``np.linalg.inv``, and it is
    what runs in a control loop.

    Do not call ``np.linalg.inv`` or ``np.linalg.solve``.

    Returns a ``(4, 4)`` array.
    """
    raise NotImplementedError("TODO: write this function")


def forward_kinematics(origins, axes, q):
    """World-frame transform of every link in the chain.

    See the module docstring for the meaning of ``origins`` and ``axes``.

    Returns a ``(n, 4, 4)`` array where entry ``i`` is the frame of link
    ``i`` expressed in the world frame.
    """
    raise NotImplementedError("TODO: write this function")


def geometric_jacobian(frames, axes, tip_offset):
    """Geometric Jacobian of a point attached to the last link.

    ``frames`` is the output of ``forward_kinematics``. ``tip_offset`` is the
    position of the point of interest in the last link's frame, for example
    a fingertip.

    Column ``i`` of the result describes what joint ``i`` does to the tip::

        a_i = R_i @ axes[i]            joint axis in the world frame
        p_i = frames[i][:3, 3]         joint position in the world frame
        J[:3, i] = cross(a_i, p_tip - p_i)     linear velocity
        J[3:, i] = a_i                         angular velocity

    Returns a ``(6, n)`` array: linear rows on top, angular rows below.
    """
    raise NotImplementedError("TODO: write this function")


def dls_ik_step(J, err, damping):
    """One damped-least-squares inverse kinematics step.

    Solves for the joint update that reduces the task-space error ``err``::

        dq = J.T @ inv(J @ J.T + damping**2 * I) @ err

    The plain pseudo-inverse produces huge joint velocities near a
    singularity. The damping term trades a little accuracy for a bounded
    step, which is why this is the default on real arms and hands.

    ``J`` is ``(m, n)`` and ``err`` is ``(m,)``. Use ``np.linalg.solve``
    rather than forming the inverse.

    Returns a ``(n,)`` array.
    """
    raise NotImplementedError("TODO: write this function")
