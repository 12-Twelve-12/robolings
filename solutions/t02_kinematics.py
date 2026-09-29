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
    # >>> solution
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out
    # <<< solution


def forward_kinematics(origins, axes, q):
    """World-frame transform of every link in the chain.

    See the module docstring for the meaning of ``origins`` and ``axes``.

    Returns a ``(n, 4, 4)`` array where entry ``i`` is the frame of link
    ``i`` expressed in the world frame.
    """
    # >>> solution
    def _rot(axis, angle):
        k = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    origins = np.asarray(origins, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    frames = np.zeros((n, 4, 4))
    T = np.eye(4)
    for i in range(n):
        joint = np.eye(4)
        joint[:3, :3] = _rot(axes[i], q[i])
        T = T @ origins[i] @ joint
        frames[i] = T
    return frames
    # <<< solution


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
    # >>> solution
    frames = np.asarray(frames, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    tip_offset = np.asarray(tip_offset, dtype=np.float64)
    n = frames.shape[0]
    p_tip = frames[-1][:3, :3] @ tip_offset + frames[-1][:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        a = frames[i][:3, :3] @ (axes[i] / np.linalg.norm(axes[i]))
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(a, p_tip - p)
        J[3:, i] = a
    return J
    # <<< solution


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
    # >>> solution
    J = np.asarray(J, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    m = J.shape[0]
    A = J @ J.T + (damping**2) * np.eye(m)
    return J.T @ np.linalg.solve(A, err)
    # <<< solution
