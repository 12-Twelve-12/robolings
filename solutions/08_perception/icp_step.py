"""Track 08 - Perception.

Everything a policy sees comes through a camera. These exercises cover the
pinhole model in both directions, lens distortion, and the two point-cloud
operations every registration pipeline starts with.

Camera frame convention: ``z`` points forward through the lens, ``x`` right,
``y`` down. Pixel ``u`` runs along columns, ``v`` along rows.
"""

import numpy as np


def icp_step(source, target, T):
    """One iteration of point-to-point ICP.

    ``T`` is the current guess of the rigid transform that takes ``source``
    points into the frame of ``target``. One step:

    1. Move the source points with ``T``.
    2. Pair every moved source point with its nearest target point
       (brute force is fine here).
    3. Solve the best rigid transform ``dT`` from the moved source points to
       their partners: centre both sets, SVD of the cross-covariance, and
       flip the last axis if the result is a reflection (Kabsch).
    4. The new estimate is ``dT @ T``. The correction was found in the
       target frame, so it goes on the left.

    Also report how good the pairing is: the root-mean-square distance
    between the same pairs after the update is applied.

    ``source`` is ``(N, 3)``, ``target`` is ``(M, 3)``, ``T`` is ``(4, 4)``.
    ``N`` and ``M`` may differ.

    Returns ``(T_new, rms)``: ``(4, 4)`` and a Python ``float``.
    """
    # >>> solution
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    moved = source @ T[:3, :3].T + T[:3, 3]
    d2 = ((moved[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
    partner = target[np.argmin(d2, axis=1)]

    p_mean, q_mean = moved.mean(axis=0), partner.mean(axis=0)
    U, _, Vt = np.linalg.svd((moved - p_mean).T @ (partner - q_mean))
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    dT = np.eye(4)
    dT[:3, :3] = R
    dT[:3, 3] = q_mean - R @ p_mean

    after = moved @ R.T + dT[:3, 3]
    rms = float(np.sqrt(np.mean(np.sum((after - partner) ** 2, axis=1))))
    return dT @ T, rms
    # <<< solution
