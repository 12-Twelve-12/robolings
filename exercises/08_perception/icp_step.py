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
    raise NotImplementedError("TODO: write this function")
