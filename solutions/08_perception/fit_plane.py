"""Track 08 - Perception.

Everything a policy sees comes through a camera. These exercises cover the
pinhole model in both directions, lens distortion, and the two point-cloud
operations every registration pipeline starts with.

Camera frame convention: ``z`` points forward through the lens, ``x`` right,
``y`` down. Pixel ``u`` runs along columns, ``v`` along rows.
"""

import numpy as np


def fit_plane(points):
    """Least-squares plane through a point cloud, as ``dot(n, p) + d = 0``.

    The table a policy puts things on is found this way. Centre the points
    on their mean, take the SVD of the centred ``(N, 3)`` matrix, and the
    normal is the right singular vector of the *smallest* singular value:
    the direction along which the cloud is thinnest. Then ``d = -dot(n, c)``
    for the centroid ``c``.

    The sign of ``n`` is free, so fix it: choose the sign that makes
    ``d >= 0``, which points ``n`` towards the side of the plane the origin
    is on. (If the plane passes through the origin, ``d = 0`` and either
    sign is accepted.)

    ``points`` is ``(N, 3)`` with ``N >= 3``.

    Returns ``(n, d)``: a unit ``(3,)`` array and a Python ``float``.
    """
    # >>> solution
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = vt[-1]
    d = -float(n @ centroid)
    if d < 0.0:
        n, d = -n, -d
    return n, d
    # <<< solution
