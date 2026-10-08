"""Track 08 - Perception.

Everything a policy sees comes through a camera. These exercises cover the
pinhole model in both directions, lens distortion, and the two point-cloud
operations every registration pipeline starts with.

Camera frame convention: ``z`` points forward through the lens, ``x`` right,
``y`` down. Pixel ``u`` runs along columns, ``v`` along rows.
"""

import numpy as np


def backproject_depth(depth, K):
    """Turn a depth image into a point cloud in the camera frame.

    ``depth[v, u]`` is the camera-frame ``z`` of the surface seen at pixel
    ``(u, v)``, not the distance along the ray. Undo the pinhole model::

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

    Pixel coordinates are the array indices: ``u`` is the column, ``v`` the
    row, no half-pixel offset.

    A pixel with no measurement has depth ``0`` or ``nan``. It gets no
    point: mark it invalid and leave ``nan`` in its row.

    ``depth`` is ``(H, W)``, ``K`` is ``(3, 3)``.

    Returns ``(points, valid)``: ``(H, W, 3)`` float and ``(H, W)`` bool.
    """
    # >>> solution
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    points = np.full((h, w, 3), np.nan)
    z = depth[valid]
    points[valid, 0] = (u[valid] - K[0, 2]) * z / K[0, 0]
    points[valid, 1] = (v[valid] - K[1, 2]) * z / K[1, 1]
    points[valid, 2] = z
    return points, valid
    # <<< solution
