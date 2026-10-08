"""Track 08 - Perception.

Everything a policy sees comes through a camera. These exercises cover the
pinhole model in both directions, lens distortion, and the two point-cloud
operations every registration pipeline starts with.

Camera frame convention: ``z`` points forward through the lens, ``x`` right,
``y`` down. Pixel ``u`` runs along columns, ``v`` along rows.
"""

import numpy as np


def project_points(points_world, T_world_cam, K):
    """Project world points into a pinhole camera.

    ``T_world_cam`` is the camera pose, the same convention as every other
    transform here: it maps camera coordinates to world coordinates. To get
    a point *into* the camera frame you need its inverse::

        p_cam = inv(T_world_cam) @ [p_world, 1]
        u = fx * x / z + cx
        v = fy * y / z + cy

    with ``K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.

    A point with ``z <= 0`` is behind the camera (or on its plane) and has
    no image. Mark it invalid and leave ``nan`` in its pixel row.

    ``points_world`` is ``(N, 3)``, ``T_world_cam`` is ``(4, 4)``, ``K`` is
    ``(3, 3)``.

    Returns ``(pixels, depth, valid)``: ``(N, 2)`` float, ``(N,)`` float
    (the camera-frame ``z``), ``(N,)`` bool.
    """
    # >>> solution
    points = np.asarray(points_world, dtype=np.float64)
    T = np.asarray(T_world_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    p_cam = (points - t) @ R  # R.T @ (p - t) for every row
    depth = p_cam[:, 2]
    valid = depth > 0.0
    pixels = np.full((points.shape[0], 2), np.nan)
    z = depth[valid]
    pixels[valid, 0] = K[0, 0] * p_cam[valid, 0] / z + K[0, 2]
    pixels[valid, 1] = K[1, 1] * p_cam[valid, 1] / z + K[1, 2]
    return pixels, depth, valid
    # <<< solution
