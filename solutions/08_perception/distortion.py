"""Track 08 - Perception.

Everything a policy sees comes through a camera. These exercises cover the
pinhole model in both directions, lens distortion, and the two point-cloud
operations every registration pipeline starts with.

Camera frame convention: ``z`` points forward through the lens, ``x`` right,
``y`` down. Pixel ``u`` runs along columns, ``v`` along rows.
"""

import numpy as np


def distort_points(xy, coeffs):
    """Apply Brown-Conrady lens distortion to normalised image coordinates.

    ``xy`` holds points on the ``z = 1`` plane, before ``K`` is applied.
    With ``coeffs = (k1, k2, p1, p2)`` and ``r2 = x**2 + y**2``::

        radial = 1 + k1 * r2 + k2 * r2**2
        x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_d = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    The radial part moves a point along its own ray from the centre; the
    tangential part (``p1``, ``p2``) is what breaks that symmetry.

    ``xy`` is ``(N, 2)``. Returns ``(N, 2)``.
    """
    # >>> solution
    xy = np.asarray(xy, dtype=np.float64)
    k1, k2, p1, p2 = (float(c) for c in coeffs)
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([x_d, y_d], axis=1)
    # <<< solution


def undistort_points(xy_d, coeffs, iters=10):
    """Invert the distortion, which has no closed form.

    Fixed-point iteration, starting from the distorted point itself. At each
    step evaluate the radial factor and the tangential offset at the current
    estimate and solve the forward model for the undistorted point::

        x = (x_d - tangential_x(x, y)) / radial(x, y)
        y = (y_d - tangential_y(x, y)) / radial(x, y)

    Run exactly ``iters`` rounds. One round is not enough: the radial factor
    has to be evaluated at the undistorted point, which you do not have yet.

    ``xy_d`` is ``(N, 2)``. Returns ``(N, 2)``.
    """
    # >>> solution
    xy_d = np.asarray(xy_d, dtype=np.float64)
    k1, k2, p1, p2 = (float(c) for c in coeffs)
    x_d, y_d = xy_d[:, 0], xy_d[:, 1]
    x, y = x_d.copy(), y_d.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = (x_d - tx) / radial, (y_d - ty) / radial
    return np.stack([x, y], axis=1)
    # <<< solution
