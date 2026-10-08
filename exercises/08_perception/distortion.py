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
    raise NotImplementedError("TODO: write this function")


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
    raise NotImplementedError("TODO: write this function")
