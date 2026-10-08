"""Track 08 - Perception.

Everything a policy sees comes through a camera. These exercises cover the
pinhole model in both directions, lens distortion, and the two point-cloud
operations every registration pipeline starts with.

Camera frame convention: ``z`` points forward through the lens, ``x`` right,
``y`` down. Pixel ``u`` runs along columns, ``v`` along rows.
"""

import numpy as np


def voxel_downsample(points, size):
    """Thin a point cloud to one point per occupied voxel.

    Cut space into cubes of edge ``size`` aligned with the axes, the cube
    with index ``(0, 0, 0)`` starting at the origin::

        index = floor(p / size)

    Every occupied cube contributes one output point: the centroid of the
    points that fell into it. Not the cube's centre, and not whichever point
    came first.

    Order the output by voxel index, lexicographically on ``(ix, iy, iz)``,
    so that the result does not depend on the order of the input.

    Mind the negative side: ``floor(-0.1)`` is ``-1``, so ``-0.1`` and
    ``0.1`` are in different voxels. Truncating towards zero puts them in
    the same one.

    ``points`` is ``(N, 3)``, ``size > 0``. Returns ``(M, 3)`` with
    ``M <= N``.
    """
    # >>> solution
    points = np.asarray(points, dtype=np.float64)
    index = np.floor(points / size).astype(np.int64)
    order = np.lexsort((index[:, 2], index[:, 1], index[:, 0]))
    index, points = index[order], points[order]
    first = np.ones(len(points), dtype=bool)
    first[1:] = np.any(index[1:] != index[:-1], axis=1)
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, len(points)))
    sums = np.add.reduceat(points, starts, axis=0)
    return sums / counts[:, None]
    # <<< solution
