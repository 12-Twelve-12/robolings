"""Wrong answers for track 08, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "project_points",
        "project_points",
        "the pose is applied instead of inverted",
        """
def project_points(points_world, T_world_cam, K):
    points = np.asarray(points_world, dtype=np.float64)
    T = np.asarray(T_world_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    p_cam = points @ T[:3, :3].T + T[:3, 3]
    depth = p_cam[:, 2]
    valid = depth > 0.0
    pixels = np.full((points.shape[0], 2), np.nan)
    z = depth[valid]
    pixels[valid, 0] = K[0, 0] * p_cam[valid, 0] / z + K[0, 2]
    pixels[valid, 1] = K[1, 1] * p_cam[valid, 1] / z + K[1, 2]
    return pixels, depth, valid
""",
    ),
    (
        "project_points",
        "project_points",
        "points behind the camera are projected anyway",
        """
def project_points(points_world, T_world_cam, K):
    points = np.asarray(points_world, dtype=np.float64)
    T = np.asarray(T_world_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    p_cam = (points - T[:3, 3]) @ T[:3, :3]
    depth = p_cam[:, 2]
    with np.errstate(all="ignore"):
        pixels = np.stack([K[0, 0] * p_cam[:, 0] / depth + K[0, 2], K[1, 1] * p_cam[:, 1] / depth + K[1, 2]], axis=1)
    return pixels, depth, np.ones(points.shape[0], dtype=bool)
""",
    ),
    (
        "project_points",
        "project_points",
        "principal point forgotten",
        """
def project_points(points_world, T_world_cam, K):
    points = np.asarray(points_world, dtype=np.float64)
    T = np.asarray(T_world_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    p_cam = (points - T[:3, 3]) @ T[:3, :3]
    depth = p_cam[:, 2]
    valid = depth > 0.0
    pixels = np.full((points.shape[0], 2), np.nan)
    z = depth[valid]
    pixels[valid, 0] = K[0, 0] * p_cam[valid, 0] / z
    pixels[valid, 1] = K[1, 1] * p_cam[valid, 1] / z
    return pixels, depth, valid
""",
    ),
    (
        "backproject_depth",
        "backproject_depth",
        "rows and columns swapped",
        """
def backproject_depth(depth, K):
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    h, w = depth.shape
    u, v = np.mgrid[0:h, 0:w]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    points = np.full((h, w, 3), np.nan)
    z = depth[valid]
    points[valid, 0] = (u[valid] - K[0, 2]) * z / K[0, 0]
    points[valid, 1] = (v[valid] - K[1, 2]) * z / K[1, 1]
    points[valid, 2] = z
    return points, valid
""",
    ),
    (
        "backproject_depth",
        "backproject_depth",
        "a zero depth still gets a point",
        """
def backproject_depth(depth, K):
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = np.isfinite(depth)
    points = np.full((h, w, 3), np.nan)
    z = depth[valid]
    points[valid, 0] = (u[valid] - K[0, 2]) * z / K[0, 0]
    points[valid, 1] = (v[valid] - K[1, 2]) * z / K[1, 1]
    points[valid, 2] = z
    return points, valid
""",
    ),
    (
        "backproject_depth",
        "backproject_depth",
        "divides by z instead of multiplying",
        """
def backproject_depth(depth, K):
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    points = np.full((h, w, 3), np.nan)
    z = depth[valid]
    points[valid, 0] = (u[valid] - K[0, 2]) / (z * K[0, 0])
    points[valid, 1] = (v[valid] - K[1, 2]) / (z * K[1, 1])
    points[valid, 2] = z
    return points, valid
""",
    ),
    (
        "distortion",
        "distort_points",
        "r instead of r squared in the radial factor",
        """
def distort_points(xy, coeffs):
    xy = np.asarray(xy, dtype=np.float64)
    k1, k2, p1, p2 = (float(c) for c in coeffs)
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    r = np.sqrt(r2)
    radial = 1.0 + k1 * r + k2 * r * r
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([x_d, y_d], axis=1)
""",
    ),
    (
        "distortion",
        "distort_points",
        "p1 and p2 swapped",
        """
def distort_points(xy, coeffs):
    xy = np.asarray(xy, dtype=np.float64)
    k1, k2, p2, p1 = (float(c) for c in coeffs)
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([x_d, y_d], axis=1)
""",
    ),
    (
        "distortion",
        "undistort_points",
        "a single division by the radial factor at the distorted point",
        """
def undistort_points(xy_d, coeffs, iters=10):
    xy_d = np.asarray(xy_d, dtype=np.float64)
    k1, k2, p1, p2 = (float(c) for c in coeffs)
    x, y = xy_d[:, 0], xy_d[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([(x - tx) / radial, (y - ty) / radial], axis=1)
""",
    ),
    (
        "distortion",
        "undistort_points",
        "the tangential part is ignored",
        """
def undistort_points(xy_d, coeffs, iters=10):
    xy_d = np.asarray(xy_d, dtype=np.float64)
    k1, k2, p1, p2 = (float(c) for c in coeffs)
    x_d, y_d = xy_d[:, 0], xy_d[:, 1]
    x, y = x_d.copy(), y_d.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x, y = x_d / radial, y_d / radial
    return np.stack([x, y], axis=1)
""",
    ),
    (
        "voxel_downsample",
        "voxel_downsample",
        "truncates towards zero instead of flooring",
        """
def voxel_downsample(points, size):
    points = np.asarray(points, dtype=np.float64)
    index = (points / size).astype(np.int64)
    order = np.lexsort((index[:, 2], index[:, 1], index[:, 0]))
    index, points = index[order], points[order]
    first = np.ones(len(points), dtype=bool)
    first[1:] = np.any(index[1:] != index[:-1], axis=1)
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, len(points)))
    return np.add.reduceat(points, starts, axis=0) / counts[:, None]
""",
    ),
    (
        "voxel_downsample",
        "voxel_downsample",
        "returns the voxel centres",
        """
def voxel_downsample(points, size):
    points = np.asarray(points, dtype=np.float64)
    index = np.unique(np.floor(points / size).astype(np.int64), axis=0)
    return (index + 0.5) * size
""",
    ),
    (
        "voxel_downsample",
        "voxel_downsample",
        "keeps the first point of each voxel",
        """
def voxel_downsample(points, size):
    points = np.asarray(points, dtype=np.float64)
    index = np.floor(points / size).astype(np.int64)
    _, first = np.unique(index, axis=0, return_index=True)
    return points[np.sort(first)]
""",
    ),
    (
        "fit_plane",
        "fit_plane",
        "points are not centred",
        """
def fit_plane(points):
    points = np.asarray(points, dtype=np.float64)
    _, _, vt = np.linalg.svd(points, full_matrices=False)
    n = vt[-1]
    d = -float(n @ points.mean(axis=0))
    if d < 0.0:
        n, d = -n, -d
    return n, d
""",
    ),
    (
        "fit_plane",
        "fit_plane",
        "takes the largest singular vector",
        """
def fit_plane(points):
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = vt[0]
    d = -float(n @ centroid)
    if d < 0.0:
        n, d = -n, -d
    return n, d
""",
    ),
    (
        "fit_plane",
        "fit_plane",
        "sign of d flipped",
        """
def fit_plane(points):
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = vt[-1]
    d = float(n @ centroid)
    if d < 0.0:
        n, d = -n, -d
    return n, d
""",
    ),
    (
        "icp_step",
        "icp_step",
        "the correction is composed on the right",
        """
def icp_step(source, target, T):
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
    return T @ dT, rms
""",
    ),
    (
        "icp_step",
        "icp_step",
        "rms measured before the update",
        """
def icp_step(source, target, T):
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
    rms = float(np.sqrt(np.mean(np.sum((moved - partner) ** 2, axis=1))))
    return dT @ T, rms
""",
    ),
    (
        "icp_step",
        "icp_step",
        "every target point is matched to a source point instead",
        """
def icp_step(source, target, T):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    moved = source @ T[:3, :3].T + T[:3, 3]
    d2 = ((target[:, None, :] - moved[None, :, :]) ** 2).sum(axis=2)
    src = moved[np.argmin(d2, axis=1)]
    p_mean, q_mean = src.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - p_mean).T @ (target - q_mean))
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    dT = np.eye(4)
    dT[:3, :3] = R
    dT[:3, 3] = q_mean - R @ p_mean
    after = src @ R.T + dT[:3, 3]
    rms = float(np.sqrt(np.mean(np.sum((after - target) ** 2, axis=1))))
    return dT @ T, rms
""",
    ),
    (
        "icp_step",
        "icp_step",
        "returns the correction alone, not dT @ T",
        """
def icp_step(source, target, T):
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
    return dT, rms
""",
    ),
]
