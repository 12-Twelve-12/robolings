"""Wrong answers for track 01, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "rodrigues",
        "rodrigues",
        "axis is not normalised",
        """
def rodrigues(axis, angle):
    k = np.asarray(axis, dtype=np.float64)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
""",
    ),
    (
        "rodrigues",
        "rodrigues",
        "rotates the wrong way",
        """
def rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) - np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
""",
    ),
    (
        "quat_mul",
        "quat_mul",
        "operands swapped",
        """
def quat_mul(q1, q2):
    w1, x1, y1, z1 = np.asarray(q2, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q1, dtype=np.float64)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
""",
    ),
    (
        "quat_mul",
        "quat_mul",
        "result is normalised",
        """
def quat_mul(q1, q2):
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    q = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return q / np.linalg.norm(q)
""",
    ),
    (
        "quat_to_matrix",
        "quat_to_matrix",
        "transposed",
        """
def quat_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]).T
""",
    ),
    (
        "quat_to_matrix",
        "quat_to_matrix",
        "input is not normalised",
        """
def quat_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
""",
    ),
    (
        "matrix_to_quat",
        "matrix_to_quat",
        "divides by w",
        """
def matrix_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    w = np.sqrt(max(1.0 + R[0, 0] + R[1, 1] + R[2, 2], 0.0)) / 2.0
    with np.errstate(all="ignore"):
        return np.array([
            w,
            (R[2, 1] - R[1, 2]) / (4 * w),
            (R[0, 2] - R[2, 0]) / (4 * w),
            (R[1, 0] - R[0, 1]) / (4 * w),
        ])
""",
    ),
    (
        "matrix_to_quat",
        "matrix_to_quat",
        "w may be negative",
        """
def matrix_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(1.0 + trace)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q
""",
    ),
    (
        "slerp",
        "slerp",
        "takes the long way round",
        """
def slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if abs(dot) > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    q = (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)
""",
    ),
    (
        "slerp",
        "slerp",
        "no fallback for nearly equal inputs",
        """
def slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    with np.errstate(all="ignore"):
        return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
""",
    ),
    (
        "slerp",
        "slerp",
        "plain linear interpolation",
        """
def slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = q0 + t * (q1 - q0)
    return q / np.linalg.norm(q)
""",
    ),
    (
        "kabsch",
        "kabsch",
        "no check for a reflection",
        """
def kabsch(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - p_mean).T @ (Q - q_mean))
    R = Vt.T @ U.T
    return R, q_mean - R @ p_mean
""",
    ),
    (
        "kabsch",
        "kabsch",
        "rotation from Q to P",
        """
def kabsch(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((Q - q_mean).T @ (P - p_mean))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, q_mean - R @ p_mean
""",
    ),
    (
        "kabsch",
        "kabsch",
        "translation ignores the rotation",
        """
def kabsch(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - p_mean).T @ (Q - q_mean))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, q_mean - p_mean
""",
    ),
    (
        "kabsch",
        "kabsch",
        "centroids are not removed",
        """
def kabsch(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Q.mean(axis=0) - R @ P.mean(axis=0)
""",
    ),
    (
        "quat_log_exp",
        "quat_log",
        "no short-way flip",
        """
def quat_log(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    v = q[1:]
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return 2.0 * v
    return 2.0 * np.arctan2(norm, q[0]) * v / norm
""",
    ),
    (
        "quat_log_exp",
        "quat_log",
        "no small-angle guard",
        """
def quat_log(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:]
    norm = np.linalg.norm(v)
    with np.errstate(all="ignore"):
        return 2.0 * np.arctan2(norm, q[0]) * v / norm
""",
    ),
    (
        "quat_log_exp",
        "quat_log",
        "half the angle",
        """
def quat_log(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:]
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return np.arctan2(norm, q[0]) * v / norm
""",
    ),
    (
        "quat_log_exp",
        "quat_exp",
        "forgets to halve the angle",
        """
def quat_exp(v):
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        q = np.concatenate(([1.0], 0.5 * v))
        return q / np.linalg.norm(q)
    q = np.concatenate(([np.cos(theta)], np.sin(theta) * v / theta))
    return q / np.linalg.norm(q)
""",
    ),
    (
        "quat_log_exp",
        "quat_exp",
        "no small-angle guard",
        """
def quat_exp(v):
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v)
    with np.errstate(all="ignore"):
        return np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * v / theta))
""",
    ),
]
