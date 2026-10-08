"""Wrong answers for track 02, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "transform_inverse",
        "transform_inverse",
        "general matrix inverse",
        """
def transform_inverse(T):
    return np.linalg.inv(np.asarray(T, dtype=np.float64))
""",
    ),
    (
        "transform_inverse",
        "transform_inverse",
        "translation is only negated",
        """
def transform_inverse(T):
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, 3]
    return out
""",
    ),
    (
        "forward_kinematics",
        "forward_kinematics",
        "joint rotation applied before the origin",
        """
def forward_kinematics(origins, axes, q):
    def _rot(axis, angle):
        k = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    origins = np.asarray(origins, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    frames = np.zeros((len(q), 4, 4))
    T = np.eye(4)
    for i in range(len(q)):
        joint = np.eye(4)
        joint[:3, :3] = _rot(axes[i], q[i])
        T = T @ joint @ origins[i]
        frames[i] = T
    return frames
""",
    ),
    (
        "forward_kinematics",
        "forward_kinematics",
        "returns only the last frame",
        """
def forward_kinematics(origins, axes, q):
    def _rot(axis, angle):
        k = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    origins = np.asarray(origins, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    T = np.eye(4)
    for i in range(len(q)):
        joint = np.eye(4)
        joint[:3, :3] = _rot(axes[i], q[i])
        T = T @ origins[i] @ joint
    return T
""",
    ),
    (
        "geometric_jacobian",
        "geometric_jacobian",
        "tip offset ignored",
        """
def geometric_jacobian(frames, axes, tip_offset):
    frames = np.asarray(frames, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    n = frames.shape[0]
    p_tip = frames[-1][:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        a = frames[i][:3, :3] @ axes[i]
        J[:3, i] = np.cross(a, p_tip - frames[i][:3, 3])
        J[3:, i] = a
    return J
""",
    ),
    (
        "geometric_jacobian",
        "geometric_jacobian",
        "cross product reversed",
        """
def geometric_jacobian(frames, axes, tip_offset):
    frames = np.asarray(frames, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    n = frames.shape[0]
    p_tip = frames[-1][:3, :3] @ np.asarray(tip_offset, dtype=np.float64) + frames[-1][:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        a = frames[i][:3, :3] @ axes[i]
        J[:3, i] = np.cross(p_tip - frames[i][:3, 3], a)
        J[3:, i] = a
    return J
""",
    ),
    (
        "geometric_jacobian",
        "geometric_jacobian",
        "axis left in the joint frame",
        """
def geometric_jacobian(frames, axes, tip_offset):
    frames = np.asarray(frames, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    n = frames.shape[0]
    p_tip = frames[-1][:3, :3] @ np.asarray(tip_offset, dtype=np.float64) + frames[-1][:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        a = axes[i]
        J[:3, i] = np.cross(a, p_tip - frames[i][:3, 3])
        J[3:, i] = a
    return J
""",
    ),
    (
        "geometric_jacobian",
        "geometric_jacobian",
        "angular rows on top",
        """
def geometric_jacobian(frames, axes, tip_offset):
    frames = np.asarray(frames, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    n = frames.shape[0]
    p_tip = frames[-1][:3, :3] @ np.asarray(tip_offset, dtype=np.float64) + frames[-1][:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        a = frames[i][:3, :3] @ axes[i]
        J[3:, i] = np.cross(a, p_tip - frames[i][:3, 3])
        J[:3, i] = a
    return J
""",
    ),
    (
        "dls_ik_step",
        "dls_ik_step",
        "damping is not squared",
        """
def dls_ik_step(J, err, damping):
    J = np.asarray(J, dtype=np.float64)
    A = J @ J.T + damping * np.eye(J.shape[0])
    return J.T @ np.linalg.solve(A, np.asarray(err, dtype=np.float64))
""",
    ),
    (
        "dls_ik_step",
        "dls_ik_step",
        "damping ignored",
        """
def dls_ik_step(J, err, damping):
    return np.linalg.pinv(np.asarray(J, dtype=np.float64)) @ np.asarray(err, dtype=np.float64)
""",
    ),
    (
        "ik_2link",
        "ik_2link",
        "the elbow term of q1 is left out",
        """
def ik_2link(x, y, l1, l2):
    r2 = float(x) ** 2 + float(y) ** 2
    cos_q2 = (r2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    if abs(cos_q2) > 1.0 + 1e-12:
        return None
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(1.0 - cos_q2**2)
    out = np.empty((2, 2), dtype=np.float64)
    for row, sign in enumerate((1.0, -1.0)):
        q2 = np.arctan2(sign * sin_q2, cos_q2)
        out[row] = (np.arctan2(y, x), q2)
    return out
""",
    ),
    (
        "ik_2link",
        "ik_2link",
        "an unreachable target is clipped instead of rejected",
        """
def ik_2link(x, y, l1, l2):
    r2 = float(x) ** 2 + float(y) ** 2
    cos_q2 = (r2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(1.0 - cos_q2**2)
    out = np.empty((2, 2), dtype=np.float64)
    for row, sign in enumerate((1.0, -1.0)):
        q2 = np.arctan2(sign * sin_q2, cos_q2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        out[row] = (q1, q2)
    return out
""",
    ),
    (
        "ik_2link",
        "ik_2link",
        "both rows return the same branch",
        """
def ik_2link(x, y, l1, l2):
    r2 = float(x) ** 2 + float(y) ** 2
    cos_q2 = (r2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    if abs(cos_q2) > 1.0 + 1e-12:
        return None
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(1.0 - cos_q2**2)
    out = np.empty((2, 2), dtype=np.float64)
    for row in (0, 1):
        q2 = np.arctan2(sin_q2, cos_q2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        out[row] = (q1, q2)
    return out
""",
    ),
    (
        "null_space_step",
        "null_space_step",
        "the secondary objective is added without the projector",
        """
def null_space_step(J, dx, q_secondary):
    J = np.asarray(J, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    q_secondary = np.asarray(q_secondary, dtype=np.float64)
    return np.linalg.pinv(J) @ dx + q_secondary
""",
    ),
    (
        "null_space_step",
        "null_space_step",
        "the transpose stands in for the pseudo-inverse",
        """
def null_space_step(J, dx, q_secondary):
    J = np.asarray(J, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    q_secondary = np.asarray(q_secondary, dtype=np.float64)
    J_pinv = J.T
    projector = np.eye(J.shape[1]) - J_pinv @ J
    return J_pinv @ dx + projector @ q_secondary
""",
    ),
    (
        "null_space_step",
        "null_space_step",
        "the projector is applied to the task term as well",
        """
def null_space_step(J, dx, q_secondary):
    J = np.asarray(J, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    q_secondary = np.asarray(q_secondary, dtype=np.float64)
    J_pinv = np.linalg.pinv(J)
    projector = np.eye(J.shape[1]) - J_pinv @ J
    return projector @ (J_pinv @ dx + q_secondary)
""",
    ),
    (
        "manipulability",
        "manipulability",
        "uses J.T @ J, which is singular whenever there are spare joints",
        """
def manipulability(J):
    J = np.asarray(J, dtype=np.float64)
    return float(np.sqrt(max(np.linalg.det(J.T @ J), 0.0)))
""",
    ),
    (
        "manipulability",
        "manipulability",
        "falls back to det(J), which only exists for a square Jacobian",
        """
def manipulability(J):
    J = np.asarray(J, dtype=np.float64)
    if J.shape[0] != J.shape[1]:
        return 0.0
    return float(abs(np.linalg.det(J)))
""",
    ),
    (
        "manipulability",
        "manipulability",
        "the square root is left out",
        """
def manipulability(J):
    J = np.asarray(J, dtype=np.float64)
    if J.shape[0] > J.shape[1]:
        return 0.0
    return float(np.linalg.det(J @ J.T))
""",
    ),
]
