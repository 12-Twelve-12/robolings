"""Wrong answers for track 09, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "complementary_filter",
        "complementary_filter",
        "alpha scales only the gyro term",
        """
def complementary_filter(angle, gyro, accel_angle, dt, alpha):
    angle = np.asarray(angle, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    accel_angle = np.asarray(accel_angle, dtype=np.float64)
    out = angle + alpha * gyro * dt + (1.0 - alpha) * accel_angle
    return float(out) if out.ndim == 0 else out
""",
    ),
    (
        "complementary_filter",
        "complementary_filter",
        "integrates from the accelerometer angle",
        """
def complementary_filter(angle, gyro, accel_angle, dt, alpha):
    gyro = np.asarray(gyro, dtype=np.float64)
    accel_angle = np.asarray(accel_angle, dtype=np.float64)
    out = alpha * (accel_angle + gyro * dt) + (1.0 - alpha) * accel_angle
    return float(out) if out.ndim == 0 else out
""",
    ),
    (
        "complementary_filter",
        "complementary_filter",
        "weights swapped",
        """
def complementary_filter(angle, gyro, accel_angle, dt, alpha):
    angle = np.asarray(angle, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    accel_angle = np.asarray(accel_angle, dtype=np.float64)
    out = (1.0 - alpha) * (angle + gyro * dt) + alpha * accel_angle
    return float(out) if out.ndim == 0 else out
""",
    ),
    (
        "gyro_integrate",
        "gyro_integrate",
        "the increment is composed on the left, as if omega were in the world frame",
        """
def gyro_integrate(q, omega, dt):
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega)) * dt
    if theta < 1e-8:
        dq = np.concatenate(([1.0], 0.5 * omega * dt))
    else:
        axis = omega / np.linalg.norm(omega)
        dq = np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * axis))
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = q
    out = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return out / np.linalg.norm(out)
""",
    ),
    (
        "gyro_integrate",
        "gyro_integrate",
        "full angle instead of half angle",
        """
def gyro_integrate(q, omega, dt):
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega)) * dt
    if theta < 1e-8:
        dq = np.concatenate(([1.0], omega * dt))
    else:
        axis = omega / np.linalg.norm(omega)
        dq = np.concatenate(([np.cos(theta)], np.sin(theta) * axis))
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = dq
    out = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return out / np.linalg.norm(out)
""",
    ),
    (
        "gyro_integrate",
        "gyro_integrate",
        "no guard for a zero rate",
        """
def gyro_integrate(q, omega, dt):
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega)) * dt
    with np.errstate(all="ignore"):
        axis = omega / np.linalg.norm(omega)
    dq = np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * axis))
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = dq
    out = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    with np.errstate(all="ignore"):
        return out / np.linalg.norm(out)
""",
    ),
    (
        "ekf_predict",
        "ekf_predict",
        "the state is pushed through F instead of f",
        """
def ekf_predict(x, P, f, F, Q):
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return F @ x, F @ P @ F.T + Q
""",
    ),
    (
        "ekf_predict",
        "ekf_predict",
        "process noise forgotten",
        """
def ekf_predict(x, P, f, F, Q):
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    return np.asarray(f(x), dtype=np.float64), F @ P @ F.T
""",
    ),
    (
        "ekf_predict",
        "ekf_predict",
        "the covariance is updated in place",
        """
def ekf_predict(x, P, f, F, Q):
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P[...] = F @ P @ F.T + Q
    return np.asarray(f(x), dtype=np.float64), P
""",
    ),
    (
        "ekf_predict",
        "ekf_predict",
        "F applied on one side only",
        """
def ekf_predict(x, P, f, F, Q):
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return np.asarray(f(x), dtype=np.float64), F @ P + Q
""",
    ),
    (
        "ekf_update",
        "ekf_update",
        "the gain divides by R instead of S",
        """
def ekf_update(x, P, z, h, H, R):
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    y = z - np.asarray(h(x), dtype=np.float64)
    K = np.linalg.solve(R, H @ P).T
    IKH = np.eye(len(x)) - K @ H
    return x + K @ y, IKH @ P @ IKH.T + K @ R @ K.T
""",
    ),
    (
        "ekf_update",
        "ekf_update",
        "innovation sign reversed",
        """
def ekf_update(x, P, z, h, H, R):
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(h(x), dtype=np.float64) - z
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    IKH = np.eye(len(x)) - K @ H
    return x + K @ y, IKH @ P @ IKH.T + K @ R @ K.T
""",
    ),
    (
        "ekf_update",
        "ekf_update",
        "the predicted measurement is H @ x instead of h(x)",
        """
def ekf_update(x, P, z, h, H, R):
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    y = z - H @ x
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    IKH = np.eye(len(x)) - K @ H
    return x + K @ y, IKH @ P @ IKH.T + K @ R @ K.T
""",
    ),
    (
        "innovation_gate",
        "innovation_gate",
        "euclidean distance",
        """
def innovation_gate(z, z_pred, S, threshold):
    y = np.asarray(z, dtype=np.float64) - np.asarray(z_pred, dtype=np.float64)
    d2 = float(y @ y)
    return bool(d2 <= threshold), d2
""",
    ),
    (
        "innovation_gate",
        "innovation_gate",
        "only the diagonal of S is used",
        """
def innovation_gate(z, z_pred, S, threshold):
    y = np.asarray(z, dtype=np.float64) - np.asarray(z_pred, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    d2 = float(np.sum(y * y / np.diag(S)))
    return bool(d2 <= threshold), d2
""",
    ),
    (
        "innovation_gate",
        "innovation_gate",
        "strict comparison",
        """
def innovation_gate(z, z_pred, S, threshold):
    y = np.asarray(z, dtype=np.float64) - np.asarray(z_pred, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    d2 = float(y @ np.linalg.solve(S, y))
    return bool(d2 < threshold), d2
""",
    ),
    (
        "innovation_gate",
        "innovation_gate",
        "returns the distance, not its square",
        """
def innovation_gate(z, z_pred, S, threshold):
    y = np.asarray(z, dtype=np.float64) - np.asarray(z_pred, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    d = float(np.sqrt(y @ np.linalg.solve(S, y)))
    return bool(d <= threshold), d
""",
    ),
]
