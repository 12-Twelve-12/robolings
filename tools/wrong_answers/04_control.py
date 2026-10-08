"""Wrong answers for track 04, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "min_jerk",
        "min_jerk",
        "chain rule forgotten",
        """
def min_jerk(q0, q1, duration, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    tau = float(np.clip(t / duration, 0.0, 1.0))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return q0 + (q1 - q0) * s, (q1 - q0) * ds, (q1 - q0) * dds
""",
    ),
    (
        "min_jerk",
        "min_jerk",
        "time is not clamped",
        """
def min_jerk(q0, q1, duration, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    tau = t / duration
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    return q0 + (q1 - q0) * s, (q1 - q0) * ds, (q1 - q0) * dds
""",
    ),
    (
        "min_jerk",
        "min_jerk",
        "cubic instead of quintic",
        """
def min_jerk(q0, q1, duration, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    tau = float(np.clip(t / duration, 0.0, 1.0))
    s = 3 * tau**2 - 2 * tau**3
    ds = (6 * tau - 6 * tau**2) / duration
    dds = (6 - 12 * tau) / duration**2
    return q0 + (q1 - q0) * s, (q1 - q0) * ds, (q1 - q0) * dds
""",
    ),
    (
        "lowpass_filter",
        "lowpass_filter",
        "starts from zero",
        """
def lowpass_filter(x, cutoff_hz, dt):
    x = np.asarray(x, dtype=np.float64)
    alpha = dt / (1.0 / (2.0 * np.pi * cutoff_hz) + dt)
    y = np.zeros_like(x)
    prev = np.zeros_like(x[0])
    for k in range(len(x)):
        prev = prev + alpha * (x[k] - prev)
        y[k] = prev
    return y
""",
    ),
    (
        "lowpass_filter",
        "lowpass_filter",
        "alpha = dt / rc",
        """
def lowpass_filter(x, cutoff_hz, dt):
    x = np.asarray(x, dtype=np.float64)
    alpha = dt * (2.0 * np.pi * cutoff_hz)
    y = np.zeros_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y
""",
    ),
    (
        "lowpass_filter",
        "lowpass_filter",
        "filters in place",
        """
def lowpass_filter(x, cutoff_hz, dt):
    alpha = dt / (1.0 / (2.0 * np.pi * cutoff_hz) + dt)
    for k in range(1, len(x)):
        x[k] = x[k - 1] + alpha * (x[k] - x[k - 1])
    return x
""",
    ),
    (
        "mit_torque",
        "mit_torque",
        "each term clipped separately",
        """
def mit_torque(kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit):
    a = np.clip(np.asarray(kp) * (np.asarray(q_des) - np.asarray(q)), -tau_limit, tau_limit)
    b = np.clip(np.asarray(kd) * (np.asarray(dq_des) - np.asarray(dq)), -tau_limit, tau_limit)
    c = np.clip(np.asarray(tau_ff, dtype=np.float64), -tau_limit, tau_limit)
    return np.clip(a + b + c, -tau_limit, tau_limit)
""",
    ),
    (
        "mit_torque",
        "mit_torque",
        "no limit",
        """
def mit_torque(kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit):
    return np.asarray(kp) * (np.asarray(q_des) - np.asarray(q)) + np.asarray(kd) * (np.asarray(dq_des) - np.asarray(dq)) + np.asarray(tau_ff)
""",
    ),
    (
        "mit_torque",
        "mit_torque",
        "damping sign reversed",
        """
def mit_torque(kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit):
    tau = np.asarray(kp) * (np.asarray(q_des) - np.asarray(q)) + np.asarray(kd) * (np.asarray(dq) - np.asarray(dq_des)) + np.asarray(tau_ff)
    return np.clip(tau, -tau_limit, tau_limit)
""",
    ),
    (
        "rate_limit",
        "rate_limit",
        "clips the target instead of the change",
        """
def rate_limit(target, prev, max_rate, dt):
    step = np.asarray(max_rate, dtype=np.float64) * dt
    return np.clip(np.asarray(target, dtype=np.float64), -step, step)
""",
    ),
    (
        "rate_limit",
        "rate_limit",
        "dt ignored",
        """
def rate_limit(target, prev, max_rate, dt):
    target = np.asarray(target, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    step = np.asarray(max_rate, dtype=np.float64)
    return prev + np.clip(target - prev, -step, step)
""",
    ),
    (
        "angle_diff",
        "wrap_to_pi",
        "wraps to [0, 2 pi)",
        """
def wrap_to_pi(a):
    return np.asarray(a, dtype=np.float64) % (2.0 * np.pi)
""",
    ),
    (
        "angle_diff",
        "wrap_to_pi",
        "clips instead of wrapping",
        """
def wrap_to_pi(a):
    return np.clip(np.asarray(a, dtype=np.float64), -np.pi, np.pi)
""",
    ),
    (
        "angle_diff",
        "angle_diff",
        "raw difference",
        """
def angle_diff(a, b):
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
""",
    ),
    (
        "angle_diff",
        "angle_diff",
        "wraps the inputs but not the difference",
        """
def angle_diff(a, b):
    def w(x):
        return (np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
    return w(a) - w(b)
""",
    ),
    (
        "angle_diff",
        "angle_diff",
        "sign reversed",
        """
def angle_diff(a, b):
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return (d + np.pi) % (2.0 * np.pi) - np.pi
""",
    ),
    (
        "pid_step",
        "pid_step",
        "no anti-windup",
        """
def pid_step(state, error, dt, kp, ki, kd, u_limit):
    integral, prev_error = state
    integral = integral + error * dt
    derivative = 0.0 if prev_error is None else (error - prev_error) / dt
    u = kp * error + ki * integral + kd * derivative
    return min(max(u, -u_limit), u_limit), (integral, error)
""",
    ),
    (
        "pid_step",
        "pid_step",
        "integral frozen whenever saturated",
        """
def pid_step(state, error, dt, kp, ki, kd, u_limit):
    integral, prev_error = state
    integral_new = integral + error * dt
    derivative = 0.0 if prev_error is None else (error - prev_error) / dt
    u_raw = kp * error + ki * integral_new + kd * derivative
    u = min(max(u_raw, -u_limit), u_limit)
    if u != u_raw:
        integral_new = integral
    return u, (integral_new, error)
""",
    ),
    (
        "pid_step",
        "pid_step",
        "derivative kick on the first call",
        """
def pid_step(state, error, dt, kp, ki, kd, u_limit):
    integral, prev_error = state
    integral_new = integral + error * dt
    derivative = (error - (0.0 if prev_error is None else prev_error)) / dt
    u_raw = kp * error + ki * integral_new + kd * derivative
    u = min(max(u_raw, -u_limit), u_limit)
    if u != u_raw and u_raw * error > 0.0:
        integral_new = integral
    return u, (integral_new, error)
""",
    ),
    (
        "pid_step",
        "pid_step",
        "output is not clipped",
        """
def pid_step(state, error, dt, kp, ki, kd, u_limit):
    integral, prev_error = state
    integral_new = integral + error * dt
    derivative = 0.0 if prev_error is None else (error - prev_error) / dt
    u_raw = kp * error + ki * integral_new + kd * derivative
    if abs(u_raw) > u_limit and u_raw * error > 0.0:
        integral_new = integral
    return u_raw, (integral_new, error)
""",
    ),
    (
        "pid_step",
        "pid_step",
        "integral from before this step",
        """
def pid_step(state, error, dt, kp, ki, kd, u_limit):
    integral, prev_error = state
    integral_new = integral + error * dt
    derivative = 0.0 if prev_error is None else (error - prev_error) / dt
    u_raw = kp * error + ki * integral + kd * derivative
    u = min(max(u_raw, -u_limit), u_limit)
    if u != u_raw and u_raw * error > 0.0:
        integral_new = integral
    return u, (integral_new, error)
""",
    ),
    (
        "trapezoid",
        "trapezoid",
        "assumes the speed limit is always reached",
        """
def trapezoid(distance, v_max, a_max, t):
    distance = float(distance)
    if distance <= 0.0:
        return 0.0, 0.0
    t_ramp = v_max / a_max
    d_ramp = v_max * t_ramp / 2.0
    t_flat = (distance - 2.0 * d_ramp) / v_max
    total = 2.0 * t_ramp + t_flat
    if t <= 0.0:
        return 0.0, 0.0
    if t >= total:
        return distance, 0.0
    if t < t_ramp:
        return 0.5 * a_max * t * t, a_max * t
    if t < t_ramp + t_flat:
        return d_ramp + v_max * (t - t_ramp), v_max
    left = total - t
    return distance - 0.5 * a_max * left * left, a_max * left
""",
    ),
    (
        "trapezoid",
        "trapezoid",
        "deceleration measured from the wrong end",
        """
def trapezoid(distance, v_max, a_max, t):
    distance = float(distance)
    if distance <= 0.0:
        return 0.0, 0.0
    peak = min(v_max, np.sqrt(a_max * distance))
    t_ramp = peak / a_max
    d_ramp = peak * t_ramp / 2.0
    t_flat = (distance - 2.0 * d_ramp) / peak
    total = 2.0 * t_ramp + t_flat
    if t <= 0.0:
        return 0.0, 0.0
    if t >= total:
        return distance, 0.0
    if t < t_ramp:
        return 0.5 * a_max * t * t, a_max * t
    if t < t_ramp + t_flat:
        return d_ramp + peak * (t - t_ramp), peak
    left = t - t_ramp - t_flat
    return distance - 0.5 * a_max * left * left, a_max * left
""",
    ),
    (
        "trapezoid",
        "trapezoid",
        "does not stop at the end",
        """
def trapezoid(distance, v_max, a_max, t):
    distance = float(distance)
    if distance <= 0.0:
        return 0.0, 0.0
    peak = min(v_max, np.sqrt(a_max * distance))
    t_ramp = peak / a_max
    d_ramp = peak * t_ramp / 2.0
    t_flat = (distance - 2.0 * d_ramp) / peak
    total = 2.0 * t_ramp + t_flat
    if t <= 0.0:
        return 0.0, 0.0
    if t < t_ramp:
        return 0.5 * a_max * t * t, a_max * t
    if t < t_ramp + t_flat:
        return d_ramp + peak * (t - t_ramp), peak
    left = total - t
    return distance - 0.5 * a_max * left * left, a_max * left
""",
    ),
    (
        "gravity_torque",
        "gravity_torque",
        "relative angles instead of absolute",
        """
def gravity_torque(q, lengths, masses, com, g=9.81):
    q = np.asarray(q, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    com = np.asarray(com, dtype=np.float64)
    n = len(q)
    tau = np.zeros(n)
    for k in range(n):
        total = 0.0
        for i in range(k, n):
            arm = com[i] * np.cos(q[i])
            for j in range(k, i):
                arm += lengths[j] * np.cos(q[j])
            total += masses[i] * arm
        tau[k] = g * total
    return tau
""",
    ),
    (
        "gravity_torque",
        "gravity_torque",
        "each joint carries only its own link",
        """
def gravity_torque(q, lengths, masses, com, g=9.81):
    q = np.asarray(q, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    com = np.asarray(com, dtype=np.float64)
    absolute = np.cumsum(q)
    return g * masses * com * np.cos(absolute)
""",
    ),
    (
        "gravity_torque",
        "gravity_torque",
        "sine instead of cosine",
        """
def gravity_torque(q, lengths, masses, com, g=9.81):
    q = np.asarray(q, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    com = np.asarray(com, dtype=np.float64)
    n = len(q)
    absolute = np.cumsum(q)
    tau = np.zeros(n)
    for k in range(n):
        total = 0.0
        for i in range(k, n):
            arm = com[i] * np.sin(absolute[i])
            for j in range(k, i):
                arm += lengths[j] * np.sin(absolute[j])
            total += masses[i] * arm
        tau[k] = g * total
    return tau
""",
    ),
    (
        "admittance_step",
        "admittance_step",
        "explicit Euler: x advances with the old velocity",
        """
def admittance_step(x, v, x_ref, f_ext, M, D, K, dt):
    x, v, x_ref, f_ext = (np.asarray(a, dtype=np.float64) for a in (x, v, x_ref, f_ext))
    M, D, K = (np.asarray(a, dtype=np.float64) for a in (M, D, K))
    a = (f_ext - D * v - K * (x - x_ref)) / M
    x_new = x + v * dt
    v_new = v + a * dt
    return x_new, v_new
""",
    ),
    (
        "admittance_step",
        "admittance_step",
        "the spring is anchored at zero instead of x_ref",
        """
def admittance_step(x, v, x_ref, f_ext, M, D, K, dt):
    x, v, x_ref, f_ext = (np.asarray(a, dtype=np.float64) for a in (x, v, x_ref, f_ext))
    M, D, K = (np.asarray(a, dtype=np.float64) for a in (M, D, K))
    a = (f_ext - D * v - K * x) / M
    v_new = v + a * dt
    x_new = x + v_new * dt
    return x_new, v_new
""",
    ),
    (
        "admittance_step",
        "admittance_step",
        "damping sign reversed",
        """
def admittance_step(x, v, x_ref, f_ext, M, D, K, dt):
    x, v, x_ref, f_ext = (np.asarray(a, dtype=np.float64) for a in (x, v, x_ref, f_ext))
    M, D, K = (np.asarray(a, dtype=np.float64) for a in (M, D, K))
    a = (f_ext + D * v - K * (x - x_ref)) / M
    v_new = v + a * dt
    x_new = x + v_new * dt
    return x_new, v_new
""",
    ),
    (
        "admittance_step",
        "admittance_step",
        "the mass is forgotten",
        """
def admittance_step(x, v, x_ref, f_ext, M, D, K, dt):
    x, v, x_ref, f_ext = (np.asarray(a, dtype=np.float64) for a in (x, v, x_ref, f_ext))
    D, K = (np.asarray(a, dtype=np.float64) for a in (D, K))
    a = f_ext - D * v - K * (x - x_ref)
    v_new = v + a * dt
    x_new = x + v_new * dt
    return x_new, v_new
""",
    ),
]
