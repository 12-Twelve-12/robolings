"""Wrong answers for track 07, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "flow_matching_target",
        "flow_matching_target",
        "velocity points from data to noise",
        """
def flow_matching_target(noise, data, t):
    noise = np.asarray(noise, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape((-1,) + (1,) * (data.ndim - 1))
    return (1.0 - t) * noise + t * data, noise - data
""",
    ),
    (
        "flow_matching_target",
        "flow_matching_target",
        "t = 0 is data",
        """
def flow_matching_target(noise, data, t):
    noise = np.asarray(noise, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape((-1,) + (1,) * (data.ndim - 1))
    return t * noise + (1.0 - t) * data, data - noise
""",
    ),
    (
        "flow_matching_target",
        "flow_matching_target",
        "first time used for the whole batch",
        """
def flow_matching_target(noise, data, t):
    noise = np.asarray(noise, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    t0 = float(np.asarray(t).reshape(-1)[0])
    return (1.0 - t0) * noise + t0 * data, data - noise
""",
    ),
    (
        "euler_sample",
        "euler_sample",
        "velocity evaluated at the end of the step",
        """
def euler_sample(v_fn, x, num_steps):
    x = np.asarray(x, dtype=np.float64).copy()
    dt = 1.0 / num_steps
    with np.errstate(all="ignore"):
        for i in range(num_steps):
            x = x + dt * v_fn(x, (i + 1) * dt)
    return x
""",
    ),
    (
        "euler_sample",
        "euler_sample",
        "input modified in place",
        """
def euler_sample(v_fn, x, num_steps):
    dt = 1.0 / num_steps
    for i in range(num_steps):
        x += dt * v_fn(x, i * dt)
    return x
""",
    ),
    (
        "euler_sample",
        "euler_sample",
        "step size of one",
        """
def euler_sample(v_fn, x, num_steps):
    x = np.asarray(x, dtype=np.float64).copy()
    for i in range(num_steps):
        x = x + v_fn(x, i / num_steps)
    return x
""",
    ),
    (
        "midpoint_sample",
        "midpoint_sample",
        "plain Euler",
        """
def midpoint_sample(v_fn, x, num_steps):
    x = np.asarray(x, dtype=np.float64).copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        v_fn(x, i * dt + 0.5 * dt)
        x = x + dt * v_fn(x, i * dt)
    return x
""",
    ),
    (
        "midpoint_sample",
        "midpoint_sample",
        "second call uses the old state",
        """
def midpoint_sample(v_fn, x, num_steps):
    x = np.asarray(x, dtype=np.float64).copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        v_fn(x, i * dt)
        x = x + dt * v_fn(x, i * dt + 0.5 * dt)
    return x
""",
    ),
    (
        "midpoint_sample",
        "midpoint_sample",
        "second call uses the old time",
        """
def midpoint_sample(v_fn, x, num_steps):
    x = np.asarray(x, dtype=np.float64).copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        x_mid = x + 0.5 * dt * v_fn(x, i * dt)
        x = x + dt * v_fn(x_mid, i * dt)
    return x
""",
    ),
    (
        "midpoint_sample",
        "midpoint_sample",
        "full step to the midpoint",
        """
def midpoint_sample(v_fn, x, num_steps):
    x = np.asarray(x, dtype=np.float64).copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        x_mid = x + dt * v_fn(x, i * dt)
        x = x + dt * v_fn(x_mid, i * dt + 0.5 * dt)
    return x
""",
    ),
    (
        "midpoint_sample",
        "midpoint_sample",
        "input modified in place",
        """
def midpoint_sample(v_fn, x, num_steps):
    dt = 1.0 / num_steps
    for i in range(num_steps):
        x_mid = x + 0.5 * dt * v_fn(x, i * dt)
        x += dt * v_fn(x_mid, i * dt + 0.5 * dt)
    return x
""",
    ),
]
