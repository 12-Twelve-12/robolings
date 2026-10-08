"""Wrong answers for track 03, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "expand_mimic",
        "expand_mimic",
        "offset ignored",
        """
def expand_mimic(q_active, n_joints, active_idx, mimic):
    q = np.zeros(n_joints)
    q[np.asarray(active_idx, dtype=int)] = q_active
    for joint, source, multiplier, offset in mimic:
        q[joint] = multiplier * q[source]
    return q
""",
    ),
    (
        "expand_mimic",
        "expand_mimic",
        "source indexes the active vector",
        """
def expand_mimic(q_active, n_joints, active_idx, mimic):
    q = np.zeros(n_joints)
    q[np.asarray(active_idx, dtype=int)] = q_active
    for joint, source, multiplier, offset in mimic:
        q[joint] = multiplier * q_active[min(source, len(q_active) - 1)] + offset
    return q
""",
    ),
    (
        "retarget_cost",
        "retarget_cost",
        "factor of one half missing",
        """
def retarget_cost(human_vecs, robot_vecs, scale, q, q_prev, beta):
    diff = scale * np.asarray(human_vecs) - np.asarray(robot_vecs)
    return float(np.sum(diff * diff) + beta * np.sum((np.asarray(q) - np.asarray(q_prev)) ** 2))
""",
    ),
    (
        "retarget_cost",
        "retarget_cost",
        "scale applied to the robot",
        """
def retarget_cost(human_vecs, robot_vecs, scale, q, q_prev, beta):
    diff = np.asarray(human_vecs) - scale * np.asarray(robot_vecs)
    return float(0.5 * np.sum(diff * diff) + 0.5 * beta * np.sum((np.asarray(q) - np.asarray(q_prev)) ** 2))
""",
    ),
    (
        "retarget_cost",
        "retarget_cost",
        "returns a numpy scalar",
        """
def retarget_cost(human_vecs, robot_vecs, scale, q, q_prev, beta):
    diff = scale * np.asarray(human_vecs) - np.asarray(robot_vecs)
    return 0.5 * np.sum(diff * diff) + 0.5 * beta * np.sum((np.asarray(q) - np.asarray(q_prev)) ** 2)
""",
    ),
    (
        "fingertip_ik",
        "fingertip_ik",
        "limits applied only at the end",
        """
def fingertip_ik(fk_fn, jac_fn, q0, target, lower, upper, iters, damping):
    q = np.asarray(q0, dtype=np.float64).copy()
    for _ in range(iters):
        J = jac_fn(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), target - fk_fn(q))
    return np.clip(q, lower, upper)
""",
    ),
    (
        "fingertip_ik",
        "fingertip_ik",
        "limits ignored",
        """
def fingertip_ik(fk_fn, jac_fn, q0, target, lower, upper, iters, damping):
    q = np.asarray(q0, dtype=np.float64).copy()
    for _ in range(iters):
        J = jac_fn(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), target - fk_fn(q))
    return q
""",
    ),
    (
        "fingertip_ik",
        "fingertip_ik",
        "q0 modified in place",
        """
def fingertip_ik(fk_fn, jac_fn, q0, target, lower, upper, iters, damping):
    q = q0
    for _ in range(iters):
        J = jac_fn(q)
        q += J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), target - fk_fn(q))
        np.clip(q, lower, upper, out=q)
    return q
""",
    ),
    (
        "friction_cone",
        "project_to_friction_cone",
        "rescales the whole force instead of projecting it",
        """
def project_to_friction_cone(force, normal, mu):
    f = np.asarray(force, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    f_n = float(f @ n)
    f_t = f - f_n * n
    t_norm = float(np.linalg.norm(f_t))
    if t_norm <= mu * f_n:
        return f.copy()
    if mu * t_norm <= -f_n:
        return np.zeros(3)
    return f * (mu * f_n / t_norm)
""",
    ),
    (
        "friction_cone",
        "project_to_friction_cone",
        "drops the tangential part instead of projecting onto the cone",
        """
def project_to_friction_cone(force, normal, mu):
    f = np.asarray(force, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    f_n = float(f @ n)
    f_t = f - f_n * n
    if np.linalg.norm(f_t) <= mu * f_n:
        return f.copy()
    if f_n <= 0.0:
        return np.zeros(3)
    return f_n * n
""",
    ),
    (
        "friction_cone",
        "project_to_friction_cone",
        "a force pressing into the surface is mirrored instead of dropped",
        """
def project_to_friction_cone(force, normal, mu):
    f = np.asarray(force, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    f_n = float(f @ n)
    f_t = f - f_n * n
    t_norm = float(np.linalg.norm(f_t))
    if t_norm <= mu * f_n:
        return f.copy()
    if mu * t_norm <= -f_n:
        return -f
    s = (mu * t_norm + f_n) / (mu**2 + 1.0)
    return s * n + mu * s * (f_t / t_norm)
""",
    ),
    (
        "force_closure",
        "is_force_closure",
        "only checks that the two normals oppose each other",
        """
def is_force_closure(contacts, normals, mu):
    n = np.asarray(normals, dtype=np.float64)
    n0 = n[0] / np.linalg.norm(n[0])
    n1 = n[1] / np.linalg.norm(n[1])
    return bool(n0 @ n1 < 0.0)
""",
    ),
    (
        "force_closure",
        "is_force_closure",
        "ignores mu, so only a perfectly aligned pair holds",
        """
def is_force_closure(contacts, normals, mu):
    c = np.asarray(contacts, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    d = c[1] - c[0]
    length = np.linalg.norm(d)
    if length == 0.0:
        return False
    d = d / length
    for direction, normal in ((d, n[0]), (-d, n[1])):
        unit = normal / np.linalg.norm(normal)
        along = float(direction @ unit)
        across = float(np.linalg.norm(direction - along * unit))
        if along <= 0.0 or across > 1e-12:
            return False
    return True
""",
    ),
    (
        "force_closure",
        "is_force_closure",
        "tests the same direction at both contacts",
        """
def is_force_closure(contacts, normals, mu):
    c = np.asarray(contacts, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    d = c[1] - c[0]
    length = np.linalg.norm(d)
    if length == 0.0:
        return False
    d = d / length
    for direction, normal in ((d, n[0]), (d, n[1])):
        unit = normal / np.linalg.norm(normal)
        along = float(direction @ unit)
        across = float(np.linalg.norm(direction - along * unit))
        if along <= 0.0 or across > mu * along + 1e-12:
            return False
    return True
""",
    ),
    (
        "coupled_jacobian",
        "coupled_jacobian",
        "the offset is folded into the derivative",
        """
def coupled_jacobian(J_full, n_joints, active_idx, mimic):
    J = np.asarray(J_full, dtype=np.float64)
    active = np.asarray(active_idx, dtype=int)
    k = active.size
    C = np.zeros((n_joints, k), dtype=np.float64)
    C[active, np.arange(k)] = 1.0
    column = {int(joint): index for index, joint in enumerate(active)}
    for joint, source, multiplier, offset in mimic:
        C[joint, column[source]] = multiplier + offset
    return J @ C
""",
    ),
    (
        "coupled_jacobian",
        "coupled_jacobian",
        "the coupled joints contribute nothing",
        """
def coupled_jacobian(J_full, n_joints, active_idx, mimic):
    J = np.asarray(J_full, dtype=np.float64)
    active = np.asarray(active_idx, dtype=int)
    k = active.size
    C = np.zeros((n_joints, k), dtype=np.float64)
    C[active, np.arange(k)] = 1.0
    return J @ C
""",
    ),
    (
        "coupled_jacobian",
        "coupled_jacobian",
        "the multiplier is inverted",
        """
def coupled_jacobian(J_full, n_joints, active_idx, mimic):
    J = np.asarray(J_full, dtype=np.float64)
    active = np.asarray(active_idx, dtype=int)
    k = active.size
    C = np.zeros((n_joints, k), dtype=np.float64)
    C[active, np.arange(k)] = 1.0
    column = {int(joint): index for index, joint in enumerate(active)}
    for joint, source, multiplier, offset in mimic:
        C[joint, column[source]] = 1.0 / multiplier
    return J @ C
""",
    ),
]
