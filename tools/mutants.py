"""Check that the tests reject plausible wrong answers.

Passing the reference solution only shows the tests are satisfiable. Each
mutant below is a mistake a learner could reasonably make. The tests of the
exercise it belongs to must fail for every one of them.

    python tools/mutants.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

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
        "minmax",
        "minmax_normalize",
        "constant dimension gives nan",
        """
def minmax_normalize(x, lo, hi):
    x, lo, hi = (np.asarray(v, dtype=np.float64) for v in (x, lo, hi))
    with np.errstate(all="ignore"):
        return 2.0 * (x - lo) / (hi - lo) - 1.0
""",
    ),
    (
        "minmax",
        "minmax_normalize",
        "maps to [0, 1]",
        """
def minmax_normalize(x, lo, hi):
    x, lo, hi = (np.asarray(v, dtype=np.float64) for v in (x, lo, hi))
    span = hi - lo
    return np.where(span == 0.0, 0.0, (x - lo) / np.where(span == 0.0, 1.0, span))
""",
    ),
    (
        "minmax",
        "minmax_unnormalize",
        "inverse of a [0, 1] mapping",
        """
def minmax_unnormalize(y, lo, hi):
    y, lo, hi = (np.asarray(v, dtype=np.float64) for v in (y, lo, hi))
    return y * (hi - lo) + lo
""",
    ),
    (
        "action_chunks",
        "make_action_chunks",
        "pads with zeros",
        """
def make_action_chunks(actions, horizon):
    actions = np.asarray(actions, dtype=np.float64)
    T = actions.shape[0]
    idx = np.arange(T)[:, None] + np.arange(horizon)[None, :]
    is_pad = idx >= T
    chunks = actions[np.minimum(idx, T - 1)]
    chunks[is_pad] = 0.0
    return chunks, is_pad
""",
    ),
    (
        "action_chunks",
        "make_action_chunks",
        "mask inverted",
        """
def make_action_chunks(actions, horizon):
    actions = np.asarray(actions, dtype=np.float64)
    T = actions.shape[0]
    idx = np.arange(T)[:, None] + np.arange(horizon)[None, :]
    return actions[np.minimum(idx, T - 1)], idx < T
""",
    ),
    (
        "action_chunks",
        "make_action_chunks",
        "drops the incomplete chunks",
        """
def make_action_chunks(actions, horizon):
    actions = np.asarray(actions, dtype=np.float64)
    T = max(actions.shape[0] - horizon + 1, 0)
    idx = np.arange(T)[:, None] + np.arange(horizon)[None, :]
    return actions[idx], np.zeros((T, horizon), dtype=bool)
""",
    ),
    (
        "temporal_ensemble",
        "temporal_ensemble",
        "newest prediction weighs most",
        """
def temporal_ensemble(preds, m):
    preds = np.asarray(preds, dtype=np.float64)
    w = np.exp(-m * np.arange(preds.shape[0]))[::-1]
    return (w[:, None] * preds).sum(axis=0) / w.sum()
""",
    ),
    (
        "temporal_ensemble",
        "temporal_ensemble",
        "weights are not normalised",
        """
def temporal_ensemble(preds, m):
    preds = np.asarray(preds, dtype=np.float64)
    w = np.exp(-m * np.arange(preds.shape[0]))
    return (w[:, None] * preds).sum(axis=0) / preds.shape[0]
""",
    ),
    (
        "obs_history",
        "stack_obs_history",
        "pads with zeros",
        """
def stack_obs_history(obs, n):
    obs = np.asarray(obs, dtype=np.float64)
    T = obs.shape[0]
    idx = np.arange(T)[:, None] + np.arange(-n + 1, 1)[None, :]
    out = obs[np.maximum(idx, 0)]
    out[idx < 0] = 0.0
    return out
""",
    ),
    (
        "obs_history",
        "stack_obs_history",
        "newest observation first",
        """
def stack_obs_history(obs, n):
    obs = np.asarray(obs, dtype=np.float64)
    T = obs.shape[0]
    idx = np.arange(T)[:, None] + np.arange(-n + 1, 1)[None, :]
    return obs[np.maximum(idx, 0)][:, ::-1]
""",
    ),
    (
        "cosine_schedule",
        "cosine_schedule",
        "beta is not capped",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    def f(u):
        return np.cos((u + s) / (1.0 + s) * np.pi / 2.0) ** 2
    i = np.arange(num_steps, dtype=np.float64)
    betas = 1.0 - f((i + 1.0) / num_steps) / f(i / num_steps)
    return betas, np.cumprod(1.0 - betas)
""",
    ),
    (
        "cosine_schedule",
        "cosine_schedule",
        "offset s ignored",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    def f(u):
        return np.cos(u * np.pi / 2.0) ** 2
    i = np.arange(num_steps, dtype=np.float64)
    betas = np.minimum(1.0 - f((i + 1.0) / num_steps) / f(i / num_steps), max_beta)
    return betas, np.cumprod(1.0 - betas)
""",
    ),
    (
        "cosine_schedule",
        "cosine_schedule",
        "linear schedule",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    betas = np.linspace(1e-4, 0.02, num_steps)
    return betas, np.cumprod(1.0 - betas)
""",
    ),
    (
        "cosine_schedule",
        "cosine_schedule",
        "returns alpha_bar of the cosine directly",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    def f(u):
        return np.cos((u + s) / (1.0 + s) * np.pi / 2.0) ** 2
    i = np.arange(num_steps, dtype=np.float64)
    betas = np.minimum(1.0 - f((i + 1.0) / num_steps) / f(i / num_steps), max_beta)
    return betas, f((i + 1.0) / num_steps) / f(0.0)
""",
    ),
    (
        "q_sample",
        "q_sample",
        "square roots missing",
        """
def q_sample(x0, t, noise, alphas_cumprod):
    x0 = np.asarray(x0, dtype=np.float64)
    a = np.asarray(alphas_cumprod)[np.asarray(t, dtype=int)].reshape((-1,) + (1,) * (x0.ndim - 1))
    return a * x0 + (1.0 - a) * np.asarray(noise, dtype=np.float64)
""",
    ),
    (
        "q_sample",
        "q_sample",
        "first timestep used for the whole batch",
        """
def q_sample(x0, t, noise, alphas_cumprod):
    a = np.asarray(alphas_cumprod)[np.asarray(t, dtype=int)][0]
    return np.sqrt(a) * np.asarray(x0, dtype=np.float64) + np.sqrt(1.0 - a) * np.asarray(noise, dtype=np.float64)
""",
    ),
    (
        "ddim_step",
        "ddim_step",
        "t_prev = -1 wraps to the last timestep",
        """
def ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    a_t = alphas_cumprod[t]
    a_prev = alphas_cumprod[t_prev]
    x0_pred = (x_t - np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
    return np.sqrt(a_prev) * x0_pred + np.sqrt(1.0 - a_prev) * eps_pred
""",
    ),
    (
        "ddim_step",
        "ddim_step",
        "always steps to t - 1",
        """
def ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    a_t = alphas_cumprod[t]
    a_prev = alphas_cumprod[t - 1] if t > 0 else 1.0
    x0_pred = (x_t - np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
    return np.sqrt(a_prev) * x0_pred + np.sqrt(1.0 - a_prev) * eps_pred
""",
    ),
    (
        "ddim_step",
        "ddim_step",
        "returns the predicted clean sample",
        """
def ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    a_t = alphas_cumprod[t]
    return (x_t - np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
""",
    ),
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
]


def main():
    registry = {e["name"]: e for e in json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))}

    unknown = sorted({m[0] for m in MUTANTS} - set(registry))
    if unknown:
        print("mutants for exercises that do not exist: " + ", ".join(unknown))
        return 1
    missing = [name for name in registry if name not in {m[0] for m in MUTANTS}]
    if missing:
        print("exercises without any mutant: " + ", ".join(missing))
        return 1

    survived = []
    killed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for number, (name, function, label, source) in enumerate(MUTANTS):
            entry = registry[name]
            if function not in entry["functions"]:
                print(f"mutant {number}: {function} is not a function of {name}")
                return 1
            if f"def {function}(" not in source:
                print(f"mutant {number}: replacement does not define {function}")
                return 1

            target = pathlib.Path(tmp) / f"m{number:03d}"
            shutil.copytree(ROOT / "solutions", target)
            module = target / entry["track"] / f"{name}.py"
            original = module.read_text(encoding="utf-8")
            if f"def {function}(" not in original:
                print(f"mutant {number}: {function} not found in {module.name}")
                return 1
            # A later definition replaces the earlier one at import time.
            module.write_text(original + "\n\n" + source.lstrip("\n"), encoding="utf-8", newline="\n")

            node = f"{ROOT / 'tests' / ('test_' + entry['track'] + '.py')}::{entry['test']}"
            env = dict(os.environ, ROBOLINGS_TARGET=str(target))
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider", node],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            # 1 = tests ran and at least one failed. Anything else is not a kill.
            if result.returncode == 1:
                killed += 1
            else:
                survived.append((name, function, label, result.returncode))

    print(f"{killed} of {len(MUTANTS)} mutants rejected by the tests of their own exercise")
    for name, function, label, code in survived:
        state = "passed every test" if code == 0 else f"pytest exit code {code}"
        print(f"  NOT rejected: {name} {function}(): {label} ({state})")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
