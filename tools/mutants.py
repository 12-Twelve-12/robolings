"""Check that the tests reject plausible wrong answers.

Passing the reference solution only shows the tests are satisfiable. Each
mutant below is a mistake a learner could reasonably make. The tests of the
exercise it belongs to must fail for every one of them.

A mutant counts as rejected only when at least one test fails on an
assertion. A mutant that only blows up with a ``NameError`` or an
``IndexError`` has not shown that the tests check the result.

    python tools/mutants.py
    python tools/mutants.py --jobs 2    limit the number of pytest processes
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

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
        "delta_actions",
        "to_delta",
        "difference between consecutive actions",
        """
def to_delta(chunk, state, absolute_mask):
    chunk = np.asarray(chunk, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    mask = np.asarray(absolute_mask, dtype=bool)
    previous = np.vstack([state[None, :], chunk[:-1]])
    return np.where(mask, chunk, chunk - previous)
""",
    ),
    (
        "delta_actions",
        "to_delta",
        "mask ignored",
        """
def to_delta(chunk, state, absolute_mask):
    return np.asarray(chunk, dtype=np.float64) - np.asarray(state, dtype=np.float64)
""",
    ),
    (
        "delta_actions",
        "to_delta",
        "mask inverted",
        """
def to_delta(chunk, state, absolute_mask):
    chunk = np.asarray(chunk, dtype=np.float64)
    mask = np.asarray(absolute_mask, dtype=bool)
    return np.where(mask, chunk - np.asarray(state, dtype=np.float64), chunk)
""",
    ),
    (
        "delta_actions",
        "from_delta",
        "accumulates the deltas",
        """
def from_delta(delta, state, absolute_mask):
    delta = np.asarray(delta, dtype=np.float64)
    mask = np.asarray(absolute_mask, dtype=bool)
    return np.where(mask, delta, np.cumsum(delta, axis=0) + np.asarray(state, dtype=np.float64))
""",
    ),
    (
        "delta_actions",
        "from_delta",
        "mask ignored",
        """
def from_delta(delta, state, absolute_mask):
    return np.asarray(delta, dtype=np.float64) + np.asarray(state, dtype=np.float64)
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "noise added on the last step",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(1.0 - beta)
    var = beta * (1.0 - alphas_cumprod[t - 1]) / (1.0 - a_t)
    return mean + np.sqrt(var) * noise
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "variance is beta",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(1.0 - beta)
    if t == 0:
        return mean
    return mean + np.sqrt(beta) * noise
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "divides by the cumulative alpha",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
    if t == 0:
        return mean
    var = beta * (1.0 - alphas_cumprod[t - 1]) / (1.0 - a_t)
    return mean + np.sqrt(var) * noise
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "noise scaled by the variance, not the std",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(1.0 - beta)
    if t == 0:
        return mean
    var = beta * (1.0 - alphas_cumprod[t - 1]) / (1.0 - a_t)
    return mean + var * noise
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
        "dct_tokens",
        "dct_matrix",
        "no scale factors",
        """
def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.cos(np.pi * (2 * i + 1) * k / (2 * n))
""",
    ),
    (
        "dct_tokens",
        "dct_matrix",
        "the same scale on every row",
        """
def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
""",
    ),
    (
        "dct_tokens",
        "tokenize",
        "truncates instead of rounding",
        """
def tokenize(chunk, step):
    chunk = np.asarray(chunk, dtype=np.float64)
    return (dct_matrix(chunk.shape[0]) @ chunk / step).astype(np.int64)
""",
    ),
    (
        "dct_tokens",
        "detokenize",
        "forward transform instead of the inverse",
        """
def detokenize(tokens, step):
    tokens = np.asarray(tokens, dtype=np.float64)
    return dct_matrix(tokens.shape[0]) @ (tokens * step)
""",
    ),
    (
        "dct_tokens",
        "detokenize",
        "does not undo the scaling",
        """
def detokenize(tokens, step):
    tokens = np.asarray(tokens, dtype=np.float64)
    return dct_matrix(tokens.shape[0]).T @ tokens
""",
    ),
    (
        "ema_weights",
        "ema_decay",
        "constant decay, no ramp",
        """
def ema_decay(step, max_decay=0.9999, warmup=10.0):
    return float(max_decay)
""",
    ),
    (
        "ema_weights",
        "ema_decay",
        "off by one in the ramp",
        """
def ema_decay(step, max_decay=0.9999, warmup=10.0):
    return float(min(max_decay, step / (warmup + step)))
""",
    ),
    (
        "ema_weights",
        "ema_decay",
        "not capped",
        """
def ema_decay(step, max_decay=0.9999, warmup=10.0):
    return float((1.0 + step) / (warmup + step))
""",
    ),
    (
        "ema_weights",
        "ema_update",
        "updates in place",
        """
def ema_update(average, weights, step, max_decay=0.9999, warmup=10.0):
    decay = ema_decay(step, max_decay, warmup)
    for k in average:
        average[k] *= decay
        average[k] += (1.0 - decay) * weights[k]
    return average
""",
    ),
    (
        "ema_weights",
        "ema_update",
        "decay on the wrong term",
        """
def ema_update(average, weights, step, max_decay=0.9999, warmup=10.0):
    decay = ema_decay(step, max_decay, warmup)
    return {k: (1.0 - decay) * np.asarray(average[k], dtype=np.float64) + decay * weights[k] for k in average}
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
    (
        "cfg_noise",
        "cfg_noise",
        "the difference is the wrong way round",
        """
def cfg_noise(eps_cond, eps_uncond, scale):
    eps_cond = np.asarray(eps_cond, dtype=np.float64)
    eps_uncond = np.asarray(eps_uncond, dtype=np.float64)
    return eps_uncond + scale * (eps_uncond - eps_cond)
""",
    ),
    (
        "cfg_noise",
        "cfg_noise",
        "the conditional prediction is used as the base",
        """
def cfg_noise(eps_cond, eps_uncond, scale):
    eps_cond = np.asarray(eps_cond, dtype=np.float64)
    eps_uncond = np.asarray(eps_uncond, dtype=np.float64)
    return eps_cond + scale * (eps_cond - eps_uncond)
""",
    ),
    (
        "cfg_noise",
        "cfg_noise",
        "interpolates between the two instead of extrapolating",
        """
def cfg_noise(eps_cond, eps_uncond, scale):
    eps_cond = np.asarray(eps_cond, dtype=np.float64)
    eps_uncond = np.asarray(eps_uncond, dtype=np.float64)
    return (1.0 - scale) * eps_cond + scale * eps_uncond
""",
    ),
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


# "path:line: assert ..." or "path:line: AssertionError" in the --tb=line output
FAILURE_LINE = re.compile(r"^.+?:\d+: (.*)$")


def failure_kinds(output):
    """What each failed test raised, from pytest's ``--tb=line`` output."""
    kinds = []
    for line in output.splitlines():
        found = FAILURE_LINE.match(line)
        if found is None:
            continue
        text = found.group(1)
        if text.startswith("assert") or text.startswith("AssertionError"):
            kinds.append("assertion")
        else:
            kinds.append(text.split(":", 1)[0].split("(", 1)[0].strip() or "exception")
    return kinds


def run_mutant(number, name, function, source, entry, tmp):
    target = pathlib.Path(tmp) / f"m{number:03d}"
    shutil.copytree(ROOT / "solutions", target)
    module = target / entry["track"] / f"{name}.py"
    original = module.read_text(encoding="utf-8")
    if f"def {function}(" not in original:
        return f"mutant {number}: {function} not found in {module.name}", []
    # A later definition replaces the earlier one at import time.
    module.write_text(original + "\n\n" + source.lstrip("\n"), encoding="utf-8", newline="\n")

    node = f"{ROOT / 'tests' / ('test_' + entry['track'] + '.py')}::{entry['test']}"
    env = dict(os.environ, ROBOLINGS_TARGET=str(target))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider", node],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    # 1 = tests ran and at least one failed. Anything else is not a kill.
    if result.returncode != 1:
        return f"pytest exit code {result.returncode}", []
    return None, failure_kinds(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 1), help="pytest processes to run at once"
    )
    args = parser.parse_args()

    registry = {e["name"]: e for e in json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))}

    unknown = sorted({m[0] for m in MUTANTS} - set(registry))
    if unknown:
        print("mutants for exercises that do not exist: " + ", ".join(unknown))
        return 1
    missing = [name for name in registry if name not in {m[0] for m in MUTANTS}]
    if missing:
        print("exercises without any mutant: " + ", ".join(missing))
        return 1

    for number, (name, function, _label, source) in enumerate(MUTANTS):
        if function not in registry[name]["functions"]:
            print(f"mutant {number}: {function} is not a function of {name}")
            return 1
        if f"def {function}(" not in source:
            print(f"mutant {number}: replacement does not define {function}")
            return 1

    survived = []
    with tempfile.TemporaryDirectory() as tmp:

        def job(item):
            number, (name, function, label, source) = item
            return run_mutant(number, name, function, source, registry[name], tmp)

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            outcomes = list(pool.map(job, enumerate(MUTANTS)))

    for (name, function, label, _), (problem, kinds) in zip(MUTANTS, outcomes, strict=True):
        if problem is not None:
            survived.append((name, function, label, problem))
        elif "assertion" not in kinds:
            raised = ", ".join(sorted(set(kinds)))
            survived.append((name, function, label, f"only raised {raised}, no assertion failed"))

    killed = len(MUTANTS) - len(survived)
    print(f"{killed} of {len(MUTANTS)} mutants rejected by the tests of their own exercise")
    for name, function, label, why in survived:
        print(f"  NOT rejected: {name} {function}(): {label} ({why})")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
