"""Reference code used by the tests.

Tests never call one exercise to check another, so a mistake in an early
exercise cannot hide or cause a failure in a later one.
"""

import numpy as np

TOL = 1e-9


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


ROT = {"x": rot_x, "y": rot_y, "z": rot_z}
AXIS = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


def euler_matrix(a, b, c):
    """A generic rotation built from the three elementary ones."""
    return rot_z(a) @ rot_y(b) @ rot_x(c)


def transform(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def is_rotation(R):
    return np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def same_rotation_quat(q_a, q_b, atol=1e-9):
    """``q`` and ``-q`` describe the same rotation."""
    q_a = np.asarray(q_a, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    return np.allclose(q_a, q_b, atol=atol) or np.allclose(q_a, -q_b, atol=atol)


def axis_angle_quat(letter, angle):
    q = np.zeros(4)
    q[0] = np.cos(angle / 2.0)
    q[1:] = np.sin(angle / 2.0) * np.asarray(AXIS[letter])
    return q


class Chain:
    """Serial chain whose joints rotate about coordinate axes."""

    def __init__(self, letters, offsets, tip_offset):
        self.letters = letters
        self.origins = np.array([transform(p=o) for o in offsets])
        self.axes = np.array([AXIS[c] for c in letters])
        self.tip_offset = np.asarray(tip_offset, dtype=np.float64)

    def frames(self, q):
        out = []
        T = np.eye(4)
        for i, letter in enumerate(self.letters):
            T = T @ self.origins[i] @ transform(R=ROT[letter](q[i]))
            out.append(T)
        return np.array(out)

    def tip(self, q):
        T = self.frames(q)[-1]
        return T[:3, :3] @ self.tip_offset + T[:3, 3]

    def numeric_position_jacobian(self, q, h=1e-6):
        q = np.asarray(q, dtype=np.float64)
        J = np.zeros((3, len(q)))
        for i in range(len(q)):
            d = np.zeros(len(q))
            d[i] = h
            J[:, i] = (self.tip(q + d) - self.tip(q - d)) / (2.0 * h)
        return J


def planar_finger():
    """Three parallel flexion joints, like one finger seen from the side."""
    return Chain("zzz", [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.03, 0.0, 0.0]], [0.02, 0.0, 0.0])


def spatial_chain():
    return Chain(
        "zyxz",
        [[0.0, 0.0, 0.1], [0.2, 0.0, 0.0], [0.0, 0.15, 0.05], [0.1, 0.0, 0.0]],
        [0.03, 0.02, 0.01],
    )
