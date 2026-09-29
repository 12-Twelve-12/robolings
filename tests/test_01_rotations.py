import numpy as np
import pytest

from helpers import axis_angle_quat, euler_matrix, is_rotation, rot_x, rot_y, rot_z, same_rotation_quat
from loader import load_track

m = load_track("01_rotations")
RNG = np.random.default_rng(1)


class TestRodrigues:
    def test_quarter_turn_about_z(self):
        R = m.rodrigues([0.0, 0.0, 1.0], np.pi / 2)
        assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)

    def test_zero_angle_is_identity(self):
        assert np.allclose(m.rodrigues([0.3, -0.2, 0.9], 0.0), np.eye(3), atol=1e-12)

    def test_axis_is_normalised(self):
        assert np.allclose(m.rodrigues([0.0, 0.0, 5.0], 0.7), rot_z(0.7), atol=1e-12)

    @pytest.mark.parametrize("fn,axis", [(rot_x, [1, 0, 0]), (rot_y, [0, 1, 0]), (rot_z, [0, 0, 1])])
    def test_matches_elementary_rotations(self, fn, axis):
        for angle in (-2.5, -0.3, 0.4, 3.0):
            assert np.allclose(m.rodrigues(axis, angle), fn(angle), atol=1e-12)

    def test_general_axis_properties(self):
        for _ in range(20):
            axis = RNG.normal(size=3)
            angle = RNG.uniform(-np.pi, np.pi)
            R = m.rodrigues(axis, angle)
            k = axis / np.linalg.norm(axis)
            assert R.shape == (3, 3)
            assert is_rotation(R)
            assert np.allclose(R @ k, k, atol=1e-12)
            assert np.isclose(np.trace(R), 1.0 + 2.0 * np.cos(angle), atol=1e-12)


class TestQuatMul:
    def test_identity(self):
        q = np.array([0.5, -0.5, 0.5, 0.5])
        one = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(m.quat_mul(one, q), q)
        assert np.allclose(m.quat_mul(q, one), q)

    def test_basis_elements(self):
        i = [0.0, 1.0, 0.0, 0.0]
        j = [0.0, 0.0, 1.0, 0.0]
        k = [0.0, 0.0, 0.0, 1.0]
        assert np.allclose(m.quat_mul(i, j), k)
        assert np.allclose(m.quat_mul(j, k), i)
        assert np.allclose(m.quat_mul(k, i), j)
        assert np.allclose(m.quat_mul(j, i), [0.0, 0.0, 0.0, -1.0])
        assert np.allclose(m.quat_mul(i, i), [-1.0, 0.0, 0.0, 0.0])

    def test_composes_rotations_in_order(self):
        # z by 90 degrees first, then x by 90 degrees.
        q = m.quat_mul(axis_angle_quat("x", np.pi / 2), axis_angle_quat("z", np.pi / 2))
        # Same rotation as 120 degrees about (1, -1, 1).
        expected = np.array([0.5, 0.5, -0.5, 0.5])
        assert np.allclose(q, expected, atol=1e-12)
        # The other order is a different rotation.
        swapped = m.quat_mul(axis_angle_quat("z", np.pi / 2), axis_angle_quat("x", np.pi / 2))
        assert np.allclose(swapped, [0.5, 0.5, 0.5, 0.5], atol=1e-12)

    def test_norm_and_associativity(self):
        for _ in range(20):
            a, b, c = RNG.normal(size=(3, 4))
            ab = m.quat_mul(a, b)
            assert ab.shape == (4,)
            assert np.isclose(np.linalg.norm(ab), np.linalg.norm(a) * np.linalg.norm(b))
            assert np.allclose(m.quat_mul(ab, c), m.quat_mul(a, m.quat_mul(b, c)))


class TestQuatToMatrix:
    def test_identity(self):
        assert np.allclose(m.quat_to_matrix([1.0, 0.0, 0.0, 0.0]), np.eye(3))

    @pytest.mark.parametrize("letter,fn", [("x", rot_x), ("y", rot_y), ("z", rot_z)])
    def test_matches_elementary_rotations(self, letter, fn):
        for angle in (-2.0, 0.3, 1.2, 3.0):
            assert np.allclose(m.quat_to_matrix(axis_angle_quat(letter, angle)), fn(angle), atol=1e-12)

    def test_input_is_normalised(self):
        q = axis_angle_quat("y", 0.8)
        assert np.allclose(m.quat_to_matrix(3.0 * q), rot_y(0.8), atol=1e-12)

    def test_sign_does_not_matter(self):
        for _ in range(10):
            q = RNG.normal(size=4)
            R = m.quat_to_matrix(q)
            assert R.shape == (3, 3)
            assert is_rotation(R)
            assert np.allclose(m.quat_to_matrix(-q), R, atol=1e-12)


class TestMatrixToQuat:
    def test_identity(self):
        assert np.allclose(m.matrix_to_quat(np.eye(3)), [1.0, 0.0, 0.0, 0.0])

    @pytest.mark.parametrize("letter,fn", [("x", rot_x), ("y", rot_y), ("z", rot_z)])
    def test_elementary_rotations(self, letter, fn):
        for angle in (0.3, 1.2, 2.9):
            assert np.allclose(m.matrix_to_quat(fn(angle)), axis_angle_quat(letter, angle), atol=1e-12)

    @pytest.mark.parametrize("letter,fn", [("x", rot_x), ("y", rot_y), ("z", rot_z)])
    def test_half_turn(self, letter, fn):
        # w is zero here, the case where dividing by w fails.
        q = m.matrix_to_quat(fn(np.pi))
        assert np.all(np.isfinite(q))
        assert same_rotation_quat(q, axis_angle_quat(letter, np.pi), atol=1e-9)

    @pytest.mark.parametrize("letter,fn", [("x", rot_x), ("y", rot_y), ("z", rot_z)])
    def test_just_below_half_turn(self, letter, fn):
        angle = np.pi - 1e-7
        q = m.matrix_to_quat(fn(angle))
        assert np.allclose(q, axis_angle_quat(letter, angle), atol=1e-9)

    def test_w_is_non_negative_and_unit(self):
        for _ in range(50):
            R = euler_matrix(*RNG.uniform(-np.pi, np.pi, size=3))
            q = m.matrix_to_quat(R)
            assert q.shape == (4,)
            assert q[0] >= 0.0
            assert np.isclose(np.linalg.norm(q), 1.0)

    def test_composition_is_preserved(self):
        # Independent of quat_to_matrix: check q(Rz @ Ry) against known factors.
        for _ in range(20):
            a, b = RNG.uniform(-3.0, 3.0, size=2)
            q = m.matrix_to_quat(rot_z(a) @ rot_y(b))
            ca, sa, cb, sb = np.cos(a / 2), np.sin(a / 2), np.cos(b / 2), np.sin(b / 2)
            expected = np.array([ca * cb, -sa * sb, ca * sb, sa * cb])
            assert same_rotation_quat(q, expected, atol=1e-9)


class TestSlerp:
    def test_endpoints(self):
        q0 = axis_angle_quat("x", 0.4)
        q1 = axis_angle_quat("x", 1.9)
        assert same_rotation_quat(m.slerp(q0, q1, 0.0), q0)
        assert same_rotation_quat(m.slerp(q0, q1, 1.0), q1)

    def test_midpoint(self):
        q = m.slerp(axis_angle_quat("z", 0.0), axis_angle_quat("z", np.pi / 2), 0.5)
        assert same_rotation_quat(q, axis_angle_quat("z", np.pi / 4))

    def test_constant_angular_velocity(self):
        q0 = axis_angle_quat("y", 0.2)
        q1 = axis_angle_quat("y", 2.2)
        for t in (0.1, 0.25, 0.6, 0.9):
            assert same_rotation_quat(m.slerp(q0, q1, t), axis_angle_quat("y", 0.2 + 2.0 * t))

    def test_takes_the_short_way(self):
        q0 = axis_angle_quat("z", 0.0)
        q1 = axis_angle_quat("z", 1.0)
        # -q1 is the same orientation; the path must not go the long way round.
        assert same_rotation_quat(m.slerp(q0, -q1, 0.5), axis_angle_quat("z", 0.5))

    def test_identical_inputs(self):
        # The dot product is exactly 1 here, on every platform, so sin(theta) is
        # exactly 0. A robot holding still sends this all the time.
        for q in ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]):
            out = m.slerp(q, q, 0.3)
            assert np.all(np.isfinite(out))
            assert np.allclose(out, q)
            # the same orientation with the opposite sign
            out = m.slerp(q, [-v for v in q], 0.7)
            assert np.all(np.isfinite(out))
            assert same_rotation_quat(out, q)

    def test_nearly_identical_inputs(self):
        q0 = axis_angle_quat("x", 1.0)
        for delta in (0.0, 1e-12, 1e-9, 1e-6):
            q = m.slerp(q0, axis_angle_quat("x", 1.0 + delta), 0.3)
            assert np.all(np.isfinite(q))
            assert np.isclose(np.linalg.norm(q), 1.0)
            assert same_rotation_quat(q, axis_angle_quat("x", 1.0 + 0.3 * delta), atol=1e-9)

    def test_result_is_unit(self):
        for _ in range(20):
            a, b = RNG.normal(size=(2, 4))
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            q = m.slerp(a, b, RNG.uniform())
            assert q.shape == (4,)
            assert np.isclose(np.linalg.norm(q), 1.0)


class TestKabsch:
    def test_recovers_a_known_transform(self):
        for _ in range(10):
            P = RNG.normal(size=(12, 3))
            R_true = euler_matrix(*RNG.uniform(-3.0, 3.0, size=3))
            t_true = RNG.normal(size=3)
            R, t = m.kabsch(P, P @ R_true.T + t_true)
            assert R.shape == (3, 3)
            assert t.shape == (3,)
            assert np.allclose(R, R_true, atol=1e-9)
            assert np.allclose(t, t_true, atol=1e-9)

    def test_pure_translation(self):
        P = RNG.normal(size=(6, 3))
        R, t = m.kabsch(P, P + [0.5, -1.0, 2.0])
        assert np.allclose(R, np.eye(3), atol=1e-9)
        assert np.allclose(t, [0.5, -1.0, 2.0], atol=1e-9)

    def test_points_in_a_plane(self):
        # A calibration board. The third singular value is zero here.
        for _ in range(20):
            P = RNG.normal(size=(9, 3))
            P[:, 2] = 0.0
            R_true = euler_matrix(*RNG.uniform(-3.0, 3.0, size=3))
            t_true = RNG.normal(size=3)
            R, t = m.kabsch(P, P @ R_true.T + t_true)
            assert np.allclose(R, R_true, atol=1e-9)
            assert np.allclose(t, t_true, atol=1e-9)

    def test_never_returns_a_reflection(self):
        # Q is the mirror image of P, so no rotation fits exactly.
        for _ in range(10):
            P = RNG.normal(size=(20, 3))
            R, _ = m.kabsch(P, P * [1.0, 1.0, -1.0])
            assert is_rotation(R)

    def test_is_the_best_rotation(self):
        P = RNG.normal(size=(15, 3))
        Q = P * [1.0, 1.0, -1.0] + [0.2, 0.0, -0.4]
        R, t = m.kabsch(P, Q)

        def residual(R_, t_):
            return np.sum((P @ R_.T + t_ - Q) ** 2)

        best = residual(R, t)
        for _ in range(200):
            other = euler_matrix(*RNG.uniform(-3.0, 3.0, size=3))
            assert best <= residual(other, Q.mean(axis=0) - other @ P.mean(axis=0)) + 1e-9

    def test_with_noise(self):
        P = RNG.normal(size=(50, 3))
        R_true = euler_matrix(0.4, -0.9, 1.3)
        Q = P @ R_true.T + [0.1, 0.2, 0.3] + 1e-3 * RNG.normal(size=(50, 3))
        R, t = m.kabsch(P, Q)
        assert is_rotation(R)
        assert np.allclose(R, R_true, atol=5e-3)
        assert np.allclose(t, [0.1, 0.2, 0.3], atol=5e-3)
