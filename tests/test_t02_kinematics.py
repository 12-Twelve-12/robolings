import numpy as np
import pytest

import t02_kinematics as m
from helpers import euler_matrix, planar_finger, spatial_chain, transform

RNG = np.random.default_rng(2)


class TestEx06TransformInverse:
    def test_pure_translation(self):
        T = transform(p=[1.0, -2.0, 3.0])
        assert np.allclose(m.transform_inverse(T), transform(p=[-1.0, 2.0, -3.0]))

    def test_random_transforms(self):
        for _ in range(20):
            T = transform(R=euler_matrix(*RNG.uniform(-3, 3, size=3)), p=RNG.normal(size=3))
            inv = m.transform_inverse(T)
            assert inv.shape == (4, 4)
            assert np.allclose(T @ inv, np.eye(4), atol=1e-12)
            assert np.allclose(inv @ T, np.eye(4), atol=1e-12)
            assert np.allclose(inv[3], [0.0, 0.0, 0.0, 1.0])

    def test_does_not_modify_input(self):
        T = transform(R=euler_matrix(0.1, 0.2, 0.3), p=[1.0, 2.0, 3.0])
        before = T.copy()
        m.transform_inverse(T)
        assert np.array_equal(T, before)

    def test_uses_the_closed_form(self, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("use the closed form, not a general matrix inverse")

        monkeypatch.setattr(np.linalg, "inv", forbidden)
        monkeypatch.setattr(np.linalg, "solve", forbidden)
        monkeypatch.setattr(np.linalg, "pinv", forbidden)
        T = transform(R=euler_matrix(0.5, -0.4, 1.1), p=[0.3, 0.2, -0.1])
        inv = m.transform_inverse(T)
        assert np.allclose(inv[:3, :3], T[:3, :3].T)


class TestEx07ForwardKinematics:
    def test_planar_two_link_closed_form(self):
        l1, l2 = 0.4, 0.3
        origins = np.array([transform(), transform(p=[l1, 0.0, 0.0])])
        axes = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        for q1, q2 in [(0.0, 0.0), (0.3, 0.5), (-1.0, 2.0), (np.pi / 2, -np.pi / 2)]:
            frames = m.forward_kinematics(origins, axes, [q1, q2])
            tip = frames[1] @ np.array([l2, 0.0, 0.0, 1.0])
            assert np.allclose(
                tip[:3],
                [l1 * np.cos(q1) + l2 * np.cos(q1 + q2), l1 * np.sin(q1) + l2 * np.sin(q1 + q2), 0.0],
                atol=1e-12,
            )

    def test_zero_angles_gives_product_of_origins(self):
        chain = spatial_chain()
        frames = m.forward_kinematics(chain.origins, chain.axes, np.zeros(4))
        T = np.eye(4)
        for i in range(4):
            T = T @ chain.origins[i]
            assert np.allclose(frames[i], T, atol=1e-12)

    @pytest.mark.parametrize("make", [planar_finger, spatial_chain])
    def test_matches_reference(self, make):
        chain = make()
        n = len(chain.letters)
        for _ in range(20):
            q = RNG.uniform(-np.pi, np.pi, size=n)
            frames = m.forward_kinematics(chain.origins, chain.axes, q)
            assert frames.shape == (n, 4, 4)
            assert np.allclose(frames, chain.frames(q), atol=1e-12)

    def test_origin_with_rotation(self):
        # A joint frame that is rotated relative to its parent link.
        origins = np.array([transform(R=euler_matrix(0.0, 0.0, np.pi / 2), p=[0.0, 0.0, 0.1])])
        axes = np.array([[0.0, 0.0, 1.0]])
        frames = m.forward_kinematics(origins, axes, [np.pi / 2])
        tip = frames[0] @ np.array([1.0, 0.0, 0.0, 1.0])
        assert np.allclose(tip[:3], [0.0, 0.0, 1.1], atol=1e-12)


class TestEx08GeometricJacobian:
    def test_planar_two_link_closed_form(self):
        l1, l2 = 0.4, 0.3
        chain_origins = np.array([transform(), transform(p=[l1, 0.0, 0.0])])
        axes = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        q1, q2 = 0.3, 0.5

        def rz(a):
            return transform(R=euler_matrix(a, 0.0, 0.0))

        f0 = chain_origins[0] @ rz(q1)
        f1 = f0 @ chain_origins[1] @ rz(q2)
        J = m.geometric_jacobian(np.array([f0, f1]), axes, [l2, 0.0, 0.0])
        s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
        assert J.shape == (6, 2)
        assert np.allclose(J[0], [-l1 * s1 - l2 * s12, -l2 * s12], atol=1e-12)
        assert np.allclose(J[1], [l1 * c1 + l2 * c12, l2 * c12], atol=1e-12)
        assert np.allclose(J[2], 0.0, atol=1e-12)
        assert np.allclose(J[3:], [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], atol=1e-12)

    @pytest.mark.parametrize("make", [planar_finger, spatial_chain])
    def test_linear_part_matches_finite_differences(self, make):
        chain = make()
        n = len(chain.letters)
        for _ in range(10):
            q = RNG.uniform(-2.0, 2.0, size=n)
            J = m.geometric_jacobian(chain.frames(q), chain.axes, chain.tip_offset)
            assert J.shape == (6, n)
            assert np.allclose(J[:3], chain.numeric_position_jacobian(q), atol=1e-7)

    def test_angular_part_is_the_world_axis(self):
        chain = spatial_chain()
        q = RNG.uniform(-2.0, 2.0, size=4)
        frames = chain.frames(q)
        J = m.geometric_jacobian(frames, chain.axes, chain.tip_offset)
        for i in range(4):
            assert np.allclose(J[3:, i], frames[i][:3, :3] @ chain.axes[i], atol=1e-12)

    def test_tip_offset_is_in_the_last_link_frame(self):
        chain = spatial_chain()
        q = np.array([0.4, -0.7, 1.1, 0.2])
        frames = chain.frames(q)
        with_offset = m.geometric_jacobian(frames, chain.axes, chain.tip_offset)
        without = m.geometric_jacobian(frames, chain.axes, [0.0, 0.0, 0.0])
        assert not np.allclose(with_offset[:3], without[:3])
        assert np.allclose(with_offset[3:], without[3:])


class TestEx09DlsIkStep:
    def test_no_damping_solves_exactly(self):
        J = RNG.normal(size=(3, 5))
        err = RNG.normal(size=3)
        dq = m.dls_ik_step(J, err, 0.0)
        assert dq.shape == (5,)
        assert np.allclose(J @ dq, err, atol=1e-9)

    def test_matches_the_equivalent_normal_equations(self):
        # (J^T J + l^2 I)^-1 J^T e  is the same vector written the other way.
        for _ in range(10):
            J = RNG.normal(size=(3, 4))
            err = RNG.normal(size=3)
            damping = RNG.uniform(0.01, 0.5)
            expected = np.linalg.solve(J.T @ J + damping**2 * np.eye(4), J.T @ err)
            assert np.allclose(m.dls_ik_step(J, err, damping), expected, atol=1e-9)

    def test_damping_shrinks_the_step(self):
        J = RNG.normal(size=(3, 4))
        err = RNG.normal(size=3)
        small = np.linalg.norm(m.dls_ik_step(J, err, 0.01))
        large = np.linalg.norm(m.dls_ik_step(J, err, 1.0))
        assert large < small

    def test_bounded_at_a_singularity(self):
        # Fully stretched planar arm: it cannot move along x.
        J = np.array([[0.0, 0.0], [0.7, 0.3]])
        dq = m.dls_ik_step(J, np.array([0.1, 0.0]), 0.1)
        assert np.all(np.isfinite(dq))
        assert np.linalg.norm(dq) < 1.0
