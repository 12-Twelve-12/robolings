import numpy as np

from loader import load_track

m = load_track("04_control")
RNG = np.random.default_rng(4)


class TestMinJerk:
    def test_starts_and_ends_at_rest(self):
        q0 = np.array([0.1, -0.4])
        q1 = np.array([1.0, 0.6])
        pos, vel, acc = m.min_jerk(q0, q1, 2.0, 0.0)
        assert np.allclose(pos, q0) and np.allclose(vel, 0.0) and np.allclose(acc, 0.0)
        pos, vel, acc = m.min_jerk(q0, q1, 2.0, 2.0)
        assert np.allclose(pos, q1) and np.allclose(vel, 0.0) and np.allclose(acc, 0.0)

    def test_midpoint(self):
        pos, vel, acc = m.min_jerk(np.array([0.0]), np.array([2.0]), 4.0, 2.0)
        assert np.allclose(pos, 1.0)
        assert np.allclose(vel, 1.875 * 2.0 / 4.0)
        assert np.allclose(acc, 0.0)

    def test_velocity_and_acceleration_are_time_derivatives(self):
        q0 = np.array([0.3, -1.0, 0.0])
        q1 = np.array([-0.5, 2.0, 0.7])
        duration = 1.7
        h = 1e-5
        for t in (0.2, 0.61, 1.0, 1.55):
            _, vel, acc = m.min_jerk(q0, q1, duration, t)
            p_hi, v_hi, _ = m.min_jerk(q0, q1, duration, t + h)
            p_lo, v_lo, _ = m.min_jerk(q0, q1, duration, t - h)
            assert np.allclose(vel, (p_hi - p_lo) / (2 * h), atol=1e-7)
            assert np.allclose(acc, (v_hi - v_lo) / (2 * h), atol=1e-7)

    def test_holds_outside_the_interval(self):
        q0 = np.array([0.0])
        q1 = np.array([1.0])
        for t, expected in ((-0.5, q0), (3.0, q1)):
            pos, vel, acc = m.min_jerk(q0, q1, 1.0, t)
            assert np.allclose(pos, expected)
            assert np.allclose(vel, 0.0)
            assert np.allclose(acc, 0.0)

    def test_peak_velocity(self):
        # Useful on hardware: v_max = 1.875 * distance / duration.
        ts = np.linspace(0.0, 3.0, 3001)
        speeds = [abs(m.min_jerk(np.array([0.0]), np.array([0.6]), 3.0, t)[1][0]) for t in ts]
        assert np.isclose(max(speeds), 1.875 * 0.6 / 3.0, atol=1e-6)


class TestLowpassFilter:
    def test_constant_input_is_unchanged(self):
        x = np.full(50, 0.7)
        assert np.allclose(m.lowpass_filter(x, 5.0, 0.01), 0.7)

    def test_step_response(self):
        cutoff, dt = 10.0, 0.002
        alpha = dt / (1.0 / (2.0 * np.pi * cutoff) + dt)
        y = m.lowpass_filter(np.array([0.0, 1.0, 1.0, 1.0]), cutoff, dt)
        assert np.allclose(y, [0.0, alpha, alpha + alpha * (1 - alpha), 1 - (1 - alpha) ** 3])

    def test_known_alpha(self):
        # rc == dt gives alpha = 0.5
        dt = 0.01
        cutoff = 1.0 / (2.0 * np.pi * dt)
        y = m.lowpass_filter(np.array([0.0, 8.0, 8.0]), cutoff, dt)
        assert np.allclose(y, [0.0, 4.0, 6.0])

    def test_each_dimension_is_independent(self):
        x = RNG.normal(size=(40, 3))
        y = m.lowpass_filter(x, 3.0, 0.01)
        assert y.shape == (40, 3)
        for d in range(3):
            assert np.allclose(y[:, d], m.lowpass_filter(x[:, d], 3.0, 0.01))

    def test_does_not_modify_input(self):
        x = RNG.normal(size=20)
        before = x.copy()
        m.lowpass_filter(x, 3.0, 0.01)
        assert np.array_equal(x, before)


class TestMitTorque:
    def test_each_term(self):
        assert np.isclose(m.mit_torque(20.0, 0.0, 1.0, 0.0, 0.9, 0.0, 0.0, 100.0), 2.0)
        assert np.isclose(m.mit_torque(0.0, 2.0, 0.0, 0.5, 0.0, 1.5, 0.0, 100.0), -2.0)
        assert np.isclose(m.mit_torque(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 100.0), 0.3)

    def test_limit_applies_to_the_total(self):
        # P term is +10, feedforward is -8. The total, 2, is inside the limit.
        tau = m.mit_torque(10.0, 0.0, 1.0, 0.0, 0.0, 0.0, -8.0, 5.0)
        assert np.isclose(tau, 2.0)

    def test_limit_is_symmetric(self):
        assert np.isclose(m.mit_torque(100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0), 3.0)
        assert np.isclose(m.mit_torque(100.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 3.0), -3.0)

    def test_per_joint_arrays(self):
        kp = np.array([10.0, 20.0, 5.0])
        kd = np.array([1.0, 0.5, 0.2])
        q_des = np.array([0.5, -0.2, 1.0])
        q = np.array([0.4, 0.0, 0.0])
        dq = np.array([0.1, -0.2, 0.5])
        limit = np.array([10.0, 2.0, 4.0])
        tau = m.mit_torque(kp, kd, q_des, 0.0, q, dq, 0.0, limit)
        assert tau.shape == (3,)
        assert np.allclose(tau, [0.9, -2.0, 4.0])


class TestRateLimit:
    def test_small_change_passes_through(self):
        assert np.allclose(m.rate_limit([0.105], [0.1], 1.0, 0.01), [0.105])

    def test_large_change_is_limited(self):
        assert np.allclose(m.rate_limit([1.0], [0.1], 2.0, 0.01), [0.12])
        assert np.allclose(m.rate_limit([-1.0], [0.1], 2.0, 0.01), [0.08])

    def test_each_joint_is_limited_separately(self):
        out = m.rate_limit([1.0, 0.101, -1.0], [0.0, 0.1, 0.0], 1.0, 0.01)
        assert np.allclose(out, [0.01, 0.101, -0.01])

    def test_per_joint_rates(self):
        out = m.rate_limit([1.0, 1.0], [0.0, 0.0], np.array([1.0, 5.0]), 0.1)
        assert np.allclose(out, [0.1, 0.5])

    def test_converges_to_the_target(self):
        cmd = np.array([0.0, 0.0])
        target = np.array([0.35, -0.2])
        for _ in range(40):
            cmd = m.rate_limit(target, cmd, 1.0, 0.01)
        assert np.allclose(cmd, target)
