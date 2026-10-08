import numpy as np

from loader import load_track

m = load_track("04_control")
RNG = np.random.default_rng(4)  # rebound before every test, see conftest.py


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


class TestAngleDiff:
    def test_wrap_leaves_small_angles_alone(self):
        a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert np.allclose(m.wrap_to_pi(a), a)

    def test_wrap_known_values(self):
        assert np.isclose(m.wrap_to_pi(1.5 * np.pi), -0.5 * np.pi)
        assert np.isclose(m.wrap_to_pi(-1.5 * np.pi), 0.5 * np.pi)
        assert np.isclose(m.wrap_to_pi(2.0 * np.pi), 0.0)
        assert np.isclose(m.wrap_to_pi(7.0 * np.pi + 0.1), -np.pi + 0.1)
        assert np.isclose(m.wrap_to_pi(-40.0 * np.pi - 0.2), -0.2)

    def test_wrap_half_turn_may_have_either_sign(self):
        for a in (np.pi, -np.pi, 3.0 * np.pi):
            assert np.isclose(abs(m.wrap_to_pi(a)), np.pi)

    def test_wrap_is_the_same_angle(self):
        a = RNG.uniform(-50.0, 50.0, size=(3, 40))
        w = m.wrap_to_pi(a)
        assert w.shape == a.shape
        assert np.all(w >= -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)
        assert np.allclose(np.sin(w), np.sin(a))
        assert np.allclose(np.cos(w), np.cos(a))

    def test_diff_without_wrapping(self):
        assert np.isclose(m.angle_diff(0.3, 0.1), 0.2)
        assert np.isclose(m.angle_diff(0.1, 0.3), -0.2)

    def test_diff_across_the_seam(self):
        a, b = np.deg2rad(179.0), np.deg2rad(-179.0)
        assert np.isclose(m.angle_diff(a, b), np.deg2rad(-2.0))
        assert np.isclose(m.angle_diff(b, a), np.deg2rad(2.0))

    def test_diff_with_many_turns(self):
        assert np.isclose(m.angle_diff(0.1 + 4.0 * np.pi, 0.0), 0.1)
        assert np.isclose(m.angle_diff(0.1, 6.0 * np.pi), 0.1)
        assert np.isclose(m.angle_diff(3.0 + 2.0 * np.pi, -3.0), 6.0 - 2.0 * np.pi)

    def test_diff_adds_back_to_the_same_angle(self):
        a = RNG.uniform(-20.0, 20.0, size=100)
        b = RNG.uniform(-20.0, 20.0, size=100)
        d = m.angle_diff(a, b)
        assert np.all(np.abs(d) <= np.pi + 1e-12)
        assert np.allclose(np.sin(b + d), np.sin(a))
        assert np.allclose(np.cos(b + d), np.cos(a))

    def test_diff_broadcasts(self):
        d = m.angle_diff(np.zeros((3, 1)), np.zeros(4))
        assert d.shape == (3, 4)


class TestPidStep:
    def test_proportional(self):
        u, state = m.pid_step((0.0, None), 0.5, 0.01, 4.0, 0.0, 0.0, 10.0)
        assert np.isclose(u, 2.0)
        assert np.isclose(state[0], 0.005)
        assert state[1] == 0.5

    def test_integral_accumulates(self):
        u, state = m.pid_step((0.0, None), 1.0, 0.1, 0.0, 2.0, 0.0, 10.0)
        assert np.isclose(u, 0.2)
        u, state = m.pid_step(state, 1.0, 0.1, 0.0, 2.0, 0.0, 10.0)
        assert np.isclose(u, 0.4)
        assert np.isclose(state[0], 0.2)

    def test_derivative(self):
        _, state = m.pid_step((0.0, None), 1.0, 0.1, 0.0, 0.0, 0.5, 10.0)
        u, _ = m.pid_step(state, 1.2, 0.1, 0.0, 0.0, 0.5, 10.0)
        assert np.isclose(u, 1.0)
        u, _ = m.pid_step(state, 0.8, 0.1, 0.0, 0.0, 0.5, 10.0)
        assert np.isclose(u, -1.0)

    def test_no_derivative_kick_on_the_first_call(self):
        u, _ = m.pid_step((0.0, None), 1.0, 0.001, 2.0, 0.0, 5.0, 1e9)
        assert np.isclose(u, 2.0)

    def test_output_is_clipped(self):
        assert m.pid_step((0.0, None), 1.0, 0.01, 100.0, 0.0, 0.0, 5.0)[0] == 5.0
        assert m.pid_step((0.0, None), -1.0, 0.01, 100.0, 0.0, 0.0, 5.0)[0] == -5.0

    def test_integral_does_not_grow_while_saturated(self):
        state = (0.0, None)
        for _ in range(100):
            u, state = m.pid_step(state, 5.0, 0.1, 1.0, 1.0, 0.0, 2.0)
            assert u == 2.0
        assert np.isclose(state[0], 0.0)

    def test_leaves_saturation_as_soon_as_the_error_changes_sign(self):
        state = (0.0, None)
        for _ in range(100):
            _, state = m.pid_step(state, 5.0, 0.1, 1.0, 1.0, 0.0, 2.0)
        u, _ = m.pid_step(state, -0.5, 0.1, 1.0, 1.0, 0.0, 2.0)
        assert np.isclose(u, -0.55)

    def test_integral_may_shrink_while_saturated(self):
        # Saturated at the upper limit, but the error already points the other way.
        u, state = m.pid_step((10.0, -1.0), -1.0, 0.1, 0.0, 1.0, 0.0, 2.0)
        assert u == 2.0
        assert np.isclose(state[0], 9.9)

    def test_integrates_normally_below_the_limit(self):
        state = (0.0, None)
        for _ in range(10):
            u, state = m.pid_step(state, 0.5, 0.1, 1.0, 1.0, 0.0, 2.0)
        assert np.isclose(state[0], 0.5)
        assert np.isclose(u, 1.0)


class TestTrapezoid:
    def sample(self, distance, v_max, a_max, n=2001, span=None):
        total = span if span is not None else 10.0
        ts = np.linspace(-0.5, total, n)
        out = np.array([m.trapezoid(distance, v_max, a_max, float(t)) for t in ts])
        return ts, out[:, 0], out[:, 1]

    def test_returns_floats(self):
        pos, vel = m.trapezoid(1.0, 1.0, 1.0, 0.5)
        assert type(pos) is float and type(vel) is float

    def test_before_and_after_the_move(self):
        assert m.trapezoid(2.0, 1.0, 1.0, -1.0) == (0.0, 0.0)
        assert m.trapezoid(2.0, 1.0, 1.0, 0.0) == (0.0, 0.0)
        pos, vel = m.trapezoid(2.0, 1.0, 1.0, 100.0)
        assert np.isclose(pos, 2.0) and vel == 0.0

    def test_long_move_reaches_the_speed_limit(self):
        # ramps take 1 s and 0.5 m each, so 1 m of cruise at 1 m/s
        for t, expected in ((0.5, (0.125, 0.5)), (1.0, (0.5, 1.0)), (1.5, (1.0, 1.0)), (2.0, (1.5, 1.0))):
            assert np.allclose(m.trapezoid(2.0, 1.0, 1.0, t), expected)
        assert np.allclose(m.trapezoid(2.0, 1.0, 1.0, 2.5), (1.875, 0.5))
        assert np.allclose(m.trapezoid(2.0, 1.0, 1.0, 3.0), (2.0, 0.0))

    def test_short_move_never_reaches_the_speed_limit(self):
        # 0.25 m at a_max 1 peaks at sqrt(1 * 0.25) = 0.5 m/s, well under v_max,
        # halfway through a move that lasts 2 * 0.5 / 1 = 1 s
        assert np.allclose(m.trapezoid(0.25, 10.0, 1.0, 0.5), (0.125, 0.5))
        _, pos, vel = self.sample(0.25, 10.0, 1.0)
        assert vel.max() <= 0.5 + 1e-12
        assert vel.max() > 0.49
        assert pos.max() <= 0.25 + 1e-12
        assert np.isclose(pos[-1], 0.25)

    def test_never_overshoots(self):
        for distance, v_max, a_max in ((0.05, 2.0, 1.0), (1.0, 0.5, 4.0), (3.0, 1.5, 0.5), (0.2, 0.3, 10.0)):
            _, pos, vel = self.sample(distance, v_max, a_max)
            assert pos.max() <= distance + 1e-9
            assert vel.max() <= v_max + 1e-9
            assert np.all(vel >= -1e-12)
            assert np.isclose(pos[-1], distance)

    def test_velocity_is_the_derivative_of_position(self):
        h = 1e-6
        for distance, v_max, a_max in ((2.0, 1.0, 1.0), (0.25, 10.0, 1.0)):
            for t in (0.2, 0.9, 1.4, 2.2):
                _, vel = m.trapezoid(distance, v_max, a_max, t)
                hi, _ = m.trapezoid(distance, v_max, a_max, t + h)
                lo, _ = m.trapezoid(distance, v_max, a_max, t - h)
                assert np.isclose(vel, (hi - lo) / (2 * h), atol=1e-4)

    def test_acceleration_never_exceeds_the_limit(self):
        for distance, v_max, a_max in ((2.0, 1.0, 1.0), (0.25, 10.0, 1.0), (1.0, 0.5, 4.0)):
            ts, _, vel = self.sample(distance, v_max, a_max, n=4001)
            acc = np.diff(vel) / np.diff(ts)
            assert np.abs(acc).max() <= a_max + 1e-3

    def test_zero_distance(self):
        assert m.trapezoid(0.0, 1.0, 1.0, 0.5) == (0.0, 0.0)


class TestGravityTorque:
    LENGTHS = np.array([0.4, 0.3])
    MASSES = np.array([2.0, 1.0])
    COM = np.array([0.2, 0.15])

    def energy(self, q, lengths, masses, com, g=9.81):
        absolute = np.cumsum(np.asarray(q, dtype=np.float64))
        total = 0.0
        for i in range(len(q)):
            height = com[i] * np.sin(absolute[i])
            for j in range(i):
                height += lengths[j] * np.sin(absolute[j])
            total += masses[i] * height
        return g * total

    def numeric(self, q, lengths, masses, com, g=9.81, h=1e-6):
        q = np.asarray(q, dtype=np.float64)
        out = np.zeros(len(q))
        for k in range(len(q)):
            d = np.zeros(len(q))
            d[k] = h
            out[k] = (
                self.energy(q + d, lengths, masses, com, g) - self.energy(q - d, lengths, masses, com, g)
            ) / (2 * h)
        return out

    def test_hanging_straight_down_needs_no_torque(self):
        out = m.gravity_torque([-np.pi / 2, 0.0], self.LENGTHS, self.MASSES, self.COM)
        assert out.shape == (2,)
        assert np.allclose(out, 0.0, atol=1e-9)

    def test_single_link_held_horizontally(self):
        # one link, mass 2 kg, centre of mass 0.2 m out
        out = m.gravity_torque([0.0], [0.4], [2.0], [0.2])
        assert np.isclose(out[0], 9.81 * 2.0 * 0.2)

    def test_single_link_at_an_angle(self):
        for angle in (0.3, 1.0, -0.7, 2.5):
            out = m.gravity_torque([angle], [0.4], [2.0], [0.2])
            assert np.isclose(out[0], 9.81 * 2.0 * 0.2 * np.cos(angle))

    def test_two_links_stretched_out(self):
        out = m.gravity_torque([0.0, 0.0], self.LENGTHS, self.MASSES, self.COM)
        # shoulder carries link 0 at 0.2 m and link 1 at 0.4 + 0.15 m
        assert np.isclose(out[0], 9.81 * (2.0 * 0.2 + 1.0 * 0.55))
        assert np.isclose(out[1], 9.81 * 1.0 * 0.15)

    def test_matches_the_gradient_of_the_potential_energy(self):
        for n in (1, 2, 3, 4):
            lengths = RNG.uniform(0.1, 0.5, size=n)
            masses = RNG.uniform(0.2, 3.0, size=n)
            com = lengths * RNG.uniform(0.2, 0.8, size=n)
            for _ in range(5):
                q = RNG.uniform(-np.pi, np.pi, size=n)
                out = m.gravity_torque(q, lengths, masses, com)
                assert out.shape == (n,)
                assert np.allclose(out, self.numeric(q, lengths, masses, com), atol=1e-5)

    def test_gravity_is_a_parameter(self):
        q = [0.4, -0.2]
        earth = m.gravity_torque(q, self.LENGTHS, self.MASSES, self.COM)
        moon = m.gravity_torque(q, self.LENGTHS, self.MASSES, self.COM, g=1.62)
        assert np.allclose(moon, earth * 1.62 / 9.81)

    def test_the_last_joint_only_carries_the_last_link(self):
        q = RNG.uniform(-2.0, 2.0, size=3)
        lengths = np.array([0.3, 0.2, 0.25])
        masses = np.array([1.0, 2.0, 0.5])
        com = np.array([0.15, 0.1, 0.1])
        out = m.gravity_torque(q, lengths, masses, com)
        assert np.isclose(out[-1], 9.81 * masses[-1] * com[-1] * np.cos(q.sum()))
