import numpy as np

from loader import load_track

m = load_track("07_flow_matching")
RNG = np.random.default_rng(7)


class TestFlowMatchingTarget:
    def test_endpoints(self):
        noise = RNG.normal(size=(2, 5, 3))
        data = RNG.normal(size=(2, 5, 3))
        x_t, v = m.flow_matching_target(noise, data, np.array([0.0, 1.0]))
        assert np.allclose(x_t[0], noise[0])
        assert np.allclose(x_t[1], data[1])
        assert np.allclose(v, data - noise)

    def test_midpoint(self):
        noise = np.array([[0.0, 4.0]])
        data = np.array([[2.0, 0.0]])
        x_t, v = m.flow_matching_target(noise, data, np.array([0.25]))
        assert np.allclose(x_t, [[0.5, 3.0]])
        assert np.allclose(v, [[2.0, -4.0]])

    def test_one_time_per_sample(self):
        noise = RNG.normal(size=(4, 8, 2))
        data = RNG.normal(size=(4, 8, 2))
        t = np.array([0.1, 0.9, 0.5, 0.33])
        x_t, v = m.flow_matching_target(noise, data, t)
        assert x_t.shape == (4, 8, 2)
        assert v.shape == (4, 8, 2)
        for b in range(4):
            assert np.allclose(x_t[b], (1 - t[b]) * noise[b] + t[b] * data[b])

    def test_velocity_is_the_time_derivative_of_the_path(self):
        noise = RNG.normal(size=(3, 6))
        data = RNG.normal(size=(3, 6))
        t = np.array([0.2, 0.5, 0.7])
        h = 1e-6
        hi, _ = m.flow_matching_target(noise, data, t + h)
        lo, _ = m.flow_matching_target(noise, data, t - h)
        _, v = m.flow_matching_target(noise, data, t)
        assert np.allclose(v, (hi - lo) / (2 * h), atol=1e-8)


class TestEulerSample:
    def test_constant_field(self):
        x = RNG.normal(size=(2, 4, 3))
        c = RNG.normal(size=(2, 4, 3))
        out = m.euler_sample(lambda x_, t_: c, x, 7)
        assert out.shape == x.shape
        assert np.allclose(out, x + c)

    def test_times_passed_to_the_network(self):
        seen = []

        def v_fn(x_, t_):
            seen.append(float(t_))
            return np.zeros_like(x_)

        m.euler_sample(v_fn, np.zeros(3), 4)
        assert np.allclose(seen, [0.0, 0.25, 0.5, 0.75])

    def test_straight_path_reaches_the_data(self):
        target = RNG.normal(size=(2, 5))
        noise = RNG.normal(size=(2, 5))

        # The exact field of a straight line from noise to target.
        def v_fn(x_, t_):
            return (target - x_) / (1.0 - t_)

        for steps in (1, 3, 10):
            out = m.euler_sample(v_fn, noise, steps)
            assert np.all(np.isfinite(out))
            assert np.allclose(out, target)

    def test_known_value(self):
        # dx/dt = x with four steps: x * 1.25 ** 4
        out = m.euler_sample(lambda x_, t_: x_, np.array([1.0, -2.0]), 4)
        assert np.allclose(out, np.array([1.0, -2.0]) * 1.25**4)

    def test_does_not_modify_input(self):
        x = RNG.normal(size=(3, 2))
        before = x.copy()
        m.euler_sample(lambda x_, t_: np.ones_like(x_), x, 5)
        assert np.array_equal(x, before)
