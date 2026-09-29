import math

import numpy as np

import t06_diffusion as m

RNG = np.random.default_rng(6)


def scalar_alpha_bar(u, s):
    return math.cos((u + s) / (1.0 + s) * math.pi / 2.0) ** 2


class TestEx21CosineSchedule:
    def test_shapes_and_ranges(self):
        betas, acp = m.cosine_schedule(100)
        assert betas.shape == (100,)
        assert acp.shape == (100,)
        assert np.all(betas > 0.0) and np.all(betas <= 0.999)
        assert np.all(acp > 0.0) and np.all(acp < 1.0)
        assert np.all(np.diff(acp) < 0.0)

    def test_values_against_a_scalar_loop(self):
        n, s = 50, 0.008
        betas, acp = m.cosine_schedule(n, s=s)
        running = 1.0
        for i in range(n):
            beta = min(1.0 - scalar_alpha_bar((i + 1) / n, s) / scalar_alpha_bar(i / n, s), 0.999)
            running *= 1.0 - beta
            assert math.isclose(betas[i], beta, rel_tol=1e-12, abs_tol=1e-15)
            assert math.isclose(acp[i], running, rel_tol=1e-9, abs_tol=1e-300)

    def test_last_beta_is_capped(self):
        betas, _ = m.cosine_schedule(100)
        assert betas[-1] == 0.999
        betas, _ = m.cosine_schedule(100, max_beta=0.5)
        assert betas.max() == 0.5

    def test_first_step_adds_almost_no_noise(self):
        _, acp = m.cosine_schedule(1000)
        assert acp[0] > 0.9999

    def test_offset_s_is_used(self):
        a, _ = m.cosine_schedule(100, s=0.008)
        b, _ = m.cosine_schedule(100, s=0.1)
        assert not np.allclose(a, b)


class TestEx22QSample:
    ACP = np.array([1.0, 0.64, 0.25])

    def test_known_values(self):
        x0 = np.array([[1.0, 2.0]])
        noise = np.array([[10.0, -10.0]])
        assert np.allclose(m.q_sample(x0, [1], noise, self.ACP), [[0.8 + 6.0, 1.6 - 6.0]])
        assert np.allclose(m.q_sample(x0, [2], noise, self.ACP), [[0.5 + 75.0**0.5, 1.0 - 75.0**0.5]])

    def test_no_noise_when_alpha_bar_is_one(self):
        x0 = RNG.normal(size=(1, 4, 3))
        assert np.allclose(m.q_sample(x0, [0], RNG.normal(size=(1, 4, 3)), self.ACP), x0)

    def test_one_timestep_per_sample(self):
        x0 = RNG.normal(size=(3, 8, 2))
        noise = RNG.normal(size=(3, 8, 2))
        t = np.array([2, 0, 1])
        out = m.q_sample(x0, t, noise, self.ACP)
        assert out.shape == (3, 8, 2)
        for b in range(3):
            a = self.ACP[t[b]]
            assert np.allclose(out[b], math.sqrt(a) * x0[b] + math.sqrt(1 - a) * noise[b])

    def test_batch_of_vectors(self):
        x0 = RNG.normal(size=(4, 5))
        noise = RNG.normal(size=(4, 5))
        out = m.q_sample(x0, [1, 1, 2, 0], noise, self.ACP)
        assert out.shape == (4, 5)
        assert np.allclose(out[2], 0.5 * x0[2] + math.sqrt(0.75) * noise[2])


class TestEx23DdimStep:
    ACP = np.array([0.9, 0.64, 0.25, 0.04])

    def noised(self, x0, eps, t):
        return math.sqrt(self.ACP[t]) * x0 + math.sqrt(1.0 - self.ACP[t]) * eps

    def test_with_the_true_noise_it_lands_on_the_same_trajectory(self):
        x0 = RNG.normal(size=(2, 6, 3))
        eps = RNG.normal(size=(2, 6, 3))
        out = m.ddim_step(self.noised(x0, eps, 3), eps, 3, 1, self.ACP)
        assert out.shape == x0.shape
        assert np.allclose(out, self.noised(x0, eps, 1))

    def test_final_step_returns_the_clean_sample(self):
        x0 = RNG.normal(size=(2, 6, 3))
        eps = RNG.normal(size=(2, 6, 3))
        out = m.ddim_step(self.noised(x0, eps, 0), eps, 0, -1, self.ACP)
        assert np.allclose(out, x0)

    def test_skipping_steps(self):
        # 4 training steps, 2 inference steps.
        x0 = RNG.normal(size=(1, 4, 2))
        eps = RNG.normal(size=(1, 4, 2))
        x = self.noised(x0, eps, 3)
        x = m.ddim_step(x, eps, 3, 1, self.ACP)
        x = m.ddim_step(x, eps, 1, -1, self.ACP)
        assert np.allclose(x, x0)

    def test_known_values(self):
        # a_t = 0.25, a_prev = 0.64, x_t = 1, eps = 0
        # x0_pred = 1 / 0.5 = 2, x_prev = 0.8 * 2
        assert np.allclose(m.ddim_step(np.array([1.0]), np.array([0.0]), 2, 1, self.ACP), [1.6])
        # eps = 1: x0_pred = (1 - sqrt(0.75)) / 0.5, x_prev = 0.8 * x0_pred + 0.6
        expected = 0.8 * (1.0 - math.sqrt(0.75)) / 0.5 + 0.6
        assert np.allclose(m.ddim_step(np.array([1.0]), np.array([1.0]), 2, 1, self.ACP), [expected])
