import numpy as np

import t05_imitation as m

RNG = np.random.default_rng(5)


class TestEx17MinMax:
    LO = np.array([-1.0, 0.0, 10.0])
    HI = np.array([3.0, 0.5, 20.0])

    def test_range_maps_to_minus_one_and_one(self):
        assert np.allclose(m.minmax_normalize(self.LO, self.LO, self.HI), -1.0)
        assert np.allclose(m.minmax_normalize(self.HI, self.LO, self.HI), 1.0)
        assert np.allclose(m.minmax_normalize((self.LO + self.HI) / 2, self.LO, self.HI), 0.0)

    def test_constant_dimension_maps_to_zero(self):
        lo = np.array([0.0, 2.0])
        hi = np.array([1.0, 2.0])
        with np.errstate(all="ignore"):
            y = m.minmax_normalize(np.array([[0.25, 2.0], [0.75, 2.0]]), lo, hi)
        assert np.all(np.isfinite(y))
        assert np.allclose(y, [[-0.5, 0.0], [0.5, 0.0]])

    def test_round_trip(self):
        x = RNG.uniform(self.LO, self.HI, size=(4, 7, 3))
        y = m.minmax_normalize(x, self.LO, self.HI)
        assert y.shape == x.shape
        assert np.all(y >= -1.0 - 1e-12) and np.all(y <= 1.0 + 1e-12)
        assert np.allclose(m.minmax_unnormalize(y, self.LO, self.HI), x)

    def test_unnormalize_known_values(self):
        y = np.array([-1.0, 0.0, 1.0])
        assert np.allclose(m.minmax_unnormalize(y, self.LO, self.HI), [-1.0, 0.25, 20.0])

    def test_unnormalize_constant_dimension_returns_lo(self):
        lo = np.array([0.0, 2.0])
        hi = np.array([1.0, 2.0])
        x = m.minmax_unnormalize(np.array([[0.0, 0.9], [1.0, -0.4]]), lo, hi)
        assert np.allclose(x, [[0.5, 2.0], [1.0, 2.0]])


class TestEx18ActionChunks:
    def test_small_example(self):
        actions = np.arange(10, dtype=np.float64).reshape(5, 2)
        chunks, is_pad = m.make_action_chunks(actions, 3)
        assert chunks.shape == (5, 3, 2)
        assert is_pad.shape == (5, 3)
        assert is_pad.dtype == np.bool_
        assert np.array_equal(chunks[0], actions[0:3])
        assert np.array_equal(chunks[2], actions[2:5])
        assert np.array_equal(chunks[3], [actions[3], actions[4], actions[4]])
        assert np.array_equal(chunks[4], [actions[4], actions[4], actions[4]])

    def test_padding_mask(self):
        _, is_pad = m.make_action_chunks(np.zeros((5, 2)), 3)
        expected = np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, True, True],
            ]
        )
        assert np.array_equal(is_pad, expected)

    def test_horizon_of_one(self):
        actions = RNG.normal(size=(6, 4))
        chunks, is_pad = m.make_action_chunks(actions, 1)
        assert np.array_equal(chunks[:, 0], actions)
        assert not is_pad.any()

    def test_horizon_longer_than_the_episode(self):
        actions = RNG.normal(size=(3, 2))
        chunks, is_pad = m.make_action_chunks(actions, 5)
        assert chunks.shape == (3, 5, 2)
        assert np.array_equal(chunks[0], [actions[0], actions[1], actions[2], actions[2], actions[2]])
        assert np.array_equal(is_pad[0], [False, False, False, True, True])
        assert np.array_equal(is_pad[2], [False, True, True, True, True])


class TestEx19TemporalEnsemble:
    def test_zero_m_is_the_plain_mean(self):
        preds = RNG.normal(size=(6, 3))
        out = m.temporal_ensemble(preds, 0.0)
        assert out.shape == (3,)
        assert np.allclose(out, preds.mean(axis=0))

    def test_single_prediction(self):
        assert np.allclose(m.temporal_ensemble(np.array([[0.3, -0.7]]), 0.5), [0.3, -0.7])

    def test_known_weights(self):
        # weights 1 and 1/2 -> (1 * 0 + 0.5 * 1) / 1.5
        out = m.temporal_ensemble(np.array([[0.0], [1.0]]), np.log(2.0))
        assert np.allclose(out, [1.0 / 3.0])

    def test_oldest_prediction_weighs_most(self):
        out = m.temporal_ensemble(np.array([[0.0], [0.0], [1.0]]), 0.3)
        reverse = m.temporal_ensemble(np.array([[1.0], [0.0], [0.0]]), 0.3)
        assert out[0] < 1.0 / 3.0 < reverse[0]

    def test_constant_predictions_are_unchanged(self):
        preds = np.tile(np.array([0.2, -0.4, 0.9]), (8, 1))
        assert np.allclose(m.temporal_ensemble(preds, 0.01), [0.2, -0.4, 0.9])


class TestEx20ObsHistory:
    def test_small_example(self):
        obs = np.arange(8, dtype=np.float64).reshape(4, 2)
        out = m.stack_obs_history(obs, 3)
        assert out.shape == (4, 3, 2)
        assert np.array_equal(out[0], [obs[0], obs[0], obs[0]])
        assert np.array_equal(out[1], [obs[0], obs[0], obs[1]])
        assert np.array_equal(out[2], [obs[0], obs[1], obs[2]])
        assert np.array_equal(out[3], [obs[1], obs[2], obs[3]])

    def test_newest_observation_is_last(self):
        obs = RNG.normal(size=(10, 3))
        out = m.stack_obs_history(obs, 4)
        assert np.array_equal(out[:, -1], obs)

    def test_window_of_one(self):
        obs = RNG.normal(size=(5, 2))
        assert np.array_equal(m.stack_obs_history(obs, 1), obs[:, None, :])

    def test_window_longer_than_the_episode(self):
        obs = RNG.normal(size=(2, 2))
        out = m.stack_obs_history(obs, 4)
        assert out.shape == (2, 4, 2)
        assert np.array_equal(out[1], [obs[0], obs[0], obs[0], obs[1]])
