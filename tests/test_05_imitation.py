import numpy as np

from loader import load_track

m = load_track("05_imitation")
RNG = np.random.default_rng(5)  # rebound before every test, see conftest.py


class TestMinMax:
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


class TestActionChunks:
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


class TestTemporalEnsemble:
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


class TestObsHistory:
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


class TestDeltaActions:
    CHUNK = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
    STATE = np.array([1.0, 5.0])

    def test_relative_to_the_state(self):
        delta = m.to_delta(self.CHUNK, self.STATE, [False, False])
        assert delta.shape == (3, 2)
        assert np.allclose(delta, [[0.0, 5.0], [1.0, 15.0], [3.0, 35.0]])

    def test_absolute_dimensions_are_copied(self):
        delta = m.to_delta(self.CHUNK, self.STATE, [False, True])
        assert np.allclose(delta, [[0.0, 10.0], [1.0, 20.0], [3.0, 40.0]])

    def test_from_delta_known_values(self):
        delta = np.array([[0.0, 0.3], [0.5, 0.4]])
        out = m.from_delta(delta, [2.0, 7.0], [False, True])
        assert np.allclose(out, [[2.0, 0.3], [2.5, 0.4]])

    def test_round_trip(self):
        chunk = RNG.normal(size=(16, 7))
        state = RNG.normal(size=7)
        mask = np.array([False, False, True, False, False, False, True])
        delta = m.to_delta(chunk, state, mask)
        assert np.allclose(m.from_delta(delta, state, mask), chunk)
        assert np.allclose(delta[:, mask], chunk[:, mask])
        assert not np.allclose(delta[:, ~mask], chunk[:, ~mask])

    def test_does_not_modify_input(self):
        chunk = self.CHUNK.copy()
        state = self.STATE.copy()
        m.from_delta(m.to_delta(chunk, state, [False, True]), state, [False, True])
        assert np.array_equal(chunk, self.CHUNK)
        assert np.array_equal(state, self.STATE)


class TestDctTokens:
    def test_matrix_is_orthonormal(self):
        for n in (1, 2, 4, 8, 16):
            C = m.dct_matrix(n)
            assert C.shape == (n, n)
            assert np.allclose(C @ C.T, np.eye(n), atol=1e-12)

    def test_first_row_is_constant(self):
        n = 8
        C = m.dct_matrix(n)
        assert np.allclose(C[0], np.sqrt(1.0 / n))

    def test_constant_signal_has_only_a_dc_coefficient(self):
        chunk = np.tile([2.0, -1.0], (16, 1))
        coefficients = m.dct_matrix(16) @ chunk
        assert not np.allclose(coefficients[0], 0.0)
        assert np.allclose(coefficients[1:], 0.0, atol=1e-12)

    def test_transform_preserves_energy(self):
        chunk = RNG.normal(size=(12, 3))
        coefficients = m.dct_matrix(12) @ chunk
        assert np.isclose(np.sum(chunk**2), np.sum(coefficients**2))

    def test_tokens_are_integers(self):
        tokens = m.tokenize(RNG.normal(size=(8, 4)), 0.05)
        assert tokens.shape == (8, 4)
        assert np.issubdtype(np.asarray(tokens).dtype, np.integer)

    def test_rounds_to_nearest_rather_than_towards_zero(self):
        # a constant chunk of 0.8 over 4 steps has a single coefficient of 1.6,
        # which rounds to 2 and truncates to 1
        for sign in (1.0, -1.0):
            tokens = m.tokenize(np.full((4, 1), sign * 0.8), 1.0)
            assert tokens[0, 0] == sign * 2
            assert np.all(tokens[1:] == 0)

    def test_every_coefficient_moves_by_at_most_half_a_step(self):
        step = 0.05
        chunk = RNG.normal(size=(16, 3))
        error = m.dct_matrix(16) @ chunk - m.tokenize(chunk, step) * step
        assert np.abs(error).max() <= step / 2 + 1e-12

    def test_round_trip_is_within_the_quantisation_step(self):
        for step in (0.5, 0.05, 0.005):
            chunk = RNG.normal(size=(16, 5))
            back = m.detokenize(m.tokenize(chunk, step), step)
            assert back.shape == chunk.shape
            # each coefficient moves by at most half a step, and the transform
            # is orthonormal, so the error in the chunk is bounded too
            assert np.abs(back - chunk).max() <= step * np.sqrt(chunk.shape[0])

    def test_a_smaller_step_is_more_accurate(self):
        chunk = np.cumsum(RNG.normal(size=(16, 2)) * 0.1, axis=0)
        coarse = np.abs(m.detokenize(m.tokenize(chunk, 0.5), 0.5) - chunk).max()
        fine = np.abs(m.detokenize(m.tokenize(chunk, 0.001), 0.001) - chunk).max()
        assert fine < coarse

    def test_a_smooth_chunk_is_mostly_zeros(self):
        t = np.linspace(0.0, 1.0, 32)[:, None]
        chunk = np.hstack([t, t**2])
        tokens = m.tokenize(chunk, 0.05)
        assert np.count_nonzero(tokens) < tokens.size // 2

    def test_detokenize_undoes_the_scaling(self):
        tokens = np.array([[2, -1], [0, 3]])
        back = m.detokenize(tokens, 0.25)
        assert np.allclose(back, m.dct_matrix(2).T @ (tokens * 0.25))


class TestEmaWeights:
    def test_decay_warms_up(self):
        assert np.isclose(m.ema_decay(0), 1.0 / 10.0)
        assert np.isclose(m.ema_decay(10), 11.0 / 20.0)
        assert np.isclose(m.ema_decay(90), 91.0 / 100.0)

    def test_decay_is_capped(self):
        assert m.ema_decay(10**9) == 0.9999
        assert m.ema_decay(10**9, max_decay=0.5) == 0.5

    def test_decay_rises_with_the_step(self):
        values = [m.ema_decay(s) for s in range(0, 500, 10)]
        assert all(b >= a for a, b in zip(values, values[1:], strict=False))
        assert type(values[0]) is float

    def test_warmup_is_a_parameter(self):
        assert np.isclose(m.ema_decay(0, warmup=100.0), 1.0 / 100.0)
        assert m.ema_decay(0, warmup=100.0) < m.ema_decay(0, warmup=10.0)

    def test_update_moves_towards_the_weights(self):
        average = {"w": np.zeros(3), "b": np.zeros(2)}
        weights = {"w": np.ones(3), "b": np.full(2, 4.0)}
        out = m.ema_update(average, weights, 0)
        decay = m.ema_decay(0)
        assert np.allclose(out["w"], (1.0 - decay) * np.ones(3))
        assert np.allclose(out["b"], (1.0 - decay) * np.full(2, 4.0))

    def test_update_does_not_modify_its_inputs(self):
        average = {"w": np.zeros(3)}
        weights = {"w": np.ones(3)}
        m.ema_update(average, weights, 5)
        assert np.allclose(average["w"], 0.0)
        assert np.allclose(weights["w"], 1.0)

    def test_update_returns_new_arrays(self):
        average = {"w": np.zeros(3)}
        out = m.ema_update(average, {"w": np.ones(3)}, 5)
        assert out["w"] is not average["w"]
        assert sorted(out) == ["w"]

    def test_constant_weights_are_approached(self):
        average = {"w": np.zeros(2)}
        weights = {"w": np.array([1.0, -2.0])}
        for step in range(3000):
            average = m.ema_update(average, weights, step)
        assert np.allclose(average["w"], weights["w"], atol=1e-2)

    def test_the_ramp_is_the_whole_point(self):
        # the same 50 steps at a constant decay of 0.9999 barely move at all
        average = {"w": np.zeros(1)}
        weights = {"w": np.ones(1)}
        constant = 0.0
        for step in range(50):
            average = m.ema_update(average, weights, step)
            constant = 0.9999 * constant + 0.0001 * 1.0
        assert constant < 0.01
        assert average["w"][0] > 100 * constant
