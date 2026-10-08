"""Wrong answers for track 05, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
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
]
