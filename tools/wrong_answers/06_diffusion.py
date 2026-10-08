"""Wrong answers for track 06, see tools/mutants.py."""

# (exercise, function, what is wrong, replacement source)
MUTANTS = [
    (
        "cosine_schedule",
        "cosine_schedule",
        "beta is not capped",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    def f(u):
        return np.cos((u + s) / (1.0 + s) * np.pi / 2.0) ** 2
    i = np.arange(num_steps, dtype=np.float64)
    betas = 1.0 - f((i + 1.0) / num_steps) / f(i / num_steps)
    return betas, np.cumprod(1.0 - betas)
""",
    ),
    (
        "cosine_schedule",
        "cosine_schedule",
        "offset s ignored",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    def f(u):
        return np.cos(u * np.pi / 2.0) ** 2
    i = np.arange(num_steps, dtype=np.float64)
    betas = np.minimum(1.0 - f((i + 1.0) / num_steps) / f(i / num_steps), max_beta)
    return betas, np.cumprod(1.0 - betas)
""",
    ),
    (
        "cosine_schedule",
        "cosine_schedule",
        "linear schedule",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    betas = np.linspace(1e-4, 0.02, num_steps)
    return betas, np.cumprod(1.0 - betas)
""",
    ),
    (
        "cosine_schedule",
        "cosine_schedule",
        "returns alpha_bar of the cosine directly",
        """
def cosine_schedule(num_steps, s=0.008, max_beta=0.999):
    def f(u):
        return np.cos((u + s) / (1.0 + s) * np.pi / 2.0) ** 2
    i = np.arange(num_steps, dtype=np.float64)
    betas = np.minimum(1.0 - f((i + 1.0) / num_steps) / f(i / num_steps), max_beta)
    return betas, f((i + 1.0) / num_steps) / f(0.0)
""",
    ),
    (
        "q_sample",
        "q_sample",
        "square roots missing",
        """
def q_sample(x0, t, noise, alphas_cumprod):
    x0 = np.asarray(x0, dtype=np.float64)
    a = np.asarray(alphas_cumprod)[np.asarray(t, dtype=int)].reshape((-1,) + (1,) * (x0.ndim - 1))
    return a * x0 + (1.0 - a) * np.asarray(noise, dtype=np.float64)
""",
    ),
    (
        "q_sample",
        "q_sample",
        "first timestep used for the whole batch",
        """
def q_sample(x0, t, noise, alphas_cumprod):
    a = np.asarray(alphas_cumprod)[np.asarray(t, dtype=int)][0]
    return np.sqrt(a) * np.asarray(x0, dtype=np.float64) + np.sqrt(1.0 - a) * np.asarray(noise, dtype=np.float64)
""",
    ),
    (
        "ddim_step",
        "ddim_step",
        "t_prev = -1 wraps to the last timestep",
        """
def ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    a_t = alphas_cumprod[t]
    a_prev = alphas_cumprod[t_prev]
    x0_pred = (x_t - np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
    return np.sqrt(a_prev) * x0_pred + np.sqrt(1.0 - a_prev) * eps_pred
""",
    ),
    (
        "ddim_step",
        "ddim_step",
        "always steps to t - 1",
        """
def ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    a_t = alphas_cumprod[t]
    a_prev = alphas_cumprod[t - 1] if t > 0 else 1.0
    x0_pred = (x_t - np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
    return np.sqrt(a_prev) * x0_pred + np.sqrt(1.0 - a_prev) * eps_pred
""",
    ),
    (
        "ddim_step",
        "ddim_step",
        "returns the predicted clean sample",
        """
def ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    a_t = alphas_cumprod[t]
    return (x_t - np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "noise added on the last step",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(1.0 - beta)
    var = beta * (1.0 - alphas_cumprod[t - 1]) / (1.0 - a_t)
    return mean + np.sqrt(var) * noise
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "variance is beta",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(1.0 - beta)
    if t == 0:
        return mean
    return mean + np.sqrt(beta) * noise
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "divides by the cumulative alpha",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(a_t)
    if t == 0:
        return mean
    var = beta * (1.0 - alphas_cumprod[t - 1]) / (1.0 - a_t)
    return mean + np.sqrt(var) * noise
""",
    ),
    (
        "ddpm_step",
        "ddpm_step",
        "noise scaled by the variance, not the std",
        """
def ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise):
    beta, a_t = betas[t], alphas_cumprod[t]
    mean = (x_t - beta / np.sqrt(1.0 - a_t) * eps_pred) / np.sqrt(1.0 - beta)
    if t == 0:
        return mean
    var = beta * (1.0 - alphas_cumprod[t - 1]) / (1.0 - a_t)
    return mean + var * noise
""",
    ),
    (
        "cfg_noise",
        "cfg_noise",
        "the difference is the wrong way round",
        """
def cfg_noise(eps_cond, eps_uncond, scale):
    eps_cond = np.asarray(eps_cond, dtype=np.float64)
    eps_uncond = np.asarray(eps_uncond, dtype=np.float64)
    return eps_uncond + scale * (eps_uncond - eps_cond)
""",
    ),
    (
        "cfg_noise",
        "cfg_noise",
        "the conditional prediction is used as the base",
        """
def cfg_noise(eps_cond, eps_uncond, scale):
    eps_cond = np.asarray(eps_cond, dtype=np.float64)
    eps_uncond = np.asarray(eps_uncond, dtype=np.float64)
    return eps_cond + scale * (eps_cond - eps_uncond)
""",
    ),
    (
        "cfg_noise",
        "cfg_noise",
        "interpolates between the two instead of extrapolating",
        """
def cfg_noise(eps_cond, eps_uncond, scale):
    eps_cond = np.asarray(eps_cond, dtype=np.float64)
    eps_uncond = np.asarray(eps_uncond, dtype=np.float64)
    return (1.0 - scale) * eps_cond + scale * eps_uncond
""",
    ),
]
