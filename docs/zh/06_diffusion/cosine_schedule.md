<!-- en: f8aef3814903 -->
# cosine_schedule：余弦噪声调度

文件：`exercises/06_diffusion/cosine_schedule.py`

要实现：`cosine_schedule(num_steps, s=0.008, max_beta=0.999)`

## 题目

实现 Nichol 和 Dhariwal 提出的余弦噪声调度。Diffusion Policy 用的就是它（`squaredcos_cap_v2`）。

```
f(u)     = cos((u + s) / (1 + s) * pi / 2) ** 2
betas[i] = min(1 - f((i + 1) / num_steps) / f(i / num_steps), max_beta)
```

然后：

```
alphas_cumprod = cumprod(1 - betas)
```

返回 `(betas, alphas_cumprod)`，形状都是 `(num_steps,)`。
