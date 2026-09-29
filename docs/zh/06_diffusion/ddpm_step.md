<!-- en: 82e28cbca932 -->
# ddpm_step：DDPM 单步

文件：`exercises/06_diffusion/ddpm_step.py`

要实现：`ddpm_step(x_t, eps_pred, t, betas, alphas_cumprod, noise)`

## 题目

做一步带随机性的 DDPM 更新，从时间步 `t` 走到 `t - 1`。

`eps_pred` 是网络对 `x_t` 预测出的噪声。记 `a = alphas_cumprod`：

```
mean = (x_t - betas[t] / sqrt(1 - a[t]) * eps_pred) / sqrt(1 - betas[t])
var  = betas[t] * (1 - a[t - 1]) / (1 - a[t])
out  = mean + sqrt(var) * noise
```

`var` 是真实后验的方差，不是 `betas[t]`。

`t = 0` 时已经没有噪声要加了，直接返回 `mean`，并且不要去读 `a[-1]`：在 Python 里它是最后一个元素，不会报错。

`t` 是 Python 整数。`noise` 和 `x_t` 形状相同，由外部传入，这样函数的结果是确定的。

返回和 `x_t` 形状相同的数组。
