<!-- en: 080c32ec7452 -->
# q_sample：前向加噪

文件：`exercises/06_diffusion/q_sample.py`

要实现：`q_sample(x0, t, noise, alphas_cumprod)`

## 题目

前向过程：把干净的动作块加噪到时间步 `t`。

```
x_t = sqrt(a_t) * x0 + sqrt(1 - a_t) * noise,   a_t = alphas_cumprod[t]
```

`x0` 和 `noise` 是 `(B, ...)`。`t` 是 `(B,)` 的整数数组，每个样本有自己的时间步，所以 `a_t` 要先调整形状，才能和后面的维度广播。

返回和 `x0` 形状相同的数组。
