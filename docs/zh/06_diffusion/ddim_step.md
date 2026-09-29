<!-- en: 37e151045969 -->
# ddim_step：DDIM 单步

文件：`exercises/06_diffusion/ddim_step.py`

要实现：`ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod)`

## 题目

做一步确定性的 DDIM 更新，从时间步 `t` 走到 `t_prev`。

`eps_pred` 是网络对 `x_t` 预测出的噪声：

```
x0_pred = (x_t - sqrt(1 - a_t) * eps_pred) / sqrt(a_t)
x_prev  = sqrt(a_prev) * x0_pred + sqrt(1 - a_prev) * eps_pred
```

`t` 和 `t_prev` 是 Python 整数。`t_prev < 0` 表示这是最后一步，此时取 `a_prev = 1`，返回的就是 `x0_pred`。

有了 DDIM，用 100 步训练的策略推理时可以只跑 10 步，控制频率也就从 2 Hz 变成 20 Hz。

返回和 `x_t` 形状相同的数组。
