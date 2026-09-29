<!-- en: 98df13c6cc4e -->
# lowpass_filter：一阶低通滤波

文件：`exercises/04_control/lowpass_filter.py`

要实现：`lowpass_filter(x, cutoff_hz, dt)`

## 题目

对一串采样做一阶低通滤波。

```
rc    = 1 / (2 * pi * cutoff_hz)
alpha = dt / (rc + dt)
y[0]  = x[0]
y[k]  = y[k - 1] + alpha * (x[k] - y[k - 1])
```

初值取 `y[0] = x[0]` 而不是零，可以避免启动瞬态。放在机器人上，那个瞬态就是指令的一次跳变。

`x` 是 `(T,)` 或 `(T, D)`，各维独立滤波。

返回和 `x` 形状相同的数组。
