<!-- en: f8f1f9a8986a -->
# midpoint_sample：中点法采样

文件：`exercises/07_flow_matching/midpoint_sample.py`

要实现：`midpoint_sample(v_fn, x, num_steps)`

## 题目

用中点法对速度场做积分。

欧拉法整步都用起点处的速度。中点法先走半步，再用那里的速度走完整步：

```
x_mid = x + dt / 2 * v_fn(x, t)
x     = x + dt * v_fn(x_mid, t + dt / 2)
```

其中 `dt = 1 / num_steps`，`t = 0, dt, ..., 1 - dt`。

每一步要调用两次网络，误差按 `dt**2` 下降，而欧拉法是按 `dt`。学到的路径有弯曲时，同样的网络调用次数下它通常是更好的选择。

`v_fn(x, t)` 接收当前样本和一个 Python `float`，返回和 `x` 形状相同的数组。

返回和 `x` 形状相同的数组。

## 约定

本专题里 `t = 0` 是纯噪声，`t = 1` 是数据。有些论文的方向正好相反，移植代码前先确认。
