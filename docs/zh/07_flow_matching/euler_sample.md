<!-- en: 0888f8149c43 -->
# euler_sample：欧拉采样

文件：`exercises/07_flow_matching/euler_sample.py`

要实现：`euler_sample(v_fn, x, num_steps)`

## 题目

对学到的速度场做积分，生成一个动作块。

从 `t = 0` 的噪声 `x` 出发，走 `num_steps` 步欧拉积分到 `t = 1`，步长 `dt = 1 / num_steps`：

```
x = x + dt * v_fn(x, t)
```

速度在每一步的**起点**取值，所以 `t` 依次是 `0, dt, 2 dt, ..., 1 - dt`，不会取到 `1`。

`v_fn(x, t)` 接收当前样本和一个 Python `float`，返回和 `x` 形状相同的数组。

返回和 `x` 形状相同的数组。

## 约定

本专题里 `t = 0` 是纯噪声，`t = 1` 是数据。有些论文的方向正好相反，移植代码前先确认。
