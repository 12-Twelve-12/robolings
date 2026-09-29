<!-- en: 7c0224c753d6 -->
# min_jerk：最小 jerk 轨迹

文件：`exercises/04_control/min_jerk.py`

要实现：`min_jerk(q0, q1, duration, t)`

## 题目

求从 `q0` 到 `q1` 的最小 jerk 轨迹在时刻 `t` 的值。

令 `tau = clip(t / duration, 0, 1)`：

```
s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
pos    = q0 + (q1 - q0) * s
```

速度和加速度是 `pos` 对时间的一阶、二阶导数。别忘了链式法则：`d tau / dt = 1 / duration`。

在 `[0, duration]` 之外，位置保持在最近的端点，速度和加速度为零。

返回 `(pos, vel, acc)`，形状都和 `q0` 相同。
