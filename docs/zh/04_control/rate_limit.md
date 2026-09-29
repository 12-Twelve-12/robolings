<!-- en: bbc60ec36ac8 -->
# rate_limit：指令限速

文件：`exercises/04_control/rate_limit.py`

要实现：`rate_limit(target, prev, max_rate, dt)`

## 题目

限制指令的变化速度。

策略给出的目标可能离当前指令很远，直接发给位置控制的关节会产生剧烈动作。把每个控制周期的变化量限制在 `max_rate * dt` 以内，各关节分别处理：

```
out = prev + clip(target - prev, -max_rate * dt, max_rate * dt)
```

`max_rate` 可以是标量，也可以是每个关节一个值的数组。

返回和 `target` 形状相同的数组。
