<!-- en: 0752e9452dba -->
# innovation_gate：新息门限剔野

文件：`exercises/09_state_estimation/innovation_gate.py`

要实现：`innovation_gate(z, z_pred, S, threshold)`

## 题目

这个测量到底该不该用？

传感器抽风、数据关联错了、玻璃反光：滤波器没办法知道，除了一点——这个测量和它自己的预测差得太远，远过了预测本身的不确定度所允许的范围。新息的马氏距离平方量的正是这件事：

```
y = z - z_pred
d2 = y.T @ inv(S) @ y
```

`d2 <= threshold` 时接受这个测量。门限来自 `m` 个自由度的卡方分布（`m = 2`、99 % 时约为 `9.21`），由调用方给。

用 `np.linalg.solve` 而不是 `inv`，并且用完整的 `S`：只取对角线会忽略测量轴之间的相关性，把不该接受的东西放进来。

`z` 和 `z_pred` 是 `(m,)`，`S` 是 `(m, m)`。

返回 `(accept, d2)`：一个 Python `bool` 和一个 Python `float`。

## 约定

机器人从来看不到自己的状态，它看到的是传感器。陀螺仪平滑但会漂，加速度计吵但不漂，卡尔曼滤波则是把预测和测量加权起来的通用办法。这条专题把这些零件一个一个搭起来。
