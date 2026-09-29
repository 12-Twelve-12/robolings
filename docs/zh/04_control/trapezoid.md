<!-- en: dd32db946121 -->
# trapezoid：梯形速度规划

文件：`exercises/04_control/trapezoid.py`

要实现：`trapezoid(distance, v_max, a_max, t)`

## 题目

实现梯形速度规划，这是大多数电机驱动器内部跑的那一套。

以 `a_max` 加速到 `v_max`，匀速一段，再以 `a_max` 减速，到达 `distance` 时速度为零。

`t_ramp = v_max / a_max`，`d_ramp = v_max ** 2 / (2 * a_max)`。如果 `2 * d_ramp > distance`，
说明这段距离太短、根本到不了 `v_max`：曲线退化成三角形，峰值速度变成 `sqrt(a_max * distance)`，
加速时间也跟着缩短。漏掉这种情况正是朴素实现会冲过头的原因。

`t = 0` 之前输出 `(0, 0)`，走完之后输出 `(distance, 0)`。

`distance` 非负，`t` 是 Python 的 float。

返回 `(position, velocity)`，都是 Python 的 float。
