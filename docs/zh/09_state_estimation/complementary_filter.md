<!-- en: 95295261ec8f -->
# complementary_filter：互补滤波

文件：`exercises/09_state_estimation/complementary_filter.py`

要实现：`complementary_filter(angle, gyro, accel_angle, dt, alpha)`

## 题目

倾角互补滤波的一次更新。

陀螺仪给出的角速度在一步之内积分得很干净，但几分钟下来会漂。加速度计给出的绝对角度每一步都有噪声，但永远不漂。把两者混起来：

```
angle = alpha * (angle + gyro * dt) + (1 - alpha) * accel_angle
```

`alpha` 接近 `1` 就是信陀螺仪，`alpha = 0` 就是把它扔掉。陀螺项是从**上一次的估计**开始积分的，不是从加速度计的角度。

`angle`、`gyro`、`accel_angle` 是浮点数，或者形状相同的数组，这样可以同时滤横滚和俯仰。`dt` 和 `alpha` 是浮点数，`0 <= alpha <= 1`。

返回新的角度：输入是浮点数就返回 Python `float`，否则返回同形状的数组。

## 约定

机器人从来看不到自己的状态，它看到的是传感器。陀螺仪平滑但会漂，加速度计吵但不漂，卡尔曼滤波则是把预测和测量加权起来的通用办法。这条专题把这些零件一个一个搭起来。
