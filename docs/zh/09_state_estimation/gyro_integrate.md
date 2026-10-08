<!-- en: 83bcb9f6518b -->
# gyro_integrate：陀螺仪姿态积分

文件：`exercises/09_state_estimation/gyro_integrate.py`

要实现：`gyro_integrate(q, omega, dt)`

## 题目

用机体系的角速度传播姿态四元数。

`q` 是 `[w, x, y, z]`，把机体坐标映射到世界坐标，和本仓库其他地方一样。`omega` 是陀螺仪报出来的角速度，机体系，单位 rad/s。一步之内机体转过旋转向量 `omega * dt`，写成四元数是：

```
dq = [cos(theta / 2), sin(theta / 2) * omega / norm(omega)]
theta = norm(omega) * dt
```

机体系的旋转要乘在**右边**：

```
q_new = q * dq
```

结果再归一化，免得舍入误差越积越多。`norm(omega)` 极小时，`dq` 取 `[1, omega * dt / 2]`（再归一化），不要除以零。

`q` 是 `(4,)`，`omega` 是 `(3,)`，`dt` 是浮点数。

返回单位长度的 `(4,)` 数组。

## 约定

机器人从来看不到自己的状态，它看到的是传感器。陀螺仪平滑但会漂，加速度计吵但不漂，卡尔曼滤波则是把预测和测量加权起来的通用办法。这条专题把这些零件一个一个搭起来。
