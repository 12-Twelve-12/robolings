<!-- en: 04d5a37706b2 -->
# mit_torque：MIT 模式阻抗控制

文件：`exercises/04_control/mit_torque.py`

要实现：`mit_torque(kp, kd, q_des, dq_des, q, dq, tau_ff, tau_limit)`

## 题目

求 MIT 模式阻抗控制器输出的关节力矩。

准直驱关节电机大多提供这个控制律：

```
tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
```

把结果限幅到 `[-tau_limit, tau_limit]`。限幅作用在**总力矩**上，不是分别作用在各项上。

所有参数之间按广播规则运算。

返回广播后形状的数组。
