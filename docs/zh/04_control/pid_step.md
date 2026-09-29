<!-- en: 5026c44d859d -->
# pid_step：带抗积分饱和的 PID

文件：`exercises/04_control/pid_step.py`

要实现：`pid_step(state, error, dt, kp, ki, kd, u_limit)`

## 题目

做一步输出带限幅的 PID 控制。

`state` 是 `(integral, prev_error)`。第一次调用时 `prev_error` 是 `None`。

```
integral_new = integral + error * dt
derivative   = (error - prev_error) / dt      第一次调用时取 0
u_raw        = kp * error + ki * integral_new + kd * derivative
u            = clip(u_raw, -u_limit, u_limit)
```

抗积分饱和：如果输出已经饱和，并且误差还在把它往饱和方向推（`u_raw` 和 `error` 同号），就丢掉 `integral_new`，保留原来的 `integral`。

不这样做的话，只要执行器顶在限幅上，积分项就会一直涨。等误差终于反号，控制器还得先把攒下的积分全部消掉才能退出饱和，关节会严重超调。

第一次调用时没有上一次的误差。如果拿零去代替，微分项会突跳，所以规定此时微分为零。

所有数值都是 Python 的 `float`。

返回 `(u, (integral, error))`，即本次输出和下一次调用要用的状态。
