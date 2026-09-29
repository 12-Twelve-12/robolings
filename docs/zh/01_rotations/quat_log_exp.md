<!-- en: 3479df3c84ed -->
# quat_log_exp：四元数对数与指数映射

文件：`exercises/01_rotations/quat_log_exp.py`

要实现：`quat_log(q)` 和 `quat_exp(v)`

## 题目

这道题有两个函数。

### quat_log

求单位四元数对应的旋转向量。

旋转向量是转轴乘以转角，所以它的模就是弧度制的转角。姿态控制器要的就是这个量：
误差 `quat_log(q_des * conj(q))` 是一个可以直接乘增益的向量。

记 `q = [w, v]`、`theta = 2 * atan2(norm(v), w)`：

```
quat_log(q) = theta * v / norm(v)
```

两个细节：

* `q` 和 `-q` 是同一个旋转，但上面的公式会把它们分别映射成长度 `theta` 和 `2 * pi - theta` 的向量。
  `w < 0` 时先把 `q` 取反，这样结果永远走近路。
* 转角接近零时 `norm(v)` 趋于零，`theta` 也趋于零。这个比值的极限是有限的，等于 2，
  所以 `norm(v)` 很小时直接返回 `2 * v`，不要做除法。

返回 `(3,)` 数组。

### quat_exp

求旋转向量对应的单位四元数，是 `quat_log` 的逆。

记 `theta = norm(v)`：

```
quat_exp(v) = [cos(theta / 2), sin(theta / 2) * v / theta]
```

小角度的问题和 `quat_log` 里一样：`sin(theta / 2) / theta` 趋于 `1 / 2`，
所以 `theta` 很小时返回归一化之后的 `[1, v / 2]`。

返回单位长度的 `(4,)` 数组。
