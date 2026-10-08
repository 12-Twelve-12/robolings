<!-- en: 01b04864a6ee -->
# ekf_update：EKF 更新步（Joseph 形式）

文件：`exercises/09_state_estimation/ekf_update.py`

要实现：`ekf_update(x, P, z, h, H, R)`

## 题目

扩展卡尔曼滤波的更新那一半。

把测量和状态预测出来的测量比一比，按卡尔曼增益把状态往测量那边挪：

```
y = z - h(x)                      新息
S = H @ P @ H.T + R               新息协方差
K = P @ H.T @ inv(S)              增益
x_new = x + K @ y
P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
```

最后一行是 Joseph 形式。教科书上的 `(I - K H) P` 代数上一样，但浮点数下跑几千步之后对称性和正定性就丢了，所以用长的那个。

`h` 是 Python 可调用对象，`h(x)` 返回预测的测量 `(m,)`。`H` 是它在 `x` 处的 `(m, n)` 雅可比。`R` 是 `(m, m)` 的测量噪声。用 `np.linalg.solve`，别真的去算 `inv(S)`。

返回 `(x_new, P_new)`。

## 约定

机器人从来看不到自己的状态，它看到的是传感器。陀螺仪平滑但会漂，加速度计吵但不漂，卡尔曼滤波则是把预测和测量加权起来的通用办法。这条专题把这些零件一个一个搭起来。
