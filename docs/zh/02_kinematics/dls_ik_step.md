<!-- en: 986b8b150bbd -->
# dls_ik_step：阻尼最小二乘逆解

文件：`exercises/02_kinematics/dls_ik_step.py`

要实现：`dls_ik_step(J, err, damping)`

## 题目

做一步阻尼最小二乘逆运动学，求出能减小任务空间误差 `err` 的关节增量：

```
dq = J.T @ inv(J @ J.T + damping**2 * I) @ err
```

直接用伪逆的话，靠近奇异位形时会算出极大的关节速度。阻尼项牺牲一点精度，换来有界的步长，所以真机上的机械臂和灵巧手默认都用它。

`J` 是 `(m, n)`，`err` 是 `(m,)`。用 `np.linalg.solve`，不要显式求逆。

返回 `(n,)` 数组。
