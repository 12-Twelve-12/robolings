<!-- en: a62d00bf3c09 -->
# distortion：镜头畸变与去畸变

文件：`exercises/08_perception/distortion.py`

要实现：`distort_points(xy, coeffs)` 和 `undistort_points(xy_d, coeffs, iters=10)`

## 题目

### distort_points

对归一化图像坐标施加 Brown-Conrady 镜头畸变。

`xy` 是 `z = 1` 平面上的点，还没有乘 `K`。记 `coeffs = (k1, k2, p1, p2)`，`r2 = x**2 + y**2`：

```
radial = 1 + k1 * r2 + k2 * r2**2
x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
y_d = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
```

径向部分把点沿着它到中心的射线移动；切向部分（`p1`、`p2`）打破的正是这种对称。

`xy` 是 `(N, 2)`。返回 `(N, 2)`。

### undistort_points

把畸变反过来。它没有闭式解。

用定点迭代，从畸变点自身出发。每一步在当前估计处算出径向因子和切向偏移，再按正向模型解出去畸变的点：

```
x = (x_d - tangential_x(x, y)) / radial(x, y)
y = (y_d - tangential_y(x, y)) / radial(x, y)
```

恰好跑 `iters` 轮。一轮不够：径向因子得在去畸变后的点上取值，而那个点你还没有。

`xy_d` 是 `(N, 2)`。返回 `(N, 2)`。

## 约定

相机坐标系：`z` 朝前穿过镜头，`x` 朝右，`y` 朝下。像素 `u` 沿列方向，`v` 沿行方向。
