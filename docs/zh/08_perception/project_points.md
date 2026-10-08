<!-- en: aabb946c1748 -->
# project_points：针孔投影

文件：`exercises/08_perception/project_points.py`

要实现：`project_points(points_world, T_world_cam, K)`

## 题目

把世界坐标系下的点投影进针孔相机。

`T_world_cam` 是相机位姿，和这里所有变换的约定一样：它把相机坐标映射到世界坐标。要把点送**进**相机坐标系，需要它的逆：

```
p_cam = inv(T_world_cam) @ [p_world, 1]
u = fx * x / z + cx
v = fy * y / z + cy
```

其中 `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`。

`z <= 0` 的点在相机后方（或正好在相机平面上），没有像。把它标成无效，像素那一行留 `nan`。

`points_world` 是 `(N, 3)`，`T_world_cam` 是 `(4, 4)`，`K` 是 `(3, 3)`。

返回 `(pixels, depth, valid)`：`(N, 2)` 浮点、`(N,)` 浮点（相机坐标系下的 `z`）、`(N,)` 布尔。

## 约定

相机坐标系：`z` 朝前穿过镜头，`x` 朝右，`y` 朝下。像素 `u` 沿列方向，`v` 沿行方向。
