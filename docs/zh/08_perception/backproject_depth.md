<!-- en: 525df7c0b5b5 -->
# backproject_depth：深度图反投影成点云

文件：`exercises/08_perception/backproject_depth.py`

要实现：`backproject_depth(depth, K)`

## 题目

把一张深度图变成相机坐标系下的点云。

`depth[v, u]` 是像素 `(u, v)` 看到的表面在相机坐标系下的 `z`，不是沿射线的距离。把针孔模型反过来：

```
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

像素坐标就是数组下标：`u` 是列，`v` 是行，没有半像素偏移。

没有测到深度的像素，深度是 `0` 或 `nan`。它不产生点：标成无效，那一行留 `nan`。

`depth` 是 `(H, W)`，`K` 是 `(3, 3)`。

返回 `(points, valid)`：`(H, W, 3)` 浮点和 `(H, W)` 布尔。

## 约定

相机坐标系：`z` 朝前穿过镜头，`x` 朝右，`y` 朝下。像素 `u` 沿列方向，`v` 沿行方向。
