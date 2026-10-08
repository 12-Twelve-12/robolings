<!-- en: 730479024656 -->
# icp_step：ICP 单步迭代

文件：`exercises/08_perception/icp_step.py`

要实现：`icp_step(source, target, T)`

## 题目

点到点 ICP 的一次迭代。

`T` 是当前对刚体变换的猜测，它把 `source` 的点送进 `target` 的坐标系。一步做四件事：

1. 用 `T` 移动 source 的点。
2. 给每个移动后的 source 点配上最近的 target 点（暴力搜索就够）。
3. 解出从移动后的 source 点到各自配对点的最优刚体变换 `dT`：两组点各自去中心，对互协方差做 SVD，结果是反射就翻转最后一根轴（Kabsch）。
4. 新的估计是 `dT @ T`。修正量是在 target 坐标系里求出来的，所以乘在左边。

同时报告这次配对有多好：施加更新之后，同一批配对点之间距离的均方根。

`source` 是 `(N, 3)`，`target` 是 `(M, 3)`，`T` 是 `(4, 4)`。`N` 和 `M` 可以不同。

返回 `(T_new, rms)`：`(4, 4)` 和一个 Python `float`。

## 约定

相机坐标系：`z` 朝前穿过镜头，`x` 朝右，`y` 朝下。像素 `u` 沿列方向，`v` 沿行方向。
