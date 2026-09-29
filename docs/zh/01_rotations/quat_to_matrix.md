<!-- en: 2542d5603667 -->
# quat_to_matrix：四元数转旋转矩阵

文件：`exercises/01_rotations/quat_to_matrix.py`

要实现：`quat_to_matrix(q)`

## 题目

求 `[w, x, y, z]` 四元数对应的旋转矩阵。

`q` 不一定严格是单位长度，要先归一化。

返回 `(3, 3)` 数组。

## 约定

- 旋转矩阵是 3x3，作用在列向量上，把本体系坐标变到世界系。
- 四元数顺序是 `[w, x, y, z]`（标量在前），Hamilton 约定。
- 全部用 `float64`。
