<!-- en: 10b2f330e0c8 -->
# quat_mul：四元数乘法

文件：`exercises/01_rotations/quat_mul.py`

要实现：`quat_mul(q1, q2)`

## 题目

求两个 `[w, x, y, z]` 四元数的 Hamilton 乘积 `q1 * q2`。

先按 `q2` 转、再按 `q1` 转，合起来就是 `q1 * q2`。乘法不满足交换律。

返回 `(4,)` 数组。结果不要归一化。

## 约定

- 旋转矩阵是 3x3，作用在列向量上，把本体系坐标变到世界系。
- 四元数顺序是 `[w, x, y, z]`（标量在前），Hamilton 约定。
- 全部用 `float64`。
