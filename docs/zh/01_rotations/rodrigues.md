<!-- en: 4f97474c351f -->
# rodrigues：罗德里格斯旋转公式

文件：`exercises/01_rotations/rodrigues.py`

要实现：`rodrigues(axis, angle)`

## 题目

求绕 `axis` 轴转 `angle` 弧度的旋转矩阵。

`axis` 是三维向量，**不保证是单位长度**，要先归一化。设 `K` 是单位转轴的反对称矩阵：

```
R = I + sin(angle) * K + (1 - cos(angle)) * K @ K
```

返回 `(3, 3)` 数组。

## 约定

- 旋转矩阵是 3x3，作用在列向量上，把本体系坐标变到世界系。
- 四元数顺序是 `[w, x, y, z]`（标量在前），Hamilton 约定。
- 全部用 `float64`。
