<!-- en: 371a18610710 -->
# transform_inverse：齐次变换求逆

文件：`exercises/02_kinematics/transform_inverse.py`

要实现：`transform_inverse(T)`

## 题目

求齐次变换的逆，不许用通用的矩阵求逆。

对 `T = [[R, p], [0, 1]]`，它的逆是 `[[R.T, -R.T @ p], [0, 1]]`。这样算比 `np.linalg.inv` 更省、数值上也更好，控制循环里用的就是它。

不要调用 `np.linalg.inv` 和 `np.linalg.solve`。

返回 `(4, 4)` 数组。
