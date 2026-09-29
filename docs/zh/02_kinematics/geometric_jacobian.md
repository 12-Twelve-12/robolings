<!-- en: 79022a5ce518 -->
# geometric_jacobian：几何雅可比

文件：`exercises/02_kinematics/geometric_jacobian.py`

要实现：`geometric_jacobian(frames, axes, tip_offset)`

## 题目

求固连在最后一个连杆上的某个点的几何雅可比。

`frames` 是 `forward_kinematics` 的输出。`tip_offset` 是关注点在最后一个连杆坐标系下的位置，比如指尖。

结果的第 `i` 列描述关节 `i` 对这个点的作用：

```
a_i = R_i @ axes[i]            关节转轴，世界系
p_i = frames[i][:3, 3]         关节位置，世界系
J[:3, i] = cross(a_i, p_tip - p_i)     线速度
J[3:, i] = a_i                         角速度
```

返回 `(6, n)` 数组：上三行是线速度，下三行是角速度。

## 链的描述方式

`n` 个转动关节的串联链用两个数组描述：

- `origins`：`(n, 4, 4)`。`origins[i]` 是关节角为零时，从连杆 `i - 1` 坐标系到关节 `i` 坐标系的固定变换。`i = 0` 时父坐标系是世界系。对应 URDF 里的 `<origin>`。
- `axes`：`(n, 3)`。`axes[i]` 是关节 `i` 的单位转轴，表达在关节自己的坐标系下。对应 URDF 里的 `<axis>`。

于是连杆 `i` 在世界系下的位姿是：

```
T[i] = T[i - 1] @ origins[i] @ Rot(axes[i], q[i])
```
