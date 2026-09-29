<!-- en: 240dbcb42141 -->
# forward_kinematics：串联链正运动学

文件：`exercises/02_kinematics/forward_kinematics.py`

要实现：`forward_kinematics(origins, axes, q)`

## 题目

求链上每个连杆在世界系下的位姿。

返回 `(n, 4, 4)` 数组，第 `i` 项是连杆 `i` 的坐标系在世界系下的表示。

## 链的描述方式

`n` 个转动关节的串联链用两个数组描述：

- `origins`：`(n, 4, 4)`。`origins[i]` 是关节角为零时，从连杆 `i - 1` 坐标系到关节 `i` 坐标系的固定变换。`i = 0` 时父坐标系是世界系。对应 URDF 里的 `<origin>`。
- `axes`：`(n, 3)`。`axes[i]` 是关节 `i` 的单位转轴，表达在关节自己的坐标系下。对应 URDF 里的 `<axis>`。

于是连杆 `i` 在世界系下的位姿是：

```
T[i] = T[i - 1] @ origins[i] @ Rot(axes[i], q[i])
```
