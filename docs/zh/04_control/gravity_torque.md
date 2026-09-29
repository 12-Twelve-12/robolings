<!-- en: 506682d33590 -->
# gravity_torque：平面机械臂重力补偿

文件：`exercises/04_control/gravity_torque.py`

要实现：`gravity_torque(q, lengths, masses, com, g=9.81)`

## 题目

求让一条平面机械臂抵抗重力所需的关节力矩。

手臂在竖直平面内，关节 `i` 绕平面外的轴转动，`q[i]` 是相对于前一根连杆的角度。重力沿 `-y`。

* `lengths[i]` 是连杆 `i` 的长度。
* `masses[i]` 是它的质量。
* `com[i]` 是从关节 `i` 到连杆 `i` 质心的距离，沿着连杆量。

把结果当作 `tau_ff` 喂给 `mit_torque`，手臂在两个设定点之间就不会往下垂。

连杆 `i` 质心的高度是：

```
y_i = 对所有 j < i 求和 lengths[j] * sin(a_j)   +   com[i] * sin(a_i)
```

其中 `a_i` 是连杆 `i` 的绝对角度，也就是 `q[0..i]` 的累加和。势能是 `g * sum_i masses[i] * y_i`，
与之平衡的力矩就是它对 `q` 的梯度。

逐个关节手推梯度很繁琐。有个捷径：关节 `k` 会带动从 `k` 开始往后的每一根连杆，所以

```
tau[k] = g * 对所有 i >= k 求和 masses[i] * d y_i / d q[k]
```

而 `d y_i / d q[k]` 是在同一批连杆上的余弦求和。

返回 `(n,)` 数组。
