<!-- en: 150d67e508ea -->
# retarget_cost：向量重定向代价

文件：`exercises/03_hand/retarget_cost.py`

要实现：`retarget_cost(human_vecs, robot_vecs, scale, q, q_prev, beta)`

## 题目

求基于向量的手部重定向代价。

重定向要把人手的姿态映射到比例不同的机器人手上。直接对齐关节角行不通，通常的做法是对齐**向量**，比如手腕到指尖：

```
cost = 0.5 * sum_i || scale * human_vecs[i] - robot_vecs[i] ||^2
     + 0.5 * beta * || q - q_prev ||^2
```

`scale` 用来补偿两只手的大小差异。第二项惩罚离上一帧的解太远，第一项有多个极小值时靠它保持动作连续。

`human_vecs` 和 `robot_vecs` 是 `(k, 3)`，`q` 和 `q_prev` 是 `(n,)`。

返回 Python 的 `float`。
