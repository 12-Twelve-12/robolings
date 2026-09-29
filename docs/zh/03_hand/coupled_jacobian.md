<!-- en: 149e97ef6300 -->
# coupled_jacobian：耦合手的雅可比

文件：`exercises/03_hand/coupled_jacobian.py`

要实现：`coupled_jacobian(J_full, n_joints, active_idx, mimic)`

## 题目

把整手的雅可比换算到驱动关节上。

`expand_mimic` 做的是从驱动关节向量得到全关节向量，这道题要的是那个映射的导数。把它写成 `q_full = C @ q_act`，其中 `C` 的形状是 `(n_joints, k)`，链式法则给出：

```
J_act = J_full @ C
```

`C` 按列构造，一个驱动关节一列：

- 第 `j` 列在第 `active_idx[j]` 行上是 1
- 一条 mimic `(joint, source, multiplier, offset)`，是把 `multiplier` 放在第 `joint` 行、以及驱动 `source` 的那一列上

两处容易写错。`offset` 是常数，根本不该出现在导数里。还有 `source` 是全关节向量里的下标，必须映射回 `C` 的列号；直接拿它当列号用，在 `active_idx` 恰好是 `[0, 1, 2, ...]` 时也能跑对，换一个顺序就悄悄错了。

除 `J_full` 外，各参数的含义和 `expand_mimic` 里一致。`J_full` 是 `(m, n_joints)`。

返回 `(m, k)` 数组。
