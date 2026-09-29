<!-- en: e883bb74bc82 -->
# expand_mimic：耦合关节展开

文件：`exercises/03_hand/expand_mimic.py`

要实现：`expand_mimic(q_active, n_joints, active_idx, mimic)`

## 题目

由主动关节的角度，求带耦合关节的手的完整关节向量。

腱绳或连杆驱动的手指，电机数比关节数少。URDF 用 `<mimic>` 表达这种关系：被动关节按 `q[joint] = multiplier * q[source] + offset` 跟随某个主动关节。

- `q_active`：`(k,)`，主动关节的角度。
- `n_joints`：手的关节总数。
- `active_idx`：`(k,)`，每个主动关节在完整向量里的下标。
- `mimic`：`(joint, source, multiplier, offset)` 的列表。`source` 是完整向量里的下标，并且一定指向主动关节。

既不是主动、也不跟随别人的关节保持为零。

返回 `(n_joints,)` 数组。
