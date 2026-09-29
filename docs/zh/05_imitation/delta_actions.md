<!-- en: f35a6c0be59d -->
# delta_actions：相对当前状态的动作

文件：`exercises/05_imitation/delta_actions.py`

要实现：`to_delta(chunk, state, absolute_mask)` 和 `from_delta(delta, state, absolute_mask)`

## 题目

这道题有两个函数。

### to_delta

把动作块表示成相对机器人当前状态的量。

策略学“从现在的位置往上 2 厘米”通常比学绝对目标更容易。块里的每个动作都相对于**块开始时**的状态：

```
delta[k] = chunk[k] - state
```

注意不是相邻两个动作之差。用相邻差的话，某一步有一点误差，后面每一步都会跟着偏。

有些维度应当保持绝对量，比如夹爪开度。`absolute_mask` 在这些维度上为 `True`，它们原样复制。

`chunk` 是 `(H, D)`，`state` 和 `absolute_mask` 是 `(D,)`。

返回 `(H, D)` 数组。

### from_delta

`to_delta` 的逆：把预测出的动作块还原成绝对目标。

`state` 必须是策略做这次预测时看到的状态，不是每个动作执行时的状态。

返回 `(H, D)` 数组。
