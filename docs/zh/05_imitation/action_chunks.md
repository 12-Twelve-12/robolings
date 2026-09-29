<!-- en: fd94ce8df07b -->
# action_chunks：动作分块与填充掩码

文件：`exercises/05_imitation/action_chunks.py`

要实现：`make_action_chunks(actions, horizon)`

## 题目

把一条轨迹切成相互重叠的动作块。

第 `t` 块是从第 `t` 步开始的 `horizon` 个动作：

```
chunks[t, k] = actions[t + k]
```

靠近轨迹末尾时 `t + k` 会越界。越界的位置用最后一个动作重复填充，并在 `is_pad` 里标出来，好让损失函数忽略它们。

`actions` 是 `(T, D)`。

返回 `(chunks, is_pad)`，形状分别是 `(T, horizon, D)` 和 `(T, horizon)`。`is_pad` 是布尔数组，填充的位置为 `True`。
