<!-- en: 423a640104cc -->
# flow_matching_target：流匹配训练目标

文件：`exercises/07_flow_matching/flow_matching_target.py`

要实现：`flow_matching_target(noise, data, t)`

## 题目

构造条件流匹配的训练样本。

```
x_t = (1 - t) * noise + t * data
v   = data - noise
```

网络的输入是 `x_t` 和 `t`，训练目标是输出 `v`。

`noise` 和 `data` 是 `(B, ...)`。`t` 是 `(B,)`，取值在 `[0, 1]`，每个样本一个，所以要先调整形状，才能和后面的维度广播。

返回 `(x_t, v)`，形状都和 `data` 相同。

## 约定

本专题里 `t = 0` 是纯噪声，`t = 1` 是数据。有些论文的方向正好相反，移植代码前先确认。
