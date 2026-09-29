<!-- en: e76052dc95e8 -->
# temporal_ensemble：时间集成（ACT）

文件：`exercises/05_imitation/temporal_ensemble.py`

要实现：`temporal_ensemble(preds, m)`

## 题目

把对当前时刻的多个重叠预测融合成一个动作，做法同 ACT。

用了动作分块之后，当前时刻的动作被预测过好几次：最近每一个覆盖到它的块都给过一个值。ACT 用指数权重对它们做平均：

```
w[i] = exp(-m * i)
out  = sum_i w[i] * preds[i] / sum_i w[i]
```

`preds` 按从旧到新排列，所以 `i = 0` 是**最旧**的预测，权重**最大**。`m` 越小，新观测被采纳得越慢。

`preds` 是 `(K, D)`。

返回 `(D,)` 数组。
