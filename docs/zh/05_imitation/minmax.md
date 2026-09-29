<!-- en: cbc1605d417c -->
# minmax：最小最大归一化与反归一化

文件：`exercises/05_imitation/minmax.py`

要实现：`minmax_normalize(x, lo, hi)` 和 `minmax_unnormalize(y, lo, hi)`

## 题目

这道题有两个函数。

### minmax_normalize

把 `x` 从 `[lo, hi]` 逐维映射到 `[-1, 1]`。

`lo` 和 `hi` 是训练集上每一维的最小值和最大值。数据里从来不动的维度会有 `hi == lo`，除以这个范围得到 `nan`，并且会悄悄污染训练。这种维度要映射成 `0`。

`x` 是 `(..., D)`，`lo` 和 `hi` 是 `(D,)`。

返回和 `x` 形状相同的数组。

### minmax_unnormalize

`minmax_normalize` 的逆：把 `[-1, 1]` 映射回 `[lo, hi]`。

对常量维度（`hi == lo`），不管 `y` 是多少都返回 `lo`。策略在这一维上的输出不带任何信息。

返回和 `y` 形状相同的数组。
