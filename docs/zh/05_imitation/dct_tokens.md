<!-- en: 292c8628e415 -->
# dct_tokens：DCT 动作分词

文件：`exercises/05_imitation/dct_tokens.py`

要实现：`dct_matrix(n)`、`tokenize(chunk, step)` 和 `detokenize(tokens, step)`

## 题目

这道题有三个函数。

### dct_matrix

构造 `n` 阶的标准正交 DCT-II 矩阵。

动作块在时间上是光滑的，所以它的能量大部分集中在少数几个低频系数上。先变换再量化，
正是 FAST 动作分词器的思路：高频系数量化后变成零，整个块就成了一小串整数。

NumPy 没有 DCT，要自己把矩阵建出来：

```
C[k, i] = s(k) * cos(pi * (2 i + 1) * k / (2 n))
s(0) = sqrt(1 / n),  s(k > 0) = sqrt(2 / n)
```

带上这组缩放系数之后矩阵是标准正交的，所以逆变换就是转置。

返回 `(n, n)` 数组。

### tokenize

把动作块变成整数 token。

沿时间轴用 `dct_matrix` 做变换，然后除以 `step` 并四舍五入到最近的整数。

`chunk` 是 `(H, D)`，变换沿 `H` 做、逐列进行，`C @ chunk` 本身就是这个意思。

返回 `(H, D)` 的整数数组。

### detokenize

从 token 还原出动作块。

乘回 `step`，再做逆变换。因为 `dct_matrix` 是标准正交的，逆变换就是它的转置，
既不需要第二个矩阵，也不需要求逆。

返回 `(H, D)` 的浮点数组。
