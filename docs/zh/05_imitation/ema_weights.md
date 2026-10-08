<!-- en: a5f41187ecd9 -->
# ema_weights：策略权重的 EMA

文件：`exercises/05_imitation/ema_weights.py`

要实现：`ema_decay(step, max_decay=0.9999, warmup=10.0)` 和 `ema_update(average, weights, step, ...)`

## 题目

这道题有两个函数。

### ema_decay

求某一步上权重滑动平均的衰减率。

Diffusion Policy 评估时用的是权重的指数滑动平均，而不是权重本身。如果衰减率从一开始就是接近 1 的常数，
这个平均值会在最初几千步里一直被钉在随机初值附近，所以要让衰减率逐渐爬上去：

```
decay = min(max_decay, (1 + step) / (warmup + step))
```

`step` 从 0 开始计。

返回 Python 的 float。

### ema_update

对一个数组字典做一步更新。

```
average[k] = decay * average[k] + (1 - decay) * weights[k]
```

返回一个新字典、装的是新数组。原地更新会改掉调用方还拿着的数组，
而训练循环拿着的恰恰就是那些：它马上要在上面再走一步梯度的那份权重。

`average` 和 `weights` 的键相同，形状一一对应。

返回一个新的字典。
