<!-- en: c1cf49b927c5 -->
# cfg_noise：无分类器引导

文件：`exercises/06_diffusion/cfg_noise.py`

要实现：`cfg_noise(eps_cond, eps_uncond, scale)`

## 题目

把条件预测和无条件预测合成一个噪声预测。

无分类器引导会让网络跑两遍，一遍带观测、一遍把观测丢掉，然后朝着背离无条件预测的方向外推：

```
eps = eps_uncond + scale * (eps_cond - eps_uncond)
```

`scale = 1` 复现条件预测，`scale = 0` 复现无条件预测，大于 1 则朝条件要求的方向推得更远。对策略来说，这个旋钮一端是无视相机看到的东西，另一端是过度照着它做。

就一行，值得单独出一题，是因为符号和基准项两处都容易写反。把两个锚点钉住，写错的版本就无处可藏：拿 `eps_cond` 当基准的版本在 `scale = 0` 处给出错误答案，把差反过来写的版本在 `scale = 1` 处给出错误答案。

`eps_cond` 和 `eps_uncond` 形状相同，`scale` 是 Python 浮点数。

返回同样形状的数组。
