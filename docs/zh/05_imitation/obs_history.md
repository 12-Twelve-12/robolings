<!-- en: b03798405222 -->
# obs_history：观测历史窗口

文件：`exercises/05_imitation/obs_history.py`

要实现：`stack_obs_history(obs, n)`

## 题目

给每一步配上它最近的 `n` 帧观测。

```
out[t] = [obs[t - n + 1], ..., obs[t - 1], obs[t]]
```

轨迹刚开始时还没有历史。缺的位置用第一帧观测重复填充，机器人部署时也是这么做的。

`obs` 是 `(T, D)`。

返回 `(T, n, D)` 数组，最旧的观测排在前面。
