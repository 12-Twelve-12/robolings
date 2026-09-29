<!-- en: 697abf7fc322 -->
# friction_cone：摩擦锥投影

文件：`exercises/03_hand/friction_cone.py`

要实现：`project_to_friction_cone(force, normal, mu)`

## 题目

把接触力投影到库仑摩擦锥上。

指尖只能推不能拉，能承担的切向力也最多是法向力的 `mu` 倍。凡是接触控制器算出来的力，都得先裁到这个集合里。

先把接触处的力按法向拆开，`n` 是 `normal` 归一化之后的结果：

```
f_n = dot(force, n)
f_t = force - f_n * n
```

三种情形都要对：

1. `norm(f_t) <= mu * f_n`：已经在锥内，原样返回。
2. `mu * norm(f_t) <= -f_n`：力压向表面内部，一点也留不下，返回零向量。
3. 其余情况，投影到锥面上：

```
s   = (mu * norm(f_t) + f_n) / (mu**2 + 1)
out = s * n + mu * s * f_t / norm(f_t)
```

错解都出在第 3 种情形。把整个向量按比例缩到"装得下"为止，法向分量会跟着变小，那不是投影，而且会悄悄削弱抓取。

`force` 和 `normal` 都是 `(3,)`，`normal` 不必是单位长度。`mu >= 0`。

返回 `(3,)` 数组。
