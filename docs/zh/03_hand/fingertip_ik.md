<!-- en: 7a56597f2651 -->
# fingertip_ik：带关节限位的指尖逆解

文件：`exercises/03_hand/fingertip_ik.py`

要实现：`fingertip_ik(fk_fn, jac_fn, q0, target, lower, upper, iters, damping)`

## 题目

把一个指尖移到 `target`，同时不超出关节限位。

重复 `iters` 次：

1. `err = target - fk_fn(q)`
2. `J = jac_fn(q)`，`(3, n)` 的位置雅可比
3. `dq = J.T @ solve(J @ J.T + damping**2 * I, err)`
4. `q = clip(q + dq, lower, upper)`

限位一定要放在循环里做。如果只在最后限一次，求解器会一直去推那个已经顶到限位的关节，其他关节得不到补偿的机会。

- `fk_fn(q)` 返回指尖位置，`(3,)`。
- `jac_fn(q)` 返回位置雅可比，`(3, n)`。

返回最终的 `q`，`(n,)` 数组。
