<!-- en: ac14683c25ed -->
# null_space_step：零空间投影

文件：`exercises/02_kinematics/null_space_step.py`

要实现：`null_space_step(J, dx, q_secondary)`

## 题目

在完成任务空间增量的同时，让次要目标不干扰任务。

冗余机械臂的关节数比任务约束多，多出来的那部分运动可以拿去做别的事：躲开关节限位、让肘部避开障碍、或者慢慢回到一个舒服的位形。标准写法把两件事分开：

```
dq = pinv(J) @ dx + (I - pinv(J) @ J) @ q_secondary
```

第一项是 `J @ dq = dx` 的最小范数解。第二项把 `q_secondary` 投影到 `J` 的零空间里，所以不管它想要什么，`J @ dq` 都不变。

这个不变性正是值得测的性质，也正是漏掉投影器之后失去的东西：直接把 `q_secondary` 加上去，末端仍然大致朝正确方向走，所以只验第一项的测试照样能过。

`J` 是 `(m, n)` 且 `m <= n`，`dx` 是 `(m,)`，`q_secondary` 是 `(n,)`。用 `np.linalg.pinv`。

返回 `(n,)` 数组。
