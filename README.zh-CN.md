# robolings

机器人学习方向的闯关练习，形式照搬 [rustlings](https://github.com/rust-lang/rustlings)。

一共 31 个小函数要你自己实现，从罗德里格斯公式开始，做到流匹配策略的采样器为止。只依赖 NumPy，不需要显卡和仿真器，全部测试不到 1 秒跑完。

[English](README.md)

## 用法

先 fork，再克隆你自己的 fork：

```bash
pip install -r requirements.txt
python robolings.py           # 看进度和下一题
python robolings.py run slerp # 只测某一题
python robolings.py list
python robolings.py show slerp --zh   # 看中文题面
python robolings.py --zh              # 题目名显示中文
```

每道题都有中文题面，放在 `docs/zh/` 下，也可以用上面的 `show` 命令直接看。不想每次都加 `--zh` 的话，设置环境变量 `ROBOLINGS_LANG=zh`。

题目在 `exercises/` 里，一题一个文件。把 `raise NotImplementedError` 换成你的实现，再跑一次就行。需要 Python 3.10 以上。

```text
robolings  [######........................]  6/31

  01_rotations
   [x]  1  rodrigues              Rodrigues' rotation formula
   [x]  2  quat_mul               Quaternion product
   ...
  02_kinematics
   [x]  7  transform_inverse      Inverse of a homogeneous transform
   [ ]  8  forward_kinematics     Forward kinematics of a serial chain

Next: forward_kinematics
  edit   exercises/02_kinematics/forward_kinematics.py
  check  python robolings.py run forward_kinematics
```

卡住了可以看 `solutions/` 里的参考解。

### 在 GitHub 上看进度

![progress](../../raw/progress/progress.svg)

每次往你的 fork 推送，都会自动跑一遍测试，在该次运行的摘要页生成进度表，并更新上面这个徽章。徽章显示的是你正在看的这个仓库的进度：这里是 0，在你自己的 fork 里就是你的题数。

fork 出来的仓库默认关闭 Actions，需要先到 Actions 页签手动开启一次。第一次运行之后徽章才会出现。

### 获取新题

在 GitHub 上点 **Sync fork**，或者合并 `upstream/main`。新题都是新增文件，不会动到你已经写好的答案。

## 题目

| 专题 | 题数 | 内容 |
|---|---|---|
| 旋转 | 6 | 罗德里格斯公式、四元数乘法、四元数与矩阵互转、球面插值、Kabsch 点集配准 |
| 运动学 | 4 | 齐次变换求逆、正运动学、几何雅可比、阻尼最小二乘 |
| 灵巧手 | 3 | 耦合关节、重定向代价、带限位的指尖逆解 |
| 控制 | 6 | 最小 jerk 轨迹、低通滤波、MIT 模式阻抗控制、指令限速、角度回绕、抗积分饱和 PID |
| 模仿学习 | 5 | 归一化、动作分块、相对动作、时间集成、观测历史 |
| 扩散策略 | 4 | 余弦调度、前向加噪、DDPM 单步、DDIM 单步 |
| 流匹配 | 3 | 训练目标、欧拉采样、中点法采样 |

## 说明

- 四元数顺序是 `[w, x, y, z]`，Hamilton 约定。
- 旋转矩阵把本体系坐标变到世界系。
- 流匹配里 `t = 0` 是噪声，`t = 1` 是数据。
- 每道题的测试都不依赖其他题，可以跳着做。
- 测试还用一批典型错解验证过（见 `tools/mutants.py`），比如矩阵转四元数时直接除以 `w`，或者逆解迭代完才做限位。如果你发现某种错解还能通过，欢迎提 issue。

## 参与贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。欢迎加新题，前提是仍然只依赖 NumPy。

## 许可证

MIT
