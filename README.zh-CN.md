# robolings

机器人学习方向的闯关练习，形式照搬 [rustlings](https://github.com/rust-lang/rustlings)。

一共 55 个小函数要你自己实现，从罗德里格斯公式开始，做到流匹配策略的采样器，连同喂给它的感知和状态估计。只依赖 NumPy，不需要显卡和仿真器，全部测试不到 1 秒跑完。

[English](README.md)

![一次 watch 会话：错的 slerp 挂了三个测试，提示点出修法，改对后全过](docs/demo.svg)

## 用法

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/12-Twelve-12/robolings?quickstart=1)

什么都不想装的话，直接在这个仓库上开一个 Codespace：Python、NumPy 和编辑器都配好了，终端里直接 `python robolings.py`。想保存自己的答案，就先 fork，再克隆你自己的 fork：

```bash
pip install -r requirements.txt
python robolings.py           # 看进度和下一题
python robolings.py watch     # 改完存盘就自动重测
python robolings.py run slerp # 只测某一题
python robolings.py show slerp --zh   # 看中文题面
python robolings.py hint slerp --zh   # 卡住了看提示
python robolings.py list
python robolings.py --zh      # 题目名显示中文
```

每道题都有中文题面，放在 `docs/zh/` 下，也可以用上面的 `show` 命令直接看。不想每次都加 `--zh` 的话，设置环境变量 `ROBOLINGS_LANG=zh`。

题目在 `exercises/` 里，一题一个文件。把 `raise NotImplementedError` 换成你的实现，再跑一次就行。需要 Python 3.10 以上。

```text
robolings  [###...........................]  6/55

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
  read   python robolings.py show forward_kinematics
  stuck  python robolings.py hint forward_kinematics
```

`watch` 是最舒服的用法：一个终端里开着它，另一个终端里改代码，每次存盘都会自动重测你刚改的那道题。

卡住了先看 `hint` 的提示，实在不行再看 `solutions/` 里的参考解。`python robolings.py reset slerp` 会丢掉你的作答，把那道题恢复成空白。

### 在 GitHub 上看进度

![progress](../../raw/progress/progress.svg)

每次往你的 fork 推送，都会自动跑一遍测试，在该次运行的摘要页生成进度表，并更新上面这个徽章。徽章属于你正在看的这个仓库：这里显示的是题目总数，在你自己的 fork 里就是你做完的题数。

fork 出来的仓库默认关闭 Actions，需要先到 Actions 页签手动开启一次。第一次运行之后徽章才会出现。

### 获取新题

在 GitHub 上点 **Sync fork**，或者合并 `upstream/main`。新题都是新增文件，不会动到你已经写好的答案。

## 题目

| 专题 | 题数 | 内容 |
|---|---|---|
| 旋转 | 7 | 罗德里格斯公式、四元数乘法、四元数与矩阵互转、球面插值、Kabsch 点集配准、对数与指数映射 |
| 运动学 | 7 | 齐次变换求逆、正运动学、两连杆解析逆解、几何雅可比、阻尼最小二乘、零空间投影、可操作度 |
| 灵巧手 | 6 | 耦合关节、耦合雅可比、重定向代价、带限位的指尖逆解、摩擦锥投影、力封闭判定 |
| 控制 | 9 | 最小 jerk 轨迹、梯形速度规划、低通滤波、MIT 模式阻抗控制、指令限速、角度回绕、抗积分饱和 PID、重力补偿、导纳控制 |
| 模仿学习 | 7 | 归一化、动作分块、DCT 分词、相对动作、时间集成、观测历史、权重 EMA |
| 扩散策略 | 5 | 余弦调度、前向加噪、DDPM 单步、DDIM 单步、无分类器引导 |
| 流匹配 | 3 | 训练目标、欧拉采样、中点法采样 |
| 感知 | 6 | 针孔投影、深度图反投影、镜头畸变与去畸变、体素下采样、平面拟合、ICP 单步 |
| 状态估计 | 5 | 互补滤波、陀螺仪姿态积分、EKF 预测步、Joseph 形式的 EKF 更新步、新息门限剔野 |

### 题目的出处

每条线都是某篇论文或某本教材里你平时会一眼扫过去的那一小块。正在读其中哪一本，就先做对应的题。

| 专题 | 配合着读 |
|---|---|
| 旋转、运动学 | Lynch 与 Park，[Modern Robotics](https://modernrobotics.org)（免费），第 3、4、5、6 章 |
| 灵巧手 | Murray、Li 与 Sastry，[A Mathematical Introduction to Robotic Manipulation](https://www.cds.caltech.edu/~murray/mlswiki/)（免费），第 5 章抓取 |
| 模仿学习 | [ACT](https://arxiv.org/abs/2304.13705) 的动作分块与时间集成，[FAST](https://arxiv.org/abs/2501.09747) 的 DCT 动作分词 |
| 扩散策略 | [Diffusion Policy](https://arxiv.org/abs/2303.04137)，以及它所依赖的 [DDPM](https://arxiv.org/abs/2006.11239)、[DDIM](https://arxiv.org/abs/2010.02502)、[余弦调度](https://arxiv.org/abs/2102.09672) 和 [无分类器引导](https://arxiv.org/abs/2207.12598) |
| 流匹配 | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)，以及 [π0](https://arxiv.org/abs/2410.24164)，它的动作头就是第 07 专题里的采样器 |
| 感知、状态估计 | 任何一本计算机视觉或估计理论教材都有。题目把教材留给读者的约定（OpenCV 相机坐标轴、Joseph 形式）钉死了 |

## 路线

每条线现在的题数都不是终点。按要紧程度排，缺口是：

- **流匹配**（3 题）是最薄的一条，也是最不好扩的——容易和扩散那条重复。
- **控制**（9 题）每个关节都是单独处理的。没有一题通过模型把关节耦合起来，也没有一题在规划时绕开关节限位。
- **模仿学习**（7 题）止步于数据管线，没有一题是评估训练好的策略的。

- **感知**（6 题）止步于一步 ICP，没有任何学习的成分，也没有多相机。
- **状态估计**（5 题）把 EKF 拆成两半，但没有在一个真实模型上跑完整的滤波，也没有零偏估计和误差状态滤波。

想开新的一条线，先开 issue 说一声——一条线的形状比里面任何单独一题都重要。

标了 [`new exercise`](https://github.com/12-Twelve-12/robolings/labels/new%20exercise) 的 issue 是还没人动手的点子。

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
