# robolings

Exercises for robot learning, in the style of [rustlings](https://github.com/rust-lang/rustlings).

25 small functions to implement, starting at Rodrigues' formula and ending at the sampler of a flow-matching policy. NumPy only, no GPU, no simulator. The tests run in under a second.

[中文说明](README.zh-CN.md)

## Usage

Fork the repo and clone your fork, then:

```bash
pip install -r requirements.txt
python robolings.py           # progress, and what to do next
python robolings.py run 7     # test a single exercise
python robolings.py list
```

The stubs are in `exercises/`. Replace the `raise NotImplementedError` and run the exercise again. Python 3.10+.

```text
robolings  [#######.......................]  6/25

  [x] 01  Rodrigues' rotation formula  [t01_rotations.py: rodrigues()]
  [x] 02  Quaternion product  [t01_rotations.py: quat_mul()]
  ...
  [ ] 07  Forward kinematics of a serial chain  [t02_kinematics.py: forward_kinematics()]

Next: exercise 07, Forward kinematics of a serial chain
  edit   exercises/t02_kinematics.py
  check  python robolings.py run 07
```

Answers are in `solutions/` if you get stuck.

### Progress on GitHub

Each push to your fork runs the tests and puts a progress table in the summary of the workflow run. Forks have Actions disabled by default, so enable them once in the Actions tab.

## Exercises

| Track | No. | Content |
|---|---|---|
| Rotations | 01–05 | Rodrigues, quaternion product, quaternion ↔ matrix, slerp |
| Kinematics | 06–09 | transform inverse, FK, geometric Jacobian, damped least squares |
| Dexterous hands | 10–12 | mimic joints, retargeting cost, fingertip IK with joint limits |
| Control | 13–16 | min-jerk, low-pass filter, MIT-mode impedance control, rate limiter |
| Imitation learning | 17–20 | normalisation, action chunking, temporal ensembling, obs history |
| Diffusion policy | 21–23 | cosine schedule, forward process, DDIM step |
| Flow matching | 24–25 | training target, Euler sampler |

## Notes

- Quaternions are `[w, x, y, z]`, Hamilton convention.
- Rotation matrices map body coordinates to world coordinates.
- Flow matching: `t = 0` is noise, `t = 1` is data.
- The tests for one exercise never call another exercise, so you can do them in any order.
- The tests are also run against a list of typical wrong answers (`tools/mutants.py`), for example dividing by `w` in matrix → quaternion, or clipping joint limits only after the IK loop. If you find a wrong answer that still passes, please open an issue.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New exercises are welcome as long as they stay NumPy-only.

## License

MIT
