# robolings

Exercises for robot learning, in the style of [rustlings](https://github.com/rust-lang/rustlings).

31 small functions to implement, starting at Rodrigues' formula and ending at the sampler of a flow-matching policy. NumPy only, no GPU, no simulator. The tests run in under a second.

[中文说明](README.zh-CN.md) (the problem statements are available in Chinese too)

## Usage

Fork the repo and clone your fork, then:

```bash
pip install -r requirements.txt
python robolings.py           # progress, and what to do next
python robolings.py run slerp # test a single exercise
python robolings.py show slerp # print the problem statement
python robolings.py list
```

The stubs are in `exercises/`, one file per exercise. Replace the `raise NotImplementedError` and run the exercise again. Python 3.10+.

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

Answers are in `solutions/` if you get stuck.

### Progress on GitHub

![progress](../../raw/progress/progress.svg)

Each push to your fork runs the tests, puts a progress table in the summary of the workflow run, and updates this badge. The badge belongs to the repository you are looking at. Here it says 0, in your fork it shows your own count.

Forks have Actions disabled by default, so enable them once in the Actions tab. The badge appears after the first run.

### Getting new exercises

Use **Sync fork** on GitHub, or merge `upstream/main`. New exercises arrive as new files, so your own answers are not touched.

## Exercises

| Track | Exercises | Content |
|---|---|---|
| Rotations | 6 | Rodrigues, quaternion product, quaternion ↔ matrix, slerp, Kabsch |
| Kinematics | 4 | transform inverse, FK, geometric Jacobian, damped least squares |
| Dexterous hands | 3 | mimic joints, retargeting cost, fingertip IK with joint limits |
| Control | 6 | min-jerk, low-pass filter, MIT-mode impedance control, rate limiter, angle wrapping, PID with anti-windup |
| Imitation learning | 5 | normalisation, action chunking, relative actions, temporal ensembling, obs history |
| Diffusion policy | 4 | cosine schedule, forward process, DDPM step, DDIM step |
| Flow matching | 3 | training target, Euler sampler, midpoint sampler |

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
