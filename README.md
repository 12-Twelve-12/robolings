# robolings

Exercises for robot learning, in the style of [rustlings](https://github.com/rust-lang/rustlings).

36 small functions to implement, starting at Rodrigues' formula and ending at the sampler of a flow-matching policy. NumPy only, no GPU, no simulator. The tests run in under a second.

[中文说明](README.zh-CN.md) (the problem statements are available in Chinese too)

## Usage

Fork the repo and clone your fork, then:

```bash
pip install -r requirements.txt
python robolings.py           # progress, and what to do next
python robolings.py watch     # rerun an exercise every time you save it
python robolings.py run slerp # test a single exercise
python robolings.py show slerp # print the problem statement
python robolings.py hint slerp # a nudge if you are stuck
python robolings.py list
```

The stubs are in `exercises/`, one file per exercise. Replace the `raise NotImplementedError` and run the exercise again. Python 3.10+.

```text
robolings  [######........................]  6/36

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

`watch` is the comfortable way to work: leave it running in one terminal, edit in another, and every save reruns the exercise you just touched.

Stuck? `hint` gives you a nudge, and the answers are in `solutions/`. `python robolings.py reset slerp` throws your attempt away and starts that exercise over.

### Progress on GitHub

![progress](../../raw/progress/progress.svg)

Each push to your fork runs the tests, puts a progress table in the summary of the workflow run, and updates this badge. The badge belongs to the repository you are looking at. Here it says 0, in your fork it shows your own count.

Forks have Actions disabled by default, so enable them once in the Actions tab. The badge appears after the first run.

### Getting new exercises

Use **Sync fork** on GitHub, or merge `upstream/main`. New exercises arrive as new files, so your own answers are not touched.

## Exercises

| Track | Exercises | Content |
|---|---|---|
| Rotations | 7 | Rodrigues, quaternion product, quaternion ↔ matrix, slerp, Kabsch, log and exp |
| Kinematics | 4 | transform inverse, FK, geometric Jacobian, damped least squares |
| Dexterous hands | 3 | mimic joints, retargeting cost, fingertip IK with joint limits |
| Control | 8 | min-jerk, trapezoidal profile, low-pass filter, MIT-mode impedance control, rate limiter, angle wrapping, PID with anti-windup, gravity compensation |
| Imitation learning | 7 | normalisation, action chunking, DCT tokenisation, relative actions, temporal ensembling, obs history, weight EMA |
| Diffusion policy | 4 | cosine schedule, forward process, DDPM step, DDIM step |
| Flow matching | 3 | training target, Euler sampler, midpoint sampler |

## Roadmap

The tracks are not meant to stay the size they are now. The gaps, roughly in the order they matter:

- **Dexterous hands** (3) is the thinnest track. Grasp analysis and contact mechanics are missing entirely.
- **Kinematics** (4) has no closed-form IK, and nothing on redundancy or singularities.
- **Diffusion policy** (4) and **flow matching** (3) are missing the knobs that matter at sampling time.

Two directions have no track at all yet: perception (camera projection, depth back-projection, point-cloud registration) and state estimation (complementary filter, IMU attitude, one EKF step). Neither is avoidable in a real stack. If you want to start one, open an issue first — the shape of a track matters more than any single exercise in it.

Issues labelled [`new exercise`](https://github.com/12-Twelve-12/robolings/labels/new%20exercise) are ideas nobody has started.

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
