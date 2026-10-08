# Contributing

Bug reports: please give the exercise number and the code you ran. The most useful reports are "this correct answer is rejected" and "this wrong answer passes".

## How changes get in

`main` is protected, so everything goes through a pull request:

1. Fork, create a branch, push to your fork.
2. Open a PR against `main`. CI has to be green.
3. I review it. PRs are squash-merged, so don't worry about tidying the commit history.

Keep your solved exercises out of the PR. If you have been working through the exercises in the same fork, branch off a clean `upstream/main` instead of your own `main`.

Issues labelled `new exercise` are ideas nobody is working on yet. Leave a comment before you start so two people don't write the same one.

## Adding an exercise

1. Write the function in its own file, `solutions/<track>/<name>.py`. The docstring is the problem statement. Put the body between markers:

   ```python
   def my_function(x):
       """Problem statement."""
       # >>> solution
       return x
       # <<< solution
   ```

2. Add an entry to `exercises.json`. The position in the file is the order learners see. Names are permanent, positions are not.
3. Add a test class to `tests/test_<track>.py`, named as in the entry. Don't call other exercises from a test; shared reference code goes in `tests/helpers.py`.
4. Add at least one wrong answer to `tools/wrong_answers/<track>.py`. `python tools/mutants.py` runs them all.
5. Add a hint to `hints.json`, in both languages, in the same position as in `exercises.json`. A hint points at the idea, it does not contain the answer.
6. Regenerate the stubs with `python tools/make_exercises.py`. Don't edit `exercises/` by hand.
7. Update the table and the exercise count in both READMEs.
8. A Chinese statement in `docs/zh/<track>/<name>.md` is optional. If you don't write Chinese, leave it out and I'll add it. The hint is not optional, but a rough translation is fine and I will tidy it.

## Before opening a PR

```bash
pip install -r requirements-dev.txt
python tools/check_all.py
```

That runs, in order: ruff, the stub generator in check mode, the solutions (all must pass), the stubs (all must fail), the wrong answers, the docs check, the link check and the tests of the tooling. It stops at the first failure and tells you how to resume from there. CI runs the same commands on Linux, macOS and Windows.

If you change the docstring of an existing exercise, CI will tell you that its Chinese statement is out of date. Update it and run `python tools/check_docs.py --stamp`, or delete the file and say so in the PR.

Don't rename or move an existing exercise. People have answers in their forks under the current path.

Exercises should need nothing but NumPy and run in well under a second.
