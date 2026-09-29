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

1. Write the function in a module under `solutions/`. The docstring is the problem statement. Put the body between markers:

   ```python
   def my_function(x):
       """Problem statement."""
       # >>> solution
       return x
       # <<< solution
   ```

2. Add an entry to `exercises.json`.
3. Add a test class in `tests/`, named as in the entry. Don't call other exercises from a test; shared reference code goes in `tests/helpers.py`.
4. Add at least one wrong answer to `tools/mutants.py`.
5. Regenerate the stubs with `python tools/make_exercises.py`. Don't edit `exercises/` by hand.
6. Update the table and the exercise count in both READMEs.

## Before opening a PR

```bash
pip install -r requirements-dev.txt

ruff check . && ruff format --check .
python tools/make_exercises.py --check
python robolings.py --target solutions --expect all-pass
python robolings.py --target exercises --expect all-fail
python tools/mutants.py
python tools/check_docs.py
```

CI runs the same commands on Linux, macOS and Windows.

Exercises should need nothing but NumPy and run in well under a second.
