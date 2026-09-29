# Contributing

Bug reports: please give the exercise number and the code you ran. The most useful reports are "this correct answer is rejected" and "this wrong answer passes".

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

## Before opening a PR

```bash
python tools/make_exercises.py --check
python robolings.py --target solutions --expect all-pass
python robolings.py --target exercises --expect all-fail
python tools/mutants.py
```

CI runs the same four commands.

Exercises should need nothing but NumPy and run in well under a second.
