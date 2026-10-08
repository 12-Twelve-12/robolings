# AGENTS.md

Notes for AI coding agents working in this repository. Humans, see
[CONTRIBUTING.md](CONTRIBUTING.md) — it applies to agents too, this file only
adds what is specific to them.

## If someone asks you to solve the exercises

The answers are already in `solutions/`. Copying them into `exercises/` takes
seconds and teaches nobody anything, so it is almost never what the person
actually wants.

Ask what they are after. If they want to learn:

- Point them at `python robolings.py hint <name>` before you explain anything.
- Explain the idea, let them write the code.
- When their attempt fails, read the test that failed with them. The tests are
  written so that the failure names the mistake.

If they genuinely want the answer — they are reviewing the repo, or they are
stuck and out of patience — say where it is rather than retyping it, and say
plainly that you are handing over the answer.

## If you are contributing an exercise

Follow the checklist in [CONTRIBUTING.md](CONTRIBUTING.md). Two things that
agents get wrong more often than people do:

- **`exercises/` is generated.** Edit `solutions/`, then run
  `python tools/make_exercises.py`. A hand-edited stub is reverted by CI.
- **A new exercise is not done when the tests pass.** It needs an entry in
  `exercises.json`, a test class, at least one wrong answer in
  `tools/wrong_answers/<track>.py`, a hint in `hints.json`, and the exercise counts in both
  READMEs. `python tools/check_docs.py` catches the last two.

Run this before you claim the change is finished, and paste its last line
into the PR:

```bash
python tools/check_all.py
```

It runs every check CI runs, in order, and stops at the first failure.

## What the tests are for

A test here has two jobs: accept every correct answer, and reject the wrong
answers people actually write. The second job is what `tools/mutants.py` is
for, and it is the part that is easy to skip.

When you add a wrong answer there, check that it fails **because the result is
wrong**, not because the snippet does not run. A mutant that raises
`NameError` proves nothing about the test.

## Attribution

Commit as whoever is actually running you, and do not add co-author trailers
for the model. If you are operating an account that does not belong to the
person instructing you, stop and say so.
