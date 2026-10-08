"""Run every check CI runs, in order, and stop at the first one that fails.

    python tools/check_all.py
    python tools/check_all.py --from mutants     skip the checks before that one

Each check is the exact command from the workflow files, so what passes here
passes in CI. The names on the left are what ``--from`` accepts.
"""

import argparse
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (name, the command as written in .github/workflows/*.yml)
CHECKS = [
    ("lint", "ruff check ."),
    ("format", "ruff format --check ."),
    ("stubs", "python tools/make_exercises.py --check"),
    ("solutions", "python robolings.py --target solutions --expect all-pass"),
    ("exercises", "python robolings.py --target exercises --expect all-fail"),
    ("mutants", "python tools/mutants.py"),
    ("docs", "python tools/check_docs.py"),
    ("links", "python tools/check_links.py"),
    ("tools", "python -m pytest tools -q -p no:cacheprovider"),
]


def argv_for(command):
    """The CI command line as a list for this interpreter.

    ``ruff`` is run as ``python -m ruff`` so that the pip-installed one is
    found even when it is not on PATH, and ``python`` is this interpreter.
    """
    words = shlex.split(command)
    if words[0] == "ruff":
        return [sys.executable, "-m", *words]
    if words[0] == "python":
        argv = [sys.executable, *words[1:]]
        if words[1] == "robolings.py":
            argv.append("--brief")
        return argv
    raise ValueError(f"unexpected command: {command}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from", dest="start", choices=[name for name, _ in CHECKS], help="first check to run"
    )
    args = parser.parse_args()

    names = [name for name, _ in CHECKS]
    todo = CHECKS[names.index(args.start) :] if args.start else CHECKS
    width = max(len(name) for name in names)
    for name, command in todo:
        started = time.monotonic()
        result = subprocess.run(argv_for(command), cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        elapsed = time.monotonic() - started
        if result.returncode == 0:
            last = (result.stdout.strip().splitlines() or [""])[-1]
            print(f"ok    {name:<{width}}  {elapsed:5.1f}s  {last}")
            continue
        print(f"FAIL  {name:<{width}}  {elapsed:5.1f}s  {command}")
        print()
        print((result.stdout + result.stderr).rstrip())
        print()
        print(f"fix this, then: python tools/check_all.py --from {name}")
        return result.returncode
    print(f"all {len(todo)} checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
