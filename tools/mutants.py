"""Check that the tests reject plausible wrong answers.

Passing the reference solution only shows the tests are satisfiable. Each
mutant in ``tools/wrong_answers/<track>.py`` is a mistake a learner could
reasonably make. The tests of the exercise it belongs to must fail for every
one of them.

A mutant counts as rejected only when at least one test fails on an
assertion. A mutant that only blows up with a ``NameError`` or an
``IndexError`` has not shown that the tests check the result.

    python tools/mutants.py
    python tools/mutants.py --jobs 2    limit the number of pytest processes
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_wrong_answers():
    """Every (exercise, function, what is wrong, replacement source) tuple.

    They live in ``tools/wrong_answers/<track>.py``, one file per track, each
    with a ``MUTANTS`` list. Order: by file name, then as written.
    """
    folder = ROOT / "tools" / "wrong_answers"
    found = []
    for path in sorted(folder.glob("*.py")):
        namespace = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        found.extend(namespace["MUTANTS"])
    return found


MUTANTS = load_wrong_answers()


# "path:line: assert ..." or "path:line: AssertionError" in the --tb=line output
FAILURE_LINE = re.compile(r"^.+?:\d+: (.*)$")


def failure_kinds(output):
    """What each failed test raised, from pytest's ``--tb=line`` output."""
    kinds = []
    for line in output.splitlines():
        found = FAILURE_LINE.match(line)
        if found is None:
            continue
        text = found.group(1)
        if text.startswith("assert") or text.startswith("AssertionError"):
            kinds.append("assertion")
        else:
            kinds.append(text.split(":", 1)[0].split("(", 1)[0].strip() or "exception")
    return kinds


def run_mutant(number, name, function, source, entry, tmp):
    target = pathlib.Path(tmp) / f"m{number:03d}"
    shutil.copytree(ROOT / "solutions", target)
    module = target / entry["track"] / f"{name}.py"
    original = module.read_text(encoding="utf-8")
    if f"def {function}(" not in original:
        return f"mutant {number}: {function} not found in {module.name}", []
    # A later definition replaces the earlier one at import time.
    module.write_text(original + "\n\n" + source.lstrip("\n"), encoding="utf-8", newline="\n")

    node = f"{ROOT / 'tests' / ('test_' + entry['track'] + '.py')}::{entry['test']}"
    env = dict(os.environ, ROBOLINGS_TARGET=str(target))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider", node],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    # 1 = tests ran and at least one failed. Anything else is not a kill.
    if result.returncode != 1:
        return f"pytest exit code {result.returncode}", []
    return None, failure_kinds(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 1), help="pytest processes to run at once"
    )
    args = parser.parse_args()

    registry = {e["name"]: e for e in json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))}

    unknown = sorted({m[0] for m in MUTANTS} - set(registry))
    if unknown:
        print("mutants for exercises that do not exist: " + ", ".join(unknown))
        return 1
    missing = [name for name in registry if name not in {m[0] for m in MUTANTS}]
    if missing:
        print("exercises without any mutant: " + ", ".join(missing))
        return 1

    for number, (name, function, _label, source) in enumerate(MUTANTS):
        if function not in registry[name]["functions"]:
            print(f"mutant {number}: {function} is not a function of {name}")
            return 1
        if f"def {function}(" not in source:
            print(f"mutant {number}: replacement does not define {function}")
            return 1

    survived = []
    with tempfile.TemporaryDirectory() as tmp:

        def job(item):
            number, (name, function, label, source) = item
            return run_mutant(number, name, function, source, registry[name], tmp)

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            outcomes = list(pool.map(job, enumerate(MUTANTS)))

    for (name, function, label, _), (problem, kinds) in zip(MUTANTS, outcomes, strict=True):
        if problem is not None:
            survived.append((name, function, label, problem))
        elif "assertion" not in kinds:
            raised = ", ".join(sorted(set(kinds)))
            survived.append((name, function, label, f"only raised {raised}, no assertion failed"))

    killed = len(MUTANTS) - len(survived)
    print(f"{killed} of {len(MUTANTS)} mutants rejected by the tests of their own exercise")
    for name, function, label, why in survived:
        print(f"  NOT rejected: {name} {function}(): {label} ({why})")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
