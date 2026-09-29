"""Generate ``exercises/`` from ``solutions/``.

Every solution marks the part a learner has to write::

    # >>> solution
    ...
    # <<< solution

This script replaces each marked block with ``raise NotImplementedError``
and writes the result to ``exercises/``. The docstrings are carried over
unchanged, so the problem statement has a single source.

    python tools/make_exercises.py            write exercises/
    python tools/make_exercises.py --check    verify, write nothing

``--check`` is for the upstream repository only. In a learner's fork the
exercises are meant to differ from the generated stubs.
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OPEN = "# >>> solution"
CLOSE = "# <<< solution"


def strip_solutions(text, name):
    out = []
    blocks = 0
    inside = False
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if stripped == OPEN:
            if inside:
                raise ValueError(f"{name}:{number}: nested '{OPEN}'")
            inside = True
            blocks += 1
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f'{indent}raise NotImplementedError("TODO: write this function")')
        elif stripped == CLOSE:
            if not inside:
                raise ValueError(f"{name}:{number}: '{CLOSE}' without '{OPEN}'")
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError(f"{name}: '{OPEN}' is never closed")
    return "\n".join(out), blocks


def python_files(folder):
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*.py")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    registry = json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))
    expected = {f"{e['track']}/{e['name']}.py": len(e["functions"]) for e in registry}
    if len(expected) != len(registry):
        print("exercises.json: two entries share a file")
        return 1

    found = python_files(ROOT / "solutions")
    if found != set(expected):
        print(f"solutions/ and exercises.json differ: {sorted(found ^ set(expected))}")
        return 1

    # Build everything first, write only if every file is consistent.
    generated = {}
    for name in sorted(expected):
        text, blocks = strip_solutions((ROOT / "solutions" / name).read_text(encoding="utf-8"), name)
        if blocks != expected[name]:
            print(f"{name}: {blocks} solution blocks, exercises.json lists {expected[name]} functions")
            return 1
        generated[name] = text

    stale = []
    for name, text in generated.items():
        target = ROOT / "exercises" / name
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != text:
            stale.append(name)
            if not args.check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8", newline="\n")

    leftover = sorted(python_files(ROOT / "exercises") - set(expected))

    if args.check:
        if stale or leftover:
            if stale:
                print("out of date: " + ", ".join(stale))
            if leftover:
                print("not in exercises.json: " + ", ".join(leftover))
            return 1
        print(f"exercises/ matches solutions/ ({len(generated)} files)")
        return 0

    for name in leftover:
        (ROOT / "exercises" / name).unlink()
    print(f"wrote {len(stale)} of {len(generated)} files, removed {len(leftover)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
