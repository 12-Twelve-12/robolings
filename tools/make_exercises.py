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


def expected_blocks():
    registry = json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))
    counts = {}
    for entry in registry:
        counts[entry["module"]] = counts.get(entry["module"], 0) + len(entry["functions"])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = expected_blocks()
    sources = sorted((ROOT / "solutions").glob("t*.py"))
    found = {path.stem for path in sources}
    if found != set(expected):
        print(f"modules differ: solutions/ has {sorted(found)}, exercises.json has {sorted(expected)}")
        return 1

    # Build everything first, write only if every file is consistent.
    generated = {}
    for path in sources:
        text, blocks = strip_solutions(path.read_text(encoding="utf-8"), path.name)
        if blocks != expected[path.stem]:
            print(f"{path.name}: {blocks} solution blocks, exercises.json lists {expected[path.stem]}")
            return 1
        generated[path.name] = text

    stale = []
    for name, text in generated.items():
        target = ROOT / "exercises" / name
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != text:
            stale.append(name)
            if not args.check:
                target.write_text(text, encoding="utf-8", newline="\n")

    if args.check:
        if stale:
            print("out of date: " + ", ".join(stale))
            return 1
        print(f"exercises/ matches solutions/ ({len(generated)} files)")
        return 0

    print(f"wrote {len(stale)} of {len(generated)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
