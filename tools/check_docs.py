"""Check that both READMEs agree with exercises.json.

The intro states the number of exercises, and the table has one row per
track with the number of exercises in it.

    python tools/check_docs.py
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# file -> pattern that captures the number of exercises stated in the intro
READMES = {
    "README.md": r"^(\d+) small functions",
    "README.zh-CN.md": r"^一共 (\d+) 个",
}


def main():
    registry = json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))
    per_track = {}
    for entry in registry:
        per_track[entry["track"]] = per_track.get(entry["track"], 0) + 1
    expected = list(per_track.values())

    problems = []
    for name, pattern in READMES.items():
        text = (ROOT / name).read_text(encoding="utf-8")

        stated = re.search(pattern, text, flags=re.MULTILINE)
        if stated is None:
            problems.append(f"{name}: cannot find the sentence that states the number of exercises")
        elif int(stated.group(1)) != len(registry):
            problems.append(f"{name}: says {stated.group(1)} exercises, exercises.json has {len(registry)}")

        listed = [int(n) for n in re.findall(r"^\| [^|]+ \| (\d+) \| ", text, flags=re.MULTILINE)]
        if listed != expected:
            problems.append(f"{name}: table has {listed} exercises per track, exercises.json has {expected}")

    for problem in problems:
        print(problem)
    if not problems:
        print(f"READMEs agree with exercises.json ({len(registry)} exercises in {len(expected)} tracks)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
