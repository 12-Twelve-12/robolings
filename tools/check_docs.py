"""Check that both READMEs agree with exercises.json.

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
    ids = [entry["id"] for entry in registry]
    expected = [f"{n:02d}" for n in range(1, len(ids) + 1)]
    if ids != expected:
        print(f"exercises.json: ids must be 01..{len(ids):02d} in order, got {ids}")
        return 1

    problems = []
    for name, pattern in READMES.items():
        text = (ROOT / name).read_text(encoding="utf-8")

        stated = re.search(pattern, text, flags=re.MULTILINE)
        if stated is None:
            problems.append(f"{name}: cannot find the sentence that states the number of exercises")
        elif int(stated.group(1)) != len(ids):
            problems.append(f"{name}: says {stated.group(1)} exercises, exercises.json has {len(ids)}")

        listed = []
        for first, last in re.findall(r"\| (\d\d)–(\d\d) \|", text):
            listed += [f"{n:02d}" for n in range(int(first), int(last) + 1)]
        if listed != ids:
            missing = sorted(set(ids) - set(listed))
            extra = sorted(set(listed) - set(ids))
            problems.append(f"{name}: table does not list every exercise once (missing {missing}, extra {extra})")

    for problem in problems:
        print(problem)
    if not problems:
        print(f"READMEs agree with exercises.json ({len(ids)} exercises)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
