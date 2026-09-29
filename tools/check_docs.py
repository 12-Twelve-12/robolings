"""Check that the docs agree with the exercises.

1. Both READMEs state the number of exercises and have one table row per
   track with the number of exercises in it.
2. Every Chinese statement in docs/zh/ belongs to an exercise and was
   written for the current English text. Each file starts with a stamp,
   a hash of the English docstrings it was translated from.

An exercise without a Chinese statement is fine. One whose English text
changed after the translation is not.

    python tools/check_docs.py            check
    python tools/check_docs.py --stamp    re-stamp after updating the Chinese text
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import robolings  # noqa: E402

# file -> pattern that captures the number of exercises stated in the intro
READMES = {
    "README.md": r"^(\d+) small functions",
    "README.zh-CN.md": r"^一共 (\d+) 个",
}

STAMP = re.compile(r"^<!-- en: ([0-9a-f]{12}) -->\n")


def english_hash(entry):
    text = robolings.english_statement(entry)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def check_readmes(registry):
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
    return problems


def check_zh(registry, stamp):
    problems = []
    known = {}
    for entry in registry:
        known[robolings.zh_file(entry)] = entry

    folder = ROOT / "docs" / "zh"
    for path in sorted(folder.rglob("*.md")) if folder.exists() else []:
        if path not in known:
            problems.append(f"{path.relative_to(ROOT).as_posix()}: no such exercise")

    translated = 0
    for path, entry in known.items():
        if not path.exists():
            continue
        translated += 1
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        current = english_hash(entry)
        found = STAMP.match(text)

        if stamp:
            body = text[found.end() :] if found else text
            path.write_text(f"<!-- en: {current} -->\n" + body, encoding="utf-8", newline="\n")
            continue

        if found is None:
            problems.append(f"{rel}: no stamp on the first line")
        elif found.group(1) != current:
            problems.append(
                f"{rel}: the English text changed after this was written. Update the Chinese text and run "
                f"'python tools/check_docs.py --stamp', or delete the file if you cannot."
            )
        for name in entry["functions"]:
            if f"{name}(" not in text:
                problems.append(f"{rel}: does not mention {name}()")
        if f"exercises/{entry['track']}/{entry['name']}.py" not in text:
            problems.append(f"{rel}: does not give the path of the exercise")
    return problems, translated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", action="store_true")
    args = parser.parse_args()

    registry = json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))

    zh_problems, translated = check_zh(registry, args.stamp)
    if args.stamp:
        print(f"stamped {translated} Chinese statements")
        return 0

    problems = check_readmes(registry) + zh_problems
    for problem in problems:
        print(problem)
    if not problems:
        tracks = len({e["track"] for e in registry})
        print(f"READMEs agree with exercises.json ({len(registry)} exercises in {tracks} tracks)")
        print(f"Chinese statements are up to date ({translated} of {len(registry)})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
