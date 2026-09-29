"""Check that the links between files in this repository resolve.

Only links to paths inside the repository are checked. They are the ones that
break when a file is renamed, and checking them needs no network, so it gives
the same answer on every machine and in CI.

External links are listed but not fetched. A CI job that reaches out to the
web fails for reasons that have nothing to do with the change under test.

    python tools/check_links.py
    python tools/check_links.py --list-external
"""

import argparse
import pathlib
import re
import sys
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}

# [text](target) but not ![image](target) handled separately; both are checked
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
EXTERNAL = re.compile(r"^(https?:|mailto:|#)")


def markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        if SKIP_DIRS.isdisjoint(part for part in path.relative_to(ROOT).parts):
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-external", action="store_true")
    args = parser.parse_args()

    problems = []
    external = set()
    checked = 0

    for path in markdown_files():
        rel = path.relative_to(ROOT).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            for target in LINK.findall(line):
                if EXTERNAL.match(target):
                    external.add(target)
                    continue
                # ../../raw/<branch>/<file> is resolved by GitHub, not on disk
                if "/raw/" in target:
                    external.add(target)
                    continue
                cleaned = urllib.parse.unquote(target.split("#", 1)[0])
                if not cleaned:
                    continue
                checked += 1
                if not (path.parent / cleaned).resolve().exists():
                    problems.append(f"{rel}:{line_number}: {target} does not exist")

    if args.list_external:
        for target in sorted(external):
            print(target)
        return 0

    for problem in problems:
        print(problem)
    if not problems:
        print(f"{checked} links between files resolve, {len(external)} external links not checked")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
