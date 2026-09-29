"""robolings - small exercises for robot learning.

    python robolings.py               show progress and the next exercise
    python robolings.py run slerp     run the tests of one exercise
    python robolings.py run 5         same, by position in the list
    python robolings.py show slerp    print the problem statement
    python robolings.py list          list every exercise
    python robolings.py --zh          Chinese titles and statements
    python robolings.py --markdown    progress as Markdown, for CI summaries

Set ROBOLINGS_LANG=zh to make --zh the default.

Maintainers:

    python robolings.py --target solutions --expect all-pass
    python robolings.py --target exercises --expect all-fail
"""

import argparse
import ast
import contextlib
import io
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def load_registry():
    return json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))


class Collector:
    """Counts test outcomes per test class."""

    def __init__(self):
        self.outcomes = {}

    def pytest_runtest_logreport(self, report):
        failed_outside_call = report.when != "call" and report.outcome != "passed"
        if report.when == "call" or failed_outside_call:
            parts = report.nodeid.split("::")
            if len(parts) >= 3:
                self.outcomes.setdefault(parts[1], []).append(report.outcome)


def run_all(target):
    import pytest

    os.environ["ROBOLINGS_TARGET"] = target
    collector = Collector()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        code = pytest.main(
            ["-q", "--tb=no", "-p", "no:cacheprovider", str(ROOT / "tests")],
            plugins=[collector],
        )
    # 0 = all passed, 1 = some failed. Anything else means pytest itself broke.
    if int(code) not in (0, 1):
        print(sink.getvalue())
        raise SystemExit(f"pytest could not run the tests (exit code {int(code)})")
    return collector.outcomes


def grade(registry, outcomes):
    """Status per exercise name: 'done', 'todo' or 'untested'."""
    known = {entry["test"] for entry in registry}
    unknown = sorted(set(outcomes) - known)
    if unknown:
        raise SystemExit("test classes missing from exercises.json: " + ", ".join(unknown))
    status = {}
    for entry in registry:
        results = outcomes.get(entry["test"], [])
        if not results:
            status[entry["name"]] = "untested"
        elif all(r == "passed" for r in results):
            status[entry["name"]] = "done"
        else:
            status[entry["name"]] = "todo"
    return status


def file_of(entry, target):
    return f"{target}/{entry['track']}/{entry['name']}.py"


def title_of(entry, zh):
    return entry["title_zh"] if zh else entry["title"]


def zh_file(entry):
    return ROOT / "docs" / "zh" / entry["track"] / f"{entry['name']}.md"


def english_statement(entry):
    """The problem statement as written in the docstrings.

    Read from solutions/, because a learner may have edited their own copy.
    """
    path = ROOT / "solutions" / entry["track"] / f"{entry['name']}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    parts = [ast.get_docstring(tree) or ""]
    for name in entry["functions"]:
        node = functions[name]
        signature = f"{name}({', '.join(a.arg for a in node.args.args)})"
        parts.append(signature + "\n\n" + (ast.get_docstring(node) or ""))
    return "\n\n".join(parts)


def chinese_statement(entry):
    """The Chinese statement, or None if nobody has written one yet."""
    path = zh_file(entry)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").split("\n")
    return "\n".join(line for line in lines if not line.startswith("<!--")).strip("\n")


def print_progress(registry, status, target, zh):
    done = sum(1 for s in status.values() if s == "done")
    total = len(registry)
    width = 30
    filled = round(width * done / total)
    print(f"robolings  [{'#' * filled}{'.' * (width - filled)}]  {done}/{total}")
    track = None
    for number, entry in enumerate(registry, start=1):
        if entry["track"] != track:
            track = entry["track"]
            print(f"\n  {track}")
        mark = {"done": "x", "todo": " ", "untested": "?"}[status[entry["name"]]]
        print(f"   [{mark}] {number:2d}  {entry['name']:<22} {title_of(entry, zh)}")
    print()
    nxt = next((e for e in registry if status[e["name"]] != "done"), None)
    if nxt is None:
        print("All exercises pass. Well done.")
        return
    print(f"Next: {nxt['name']}")
    print(f"  edit   {file_of(nxt, target)}")
    print(f"  check  python robolings.py run {nxt['name']}")
    if zh and zh_file(nxt).exists():
        print(f"  read   python robolings.py show {nxt['name']} --zh")


def print_markdown(registry, status, zh):
    done = sum(1 for s in status.values() if s == "done")
    print(f"## robolings progress: {done}/{len(registry)}")
    print()
    print("| | Exercise | | File |")
    print("|---|---|---|---|")
    for entry in registry:
        mark = {"done": "✅", "todo": "⬜", "untested": "❓"}[status[entry["name"]]]
        print(f"| {mark} | `{entry['name']}` | {title_of(entry, zh)} | `{file_of(entry, 'exercises')}` |")


def find(registry, key):
    if key.isdigit():
        position = int(key)
        if 1 <= position <= len(registry):
            return registry[position - 1]
        raise SystemExit(f"there are {len(registry)} exercises, got {key}")
    for entry in registry:
        if entry["name"] == key:
            return entry
    raise SystemExit(f"no exercise named {key!r}; see 'python robolings.py list'")


def run_one(registry, key, target):
    import pytest

    entry = find(registry, key)
    os.environ["ROBOLINGS_TARGET"] = target
    print(f"{entry['name']}  ({file_of(entry, target)})")
    node = f"{ROOT / 'tests' / ('test_' + entry['track'] + '.py')}::{entry['test']}"
    return int(pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", node]))


def show(registry, key, target, zh):
    entry = find(registry, key)
    text = chinese_statement(entry) if zh else None
    if zh and text is None:
        print(f"(no Chinese statement for {entry['name']} yet, showing the English one)\n")
    if text is None:
        text = f"{entry['name']}: {entry['title']}\nfile: {file_of(entry, target)}\n\n" + english_statement(
            entry
        )
    print(text)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Small exercises for robot learning.")
    parser.add_argument("command", nargs="?", default="progress", choices=["progress", "run", "show", "list"])
    parser.add_argument("exercise", nargs="?")
    parser.add_argument("--target", default="exercises", choices=["exercises", "solutions"])
    parser.add_argument("--zh", action="store_true", help="Chinese titles and statements")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--expect", choices=["all-pass", "all-fail"])
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    zh = args.zh or os.environ.get("ROBOLINGS_LANG", "").lower().startswith("zh")
    registry = load_registry()

    if args.command == "list":
        for number, entry in enumerate(registry, start=1):
            title = title_of(entry, zh)
            print(f"{number:2d}  {entry['name']:<22} {title:<45} {file_of(entry, args.target)}")
        return 0

    if args.command in ("run", "show"):
        if not args.exercise:
            raise SystemExit(f"usage: python robolings.py {args.command} <name>")
        if args.command == "run":
            return run_one(registry, args.exercise, args.target)
        return show(registry, args.exercise, args.target, zh)

    status = grade(registry, run_all(args.target))
    if args.markdown:
        print_markdown(registry, status, zh)
    else:
        print_progress(registry, status, args.target, zh)

    if args.expect:
        wanted = "done" if args.expect == "all-pass" else "todo"
        wrong = [entry["name"] for entry in registry if status[entry["name"]] != wanted]
        if wrong:
            print(f"\nexpected {args.expect} in {args.target}/, but these differ: {', '.join(wrong)}")
            return 1
        print(f"\n{args.target}/: {args.expect} holds for all {len(registry)} exercises")
    return 0


if __name__ == "__main__":
    sys.exit(main())
