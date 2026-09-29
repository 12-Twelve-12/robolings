"""robolings - small exercises for robot learning.

    python robolings.py              show progress and the next exercise
    python robolings.py run 07       run the tests of exercise 07
    python robolings.py list         list every exercise
    python robolings.py --markdown   progress as Markdown, for CI summaries

Maintainers:

    python robolings.py --target solutions --expect all-pass
    python robolings.py --target exercises --expect all-fail
"""

import argparse
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
    """Status per exercise: 'done', 'todo' or 'untested'."""
    known = {entry["test"] for entry in registry}
    unknown = sorted(set(outcomes) - known)
    if unknown:
        raise SystemExit("test classes missing from exercises.json: " + ", ".join(unknown))
    status = {}
    for entry in registry:
        results = outcomes.get(entry["test"], [])
        if not results:
            status[entry["id"]] = "untested"
        elif all(r == "passed" for r in results):
            status[entry["id"]] = "done"
        else:
            status[entry["id"]] = "todo"
    return status


def describe(entry):
    functions = ", ".join(f"{name}()" for name in entry["functions"])
    return f"{entry['id']}  {entry['title']}  [{entry['module']}.py: {functions}]"


def print_progress(registry, status, target):
    done = sum(1 for s in status.values() if s == "done")
    total = len(registry)
    width = 30
    filled = round(width * done / total)
    print(f"robolings  [{'#' * filled}{'.' * (width - filled)}]  {done}/{total}")
    print()
    for entry in registry:
        mark = {"done": "x", "todo": " ", "untested": "?"}[status[entry["id"]]]
        print(f"  [{mark}] {describe(entry)}")
    print()
    nxt = next((e for e in registry if status[e["id"]] != "done"), None)
    if nxt is None:
        print("All exercises pass. Well done.")
        return
    print(f"Next: exercise {nxt['id']}, {nxt['title']}")
    print(f"  edit   {target}/{nxt['module']}.py")
    print(f"  check  python robolings.py run {nxt['id']}")


def print_markdown(registry, status):
    done = sum(1 for s in status.values() if s == "done")
    print(f"## robolings progress: {done}/{len(registry)}")
    print()
    print("| | Exercise | File |")
    print("|---|---|---|")
    for entry in registry:
        mark = {"done": "✅", "todo": "⬜", "untested": "❓"}[status[entry["id"]]]
        print(f"| {mark} | {entry['id']} {entry['title']} | `{entry['module']}.py` |")


def run_one(registry, exercise_id, target):
    import pytest

    wanted = exercise_id.zfill(2)
    entry = next((e for e in registry if e["id"] == wanted), None)
    if entry is None:
        raise SystemExit(f"no exercise {exercise_id!r}; see 'python robolings.py list'")
    os.environ["ROBOLINGS_TARGET"] = target
    print(describe(entry))
    node = f"{ROOT / 'tests' / ('test_' + entry['module'] + '.py')}::{entry['test']}"
    return int(pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", node]))


def main():
    parser = argparse.ArgumentParser(description="Small exercises for robot learning.")
    parser.add_argument("command", nargs="?", default="progress", choices=["progress", "run", "list"])
    parser.add_argument("exercise", nargs="?")
    parser.add_argument("--target", default="exercises", choices=["exercises", "solutions"])
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--expect", choices=["all-pass", "all-fail"])
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    registry = load_registry()

    if args.command == "list":
        for entry in registry:
            print(describe(entry))
        return 0

    if args.command == "run":
        if not args.exercise:
            raise SystemExit("usage: python robolings.py run <id>")
        return run_one(registry, args.exercise, args.target)

    status = grade(registry, run_all(args.target))
    if args.markdown:
        print_markdown(registry, status)
    else:
        print_progress(registry, status, args.target)

    if args.expect:
        wanted = "done" if args.expect == "all-pass" else "todo"
        wrong = [entry["id"] for entry in registry if status[entry["id"]] != wanted]
        if wrong:
            print(f"\nexpected {args.expect} in {args.target}/, but these differ: {', '.join(wrong)}")
            return 1
        print(f"\n{args.target}/: {args.expect} holds for all {len(registry)} exercises")
    return 0


if __name__ == "__main__":
    sys.exit(main())
