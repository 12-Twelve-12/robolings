"""robolings - small exercises for robot learning.

    python robolings.py               show progress and the next exercise
    python robolings.py watch         rerun an exercise every time you save it
    python robolings.py run slerp     run the tests of one exercise
    python robolings.py run 5         same, by position in the list
    python robolings.py show slerp    print the problem statement
    python robolings.py hint slerp    print a hint
    python robolings.py reset slerp   throw away your answer and start over
    python robolings.py list          list every exercise
    python robolings.py --zh          Chinese titles, statements and hints
    python robolings.py --markdown    progress as Markdown, for CI summaries
    python robolings.py --badge FILE  also write a progress badge
    python robolings.py --badge FILE --badge-count   a badge with the number of exercises

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
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent


def load_registry():
    return json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))


def load_hints():
    return json.loads((ROOT / "hints.json").read_text(encoding="utf-8"))


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
        signature = f"{name}({ast.unparse(node.args)})"
        parts.append(signature + "\n\n" + (ast.get_docstring(node) or ""))
    return "\n\n".join(parts)


def chinese_statement(entry):
    """The Chinese statement, or None if nobody has written one yet."""
    path = zh_file(entry)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").split("\n")
    return "\n".join(line for line in lines if not line.startswith("<!--")).strip("\n")


def print_progress(registry, status, target, zh, brief=False):
    done = sum(1 for s in status.values() if s == "done")
    total = len(registry)
    width = 30
    filled = round(width * done / total)
    print(f"robolings  [{'#' * filled}{'.' * (width - filled)}]  {done}/{total}")
    if not brief:
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
    suffix = " --zh" if zh else ""
    print(f"Next: {nxt['name']}")
    print(f"  edit   {file_of(nxt, target)}")
    print(f"  check  python robolings.py run {nxt['name']}")
    print(f"  read   python robolings.py show {nxt['name']}{suffix}")
    print(f"  stuck  python robolings.py hint {nxt['name']}{suffix}")


def print_markdown(registry, status, zh):
    done = sum(1 for s in status.values() if s == "done")
    print(f"## robolings progress: {done}/{len(registry)}")
    print()
    print("| | Exercise | | File |")
    print("|---|---|---|---|")
    for entry in registry:
        mark = {"done": "✅", "todo": "⬜", "untested": "❓"}[status[entry["name"]]]
        print(f"| {mark} | `{entry['name']}` | {title_of(entry, zh)} | `{file_of(entry, 'exercises')}` |")


def badge_svg(done, total, count_only=False):
    """A small "robolings | 12/31" badge, drawn here so that no outside service is needed.

    With ``count_only`` it says "robolings | 31 exercises" instead: the
    upstream repository has no progress to show, only stubs.
    """
    label = "robolings"
    if count_only:
        value, color = f"{total} exercises", "#007ec6"
    elif done == total:
        value, color = f"{done}/{total}", "#4c1"
    elif done == 0:
        value, color = f"{done}/{total}", "#9f9f9f"
    else:
        value, color = f"{done}/{total}", ("#fe7d37", "#dfb317", "#97ca00")[min(3 * done // total, 2)]
    left = 10 + round(6.5 * len(label))
    right = 10 + round(6.5 * len(value))
    width = left + right
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" '
        f'aria-label="{label}: {value}">\n'
        f"<title>{label}: {value}</title>\n"
        '<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/></linearGradient>\n'
        f'<clipPath id="r"><rect width="{width}" height="20" rx="3" fill="#fff"/></clipPath>\n'
        f'<g clip-path="url(#r)"><rect width="{left}" height="20" fill="#555"/>'
        f'<rect x="{left}" width="{right}" height="20" fill="{color}"/>'
        f'<rect width="{width}" height="20" fill="url(#s)"/></g>\n'
        '<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        'font-size="11">'
        f'<text x="{left / 2:g}" y="14">{label}</text>'
        f'<text x="{left + right / 2:g}" y="14">{value}</text></g>\n'
        "</svg>\n"
    )


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


def hint(registry, key, zh):
    entry = find(registry, key)
    hints = load_hints()
    if entry["name"] not in hints:
        print(f"no hint for {entry['name']} yet")
        return 0
    print(hints[entry["name"]]["zh" if zh else "en"])
    return 0


def reset(registry, key, target, force):
    """Overwrite one exercise with a fresh stub."""
    sys.path.insert(0, str(ROOT / "tools"))
    import make_exercises

    entry = find(registry, key)
    path = ROOT / file_of(entry, target)
    stub, _ = make_exercises.strip_solutions(
        (ROOT / "solutions" / entry["track"] / f"{entry['name']}.py").read_text(encoding="utf-8"),
        entry["name"],
    )
    if path.exists() and path.read_text(encoding="utf-8") == stub:
        print(f"{file_of(entry, target)} is already a fresh stub")
        return 0
    if not force:
        print(f"This throws away your answer in {file_of(entry, target)}.")
        try:
            answer = input("Type 'yes' to continue: ")
        except EOFError:
            # not a terminal, so nobody can answer
            print("\nnothing was changed; pass --force to reset without asking")
            return 1
        if answer.strip().lower() != "yes":
            print("nothing was changed")
            return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stub, encoding="utf-8", newline="\n")
    print(f"reset {file_of(entry, target)}")
    return 0


def snapshot(folder):
    """What the exercise files look like right now, for spotting edits."""
    out = {}
    for path in sorted(pathlib.Path(folder).rglob("*.py")):
        try:
            info = path.stat()
        except OSError:  # deleted between the glob and the stat
            continue
        out[path] = (info.st_mtime_ns, info.st_size)
    return out


def changed_exercises(registry, target, before, after):
    """Names of the exercises whose file differs between two snapshots."""
    touched = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    names = []
    for entry in registry:
        if (ROOT / file_of(entry, target)).resolve() in {p.resolve() for p in touched}:
            names.append(entry["name"])
    return names


def watch(registry, target, zh, interval=0.4):
    """Rerun an exercise every time its file is saved.

    Every check runs in a fresh process. Within one process, pytest keeps the
    test modules in ``sys.modules``, and those hold the functions loaded at
    import time, so a second run would still be testing the code as it was
    when the watch started.
    """
    common = ["--target", target] + (["--zh"] if zh else [])

    def cli(*args):
        subprocess.run([sys.executable, str(ROOT / "robolings.py"), *args, *common], cwd=ROOT)

    folder = ROOT / target
    cli("--brief")
    print("\nWatching for changes. Press Ctrl-C to stop.")

    before = snapshot(folder)
    try:
        while True:
            time.sleep(interval)
            after = snapshot(folder)
            names = changed_exercises(registry, target, before, after)
            before = after
            for name in names:
                print(f"\n--- {name} changed")
                cli("run", name)
            if names:
                cli("--brief")
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Small exercises for robot learning.")
    parser.add_argument(
        "command",
        nargs="?",
        default="progress",
        choices=["progress", "watch", "run", "show", "hint", "reset", "list"],
    )
    parser.add_argument("exercise", nargs="?")
    parser.add_argument("--target", default="exercises", choices=["exercises", "solutions"])
    parser.add_argument("--zh", action="store_true", help="Chinese titles, statements and hints")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--brief", action="store_true", help="progress without the per-exercise list")
    parser.add_argument("--badge", metavar="FILE", help="also write a progress badge (SVG)")
    parser.add_argument(
        "--badge-count",
        action="store_true",
        help="the badge shows the number of exercises, not your progress",
    )
    parser.add_argument("--force", action="store_true", help="reset without asking")
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

    if args.command in ("run", "show", "hint", "reset"):
        if not args.exercise:
            raise SystemExit(f"usage: python robolings.py {args.command} <name>")
        if args.command == "run":
            return run_one(registry, args.exercise, args.target)
        if args.command == "show":
            return show(registry, args.exercise, args.target, zh)
        if args.command == "hint":
            return hint(registry, args.exercise, zh)
        return reset(registry, args.exercise, args.target, args.force)

    if args.command == "watch":
        return watch(registry, args.target, zh)

    status = grade(registry, run_all(args.target))
    if args.markdown:
        print_markdown(registry, status, zh)
    else:
        print_progress(registry, status, args.target, zh, args.brief)

    if args.badge:
        done = sum(1 for s in status.values() if s == "done")
        svg = badge_svg(done, len(registry), count_only=args.badge_count)
        pathlib.Path(args.badge).write_text(svg, encoding="utf-8", newline="\n")

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
