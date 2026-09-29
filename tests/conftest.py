"""Point the tests at ``exercises/`` (default) or ``solutions/``.

    pytest                                  your work in exercises/
    ROBOLINGS_TARGET=solutions pytest       the reference solutions

An absolute directory is accepted too; ``tools/mutants.py`` uses that.
"""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = os.environ.get("ROBOLINGS_TARGET", "exercises")

if TARGET in ("exercises", "solutions"):
    TARGET_DIR = ROOT / TARGET
elif os.path.isabs(TARGET) and os.path.isdir(TARGET):
    TARGET_DIR = pathlib.Path(TARGET)
else:
    raise RuntimeError(
        f"ROBOLINGS_TARGET must be 'exercises', 'solutions' or an existing absolute directory, got {TARGET!r}"
    )

sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(TARGET_DIR))


def pytest_report_header(config):
    return f"robolings target: {TARGET_DIR}"
