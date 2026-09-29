"""Find and load the exercise files the tests run against.

    pytest                                  your work in exercises/
    ROBOLINGS_TARGET=solutions pytest       the reference solutions

An absolute directory is accepted too; ``tools/mutants.py`` uses that.
"""

import importlib.util
import json
import os
import pathlib
import types

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

REGISTRY = json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))


def _unavailable(path, error):
    def function(*args, **kwargs):
        raise RuntimeError(f"{path} could not be loaded: {error!r}")

    return function


def load_track(track):
    """Namespace with every function of a track.

    A file that is missing or does not import only breaks its own exercise:
    its functions are replaced by ones that raise when called.
    """
    entries = [e for e in REGISTRY if e["track"] == track]
    if not entries:
        raise RuntimeError(f"no exercises registered for track {track!r}")

    namespace = types.SimpleNamespace()
    for entry in entries:
        path = TARGET_DIR / track / f"{entry['name']}.py"
        module = None
        error = None
        try:
            spec = importlib.util.spec_from_file_location(f"robolings_{track}_{entry['name']}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # a learner's file can fail in any way
            error = exc
        for name in entry["functions"]:
            if error is None and hasattr(module, name):
                setattr(namespace, name, getattr(module, name))
            else:
                reason = error if error is not None else AttributeError(f"no function {name}()")
                setattr(namespace, name, _unavailable(path, reason))
    return namespace
