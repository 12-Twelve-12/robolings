"""Tests for the tooling itself.

python -m pytest tools
"""

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_exercises  # noqa: E402


def copy_with_solved_exercises(tmp_path):
    """A copy of the repository in which every exercise is solved."""
    work = tmp_path / "repo"
    work.mkdir()
    for name in ("tests", "solutions", "robolings.py", "exercises.json"):
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, work / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(source, work / name)
    shutil.copytree(work / "solutions", work / "exercises")
    return work


def progress(work):
    result = subprocess.run(
        [sys.executable, "robolings.py"], cwd=work, capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_solved_copy_is_complete(tmp_path):
    out = progress(copy_with_solved_exercises(tmp_path))
    assert "25/25" in out


def test_a_file_with_a_syntax_error_only_fails_its_own_exercise(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    (work / "exercises" / "01_rotations" / "slerp.py").write_text("def slerp(:\n", encoding="utf-8")
    out = progress(work)
    assert "24/25" in out
    assert "[ ]  5  slerp" in out
    assert "[x]  4  matrix_to_quat" in out


def test_a_missing_file_only_fails_its_own_exercise(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    (work / "exercises" / "04_control" / "min_jerk.py").unlink()
    out = progress(work)
    assert "24/25" in out
    assert "[ ] 13  min_jerk" in out


def test_a_renamed_function_only_fails_its_own_exercise(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    path = work / "exercises" / "05_imitation" / "minmax.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace("def minmax_unnormalize(", "def unnormalize("),
        encoding="utf-8",
    )
    out = progress(work)
    assert "24/25" in out
    assert "[ ] 17  minmax" in out


def test_run_accepts_a_name_or_a_position(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    for key in ("slerp", "5"):
        result = subprocess.run(
            [sys.executable, "robolings.py", "run", key],
            cwd=work,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout
        assert result.stdout.startswith("slerp ")


def test_strip_replaces_the_marked_block():
    text = 'def f(x):\n    """doc"""\n    # >>> solution\n    return x\n    # <<< solution\n'
    out, blocks = make_exercises.strip_solutions(text, "f.py")
    assert blocks == 1
    assert "return x" not in out
    assert '"""doc"""' in out
    assert "    raise NotImplementedError" in out


@pytest.mark.parametrize(
    "text",
    [
        "def f():\n    # >>> solution\n    return 1\n",
        "def f():\n    return 1\n    # <<< solution\n",
        "def f():\n    # >>> solution\n    # >>> solution\n    return 1\n    # <<< solution\n",
    ],
)
def test_strip_rejects_unbalanced_markers(text):
    with pytest.raises(ValueError):
        make_exercises.strip_solutions(text, "f.py")
