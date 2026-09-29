"""Tests for the tooling itself.

python -m pytest tools
"""

import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_exercises  # noqa: E402

TOTAL = len(json.loads((ROOT / "exercises.json").read_text(encoding="utf-8")))


def is_open(out, name):
    return re.search(rf"^\s*\[ \]\s+\d+\s+{name}\s", out, flags=re.MULTILINE) is not None


def is_done(out, name):
    return re.search(rf"^\s*\[x\]\s+\d+\s+{name}\s", out, flags=re.MULTILINE) is not None


def copy_with_solved_exercises(tmp_path):
    """A copy of the repository in which every exercise is solved."""
    work = tmp_path / "repo"
    work.mkdir()
    for name in ("tests", "solutions", "docs", "robolings.py", "exercises.json"):
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
    assert f"{TOTAL}/{TOTAL}" in out


def test_a_file_with_a_syntax_error_only_fails_its_own_exercise(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    (work / "exercises" / "01_rotations" / "slerp.py").write_text("def slerp(:\n", encoding="utf-8")
    out = progress(work)
    assert f"{TOTAL - 1}/{TOTAL}" in out
    assert is_open(out, "slerp")
    assert is_done(out, "matrix_to_quat")


def test_a_missing_file_only_fails_its_own_exercise(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    (work / "exercises" / "04_control" / "min_jerk.py").unlink()
    out = progress(work)
    assert f"{TOTAL - 1}/{TOTAL}" in out
    assert is_open(out, "min_jerk")


def test_a_renamed_function_only_fails_its_own_exercise(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    path = work / "exercises" / "05_imitation" / "minmax.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace("def minmax_unnormalize(", "def unnormalize("),
        encoding="utf-8",
    )
    out = progress(work)
    assert f"{TOTAL - 1}/{TOTAL}" in out
    assert is_open(out, "minmax")


def test_run_accepts_a_name_or_a_position(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    names = [e["name"] for e in json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))]
    for key in ("slerp", str(names.index("slerp") + 1)):
        result = subprocess.run(
            [sys.executable, "robolings.py", "run", key],
            cwd=work,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout
        assert result.stdout.startswith("slerp ")


def cli(work, *args, env=None):
    result = subprocess.run(
        [sys.executable, "robolings.py", *args],
        cwd=work,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_show_prints_the_english_statement(tmp_path):
    out = cli(copy_with_solved_exercises(tmp_path), "show", "ddim_step")
    assert "ddim_step(x_t, eps_pred, t, t_prev, alphas_cumprod)" in out
    assert "One deterministic DDIM step" in out
    assert "exercises/06_diffusion/ddim_step.py" in out


def test_show_prints_the_chinese_statement(tmp_path):
    out = cli(copy_with_solved_exercises(tmp_path), "show", "ddim_step", "--zh")
    assert "DDIM 单步" in out
    assert "<!--" not in out
    assert "One deterministic DDIM step" not in out


def test_show_falls_back_to_english(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    (work / "docs" / "zh" / "06_diffusion" / "ddim_step.md").unlink()
    out = cli(work, "show", "ddim_step", "--zh")
    assert "no Chinese statement" in out
    assert "One deterministic DDIM step" in out


def test_language_from_the_environment(tmp_path):
    import os

    work = copy_with_solved_exercises(tmp_path)
    out = cli(work, "list", env=dict(os.environ, ROBOLINGS_LANG="zh"))
    assert "球面线性插值" in out
    out = cli(work, "list", env={k: v for k, v in os.environ.items() if k != "ROBOLINGS_LANG"})
    assert "球面线性插值" not in out


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
