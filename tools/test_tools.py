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
import mutants  # noqa: E402

TOTAL = len(json.loads((ROOT / "exercises.json").read_text(encoding="utf-8")))


def is_open(out, name):
    return re.search(rf"^\s*\[ \]\s+\d+\s+{name}\s", out, flags=re.MULTILINE) is not None


def is_done(out, name):
    return re.search(rf"^\s*\[x\]\s+\d+\s+{name}\s", out, flags=re.MULTILINE) is not None


def copy_with_solved_exercises(tmp_path):
    """A copy of the repository in which every exercise is solved."""
    work = tmp_path / "repo"
    work.mkdir()
    for name in ("tests", "solutions", "docs", "tools", "robolings.py", "exercises.json", "hints.json"):
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


def test_badge_shows_the_progress(tmp_path):
    import xml.etree.ElementTree as ET

    work = copy_with_solved_exercises(tmp_path)
    (work / "exercises" / "01_rotations" / "slerp.py").unlink()
    cli(work, "--badge", "badge.svg")
    text = (work / "badge.svg").read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert root.tag.endswith("svg")
    assert f">{TOTAL - 1}/{TOTAL}<" in text
    assert "robolings" in text


def test_badge_colours():
    sys.path.insert(0, str(ROOT))
    import robolings

    def colour(done, total):
        return re.search(
            r'<rect x="\d+" width="\d+" height="20" fill="(#[0-9a-f]+)"', robolings.badge_svg(done, total)
        )[1]

    assert colour(0, 30) == "#9f9f9f"
    assert colour(1, 30) == colour(9, 30) == "#fe7d37"
    assert colour(10, 30) == colour(19, 30) == "#dfb317"
    assert colour(20, 30) == colour(29, 30) == "#97ca00"
    assert colour(30, 30) == "#4c1"


def cli_raw(work, *args, stdin=""):
    return subprocess.run(
        [sys.executable, "robolings.py", *args],
        cwd=work,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_hint_in_both_languages(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    assert "branches" in cli(work, "hint", "matrix_to_quat")
    assert "分支" in cli(work, "hint", "matrix_to_quat", "--zh")


def test_hint_accepts_a_position(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    names = [e["name"] for e in json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))]
    assert cli(work, "hint", "slerp") == cli(work, "hint", str(names.index("slerp") + 1))


def test_every_exercise_has_a_hint(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    for name in json.loads((work / "hints.json").read_text(encoding="utf-8")):
        for extra in ([], ["--zh"]):
            assert cli(work, "hint", name, *extra).strip()


def test_reset_restores_the_stub(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    path = work / "exercises" / "01_rotations" / "slerp.py"
    assert "NotImplementedError" not in path.read_text(encoding="utf-8")
    result = cli_raw(work, "reset", "slerp", "--force")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NotImplementedError" in path.read_text(encoding="utf-8")
    assert is_open(progress(work), "slerp")


def test_reset_needs_confirmation(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    path = work / "exercises" / "04_control" / "min_jerk.py"
    before = path.read_text(encoding="utf-8")
    # no input at all, a refusal, and an answer that is not exactly "yes"
    for stdin in ("", "no" + chr(10), "y" + chr(10)):
        result = cli_raw(work, "reset", "min_jerk", stdin=stdin)
        assert result.returncode == 1, result.stdout
        assert path.read_text(encoding="utf-8") == before
    result = cli_raw(work, "reset", "min_jerk", stdin="yes" + chr(10))
    assert result.returncode == 0, result.stdout
    assert path.read_text(encoding="utf-8") != before


def test_reset_of_an_untouched_exercise_says_so(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    cli_raw(work, "reset", "slerp", "--force")
    result = cli_raw(work, "reset", "slerp")
    assert result.returncode == 0
    assert "already a fresh stub" in result.stdout


def test_brief_progress_leaves_out_the_list(tmp_path):
    work = copy_with_solved_exercises(tmp_path)
    full = cli(work)
    brief = cli(work, "--brief")
    assert f"{TOTAL}/{TOTAL}" in brief
    assert "slerp" not in brief
    assert "slerp" in full


def test_watch_spots_an_edited_exercise(tmp_path):
    sys.path.insert(0, str(ROOT))
    import robolings

    work = copy_with_solved_exercises(tmp_path)
    folder = work / "exercises"
    registry = json.loads((work / "exercises.json").read_text(encoding="utf-8"))
    before = robolings.snapshot(folder)
    assert robolings.changed_exercises(registry, str(folder), before, before) == []

    path = folder / "05_imitation" / "minmax.py"
    path.write_text(path.read_text(encoding="utf-8") + chr(10) + "# edited" + chr(10), encoding="utf-8")
    after = robolings.snapshot(folder)
    assert robolings.changed_exercises(registry, str(folder), before, after) == ["minmax"]

    (folder / "01_rotations" / "slerp.py").unlink()
    gone = robolings.changed_exercises(registry, str(folder), after, robolings.snapshot(folder))
    assert gone == ["slerp"]


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


def test_failure_kinds_tells_assertions_from_crashes():
    output = "\n".join(
        [
            "F.F.F                                                                    [100%]",
            "/home/me/robolings/tests/test_01_rotations.py:30: assert np.allclose(R, expected)",
            "D:\\robolings\\tests\\test_01_rotations.py:41: AssertionError",
            "/home/me/robolings/tests/test_01_rotations.py:52: IndexError: index 3 is out of bounds",
            "3 failed, 2 passed in 0.12s",
        ]
    )
    assert mutants.failure_kinds(output) == ["assertion", "assertion", "IndexError"]


def test_failure_kinds_of_a_mutant_that_never_runs():
    output = "\n".join(
        [
            "FF                                                                       [100%]",
            "/home/me/robolings/tests/test_02_kinematics.py:15: NameError: name 'np' is not defined",
            "/home/me/robolings/tests/test_02_kinematics.py:22: RuntimeError: x.py could not be loaded",
            "2 failed in 0.05s",
        ]
    )
    assert mutants.failure_kinds(output) == ["NameError", "RuntimeError"]


def test_a_mutant_that_crashes_is_not_a_kill(tmp_path):
    entry = next(
        e
        for e in json.loads((ROOT / "exercises.json").read_text(encoding="utf-8"))
        if e["name"] == "rodrigues"
    )
    crashes = "def rodrigues(axis, angle):\n    return undefined_name\n"
    problem, kinds = mutants.run_mutant(0, "rodrigues", "rodrigues", crashes, entry, tmp_path)
    assert problem is None
    assert kinds and "assertion" not in kinds

    wrong = "def rodrigues(axis, angle):\n    return np.eye(3)\n"
    problem, kinds = mutants.run_mutant(1, "rodrigues", "rodrigues", wrong, entry, tmp_path)
    assert problem is None
    assert "assertion" in kinds
