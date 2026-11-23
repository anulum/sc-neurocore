# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine Python-binding location rule tests

"""Exercise the binding-location rule on the live tree and on planted repositories."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tools import engine_bindings_location as rule

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "engine_bindings_location.py"
ATTRIBUTE_FORMS = (
    "#[pyfunction]",
    "#[pyfunction(signature = (x, dt=None))]",
    "#[pyo3::pyfunction]",
    "    # [ pyo3 :: pyfunction ]",
)
FUNCTION = "pub fn py_example() -> u8 {\n    0\n}\n"


def _git(repo: Path, *arguments: str) -> None:
    """Run one Git command in a planted repository and require success."""
    subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True)


def _planted_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a Git repository whose tracked files are exactly ``files``."""
    repo = tmp_path / "planted"
    repo.mkdir()
    _git(repo, "init", "-q")
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "--", *files)
    return repo


def test_live_tree_keeps_every_pyfunction_in_a_binding_module(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The committed engine sources satisfy the rule."""
    assert rule.main(["--check", "--repo", str(REPO_ROOT)]) == 0
    assert "Engine bindings location accepted:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("engine/src/bindings/exp_if.rs", True),
        ("engine/src/analysis/bindings.rs", True),
        ("engine/src/analysis/bindings/decoding.rs", True),
        ("engine/src/rk4_neurons/bindings.rs", True),
        ("engine/src/rk4_neurons.rs", False),
        ("engine/src/learning_bindings.rs", False),
        ("engine/src/bindings_helpers.rs", False),
        ("engine/src/neurons/motor.rs", False),
    ],
)
def test_binding_module_is_a_bindings_directory_or_stem(path: str, expected: bool) -> None:
    """Only a ``bindings`` directory component or file stem marks a binding module."""
    assert rule.is_binding_module(path) is expected


@pytest.mark.parametrize("attribute", ATTRIBUTE_FORMS)
def test_every_attribute_form_is_refused_outside_and_accepted_inside(
    tmp_path: Path, attribute: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each attribute form fails in a model file and passes in a binding module."""
    source = f"//! Example.\n\n{attribute}\n{FUNCTION}"
    repo = _planted_repo(
        tmp_path,
        {"engine/src/model.rs": source, "engine/src/domain/bindings.rs": source},
    )
    assert rule.main(["--check", "--repo", str(repo)]) == 1
    error = capsys.readouterr().err
    assert "engine/src/model.rs:3" in error
    assert "engine/src/domain/bindings.rs" not in error


def test_comment_mentions_and_untracked_files_are_not_attributes(tmp_path: Path) -> None:
    """Doc-comment text is ignored, and so is a Rust file Git does not track."""
    repo = _planted_repo(
        tmp_path,
        {"engine/src/model.rs": "/// Wrapped by a #[pyfunction] elsewhere.\n" + FUNCTION},
    )
    (repo / "engine/src/scratch.rs").write_text("#[pyfunction]\n" + FUNCTION, encoding="utf-8")
    assert rule.tracked_engine_sources(repo) == ["engine/src/model.rs"]
    assert rule.misplaced_bindings(repo, rule.tracked_engine_sources(repo)) == []


def test_findings_are_reported_in_path_and_line_order(tmp_path: Path) -> None:
    """Every misplaced attribute is listed with its own line number."""
    repo = _planted_repo(
        tmp_path,
        {
            "engine/src/b.rs": "#[pyfunction]\n" + FUNCTION + "#[pyfunction]\n" + FUNCTION,
            "engine/src/a.rs": "#[pyfunction]\n" + FUNCTION,
            "engine/src/notes.txt": "#[pyfunction]\n",
        },
    )
    found = rule.misplaced_bindings(repo, rule.tracked_engine_sources(repo))
    assert found == [
        rule.MisplacedBinding("engine/src/a.rs", 1),
        rule.MisplacedBinding("engine/src/b.rs", 1),
        rule.MisplacedBinding("engine/src/b.rs", 5),
    ]


def test_a_directory_outside_git_is_refused_as_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed Git listing exits two with Git's message."""
    assert rule.main(["--check", "--repo", str(tmp_path)]) == 2
    assert "git ls-files failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error", [OSError("no git"), subprocess.TimeoutExpired(cmd="git", timeout=60)]
)
def test_git_that_cannot_run_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    """A missing or hung Git raises the rule's own error."""

    def fail(*_args: object, **_kwargs: object) -> None:
        """Raise the injected launch failure."""
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(rule.BindingsLocationError, match="could not run"):
        rule.tracked_engine_sources(tmp_path)


@pytest.mark.parametrize(
    ("content", "remove"),
    [(b"#[pyfunction]\n\xff\xfe\n", False), (b"#[pyfunction]\n", True)],
)
def test_unreadable_tracked_source_exits_two(
    tmp_path: Path, content: bytes, remove: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tracked file that is not UTF-8 or is missing from the tree is unavailable input."""
    repo = _planted_repo(tmp_path, {"engine/src/model.rs": ""})
    target = repo / "engine/src/model.rs"
    target.write_bytes(content)
    if remove:
        target.unlink()
    assert rule.main(["--check", "--repo", str(repo)]) == 2
    assert "engine/src/model.rs could not be read" in capsys.readouterr().err


def test_check_mode_is_required() -> None:
    """The tool refuses to run without an explicit ``--check``."""
    with pytest.raises(SystemExit) as raised:
        rule.main(["--repo", str(REPO_ROOT)])
    assert raised.value.code == 2


def test_script_entry_point_checks_the_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the file as a script checks the current repository and exits zero."""
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", [str(TOOL_PATH), "--check"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(TOOL_PATH), run_name="__main__")
    assert raised.value.code == 0
