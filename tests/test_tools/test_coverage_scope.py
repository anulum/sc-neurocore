# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The validation document states the scope the coverage gate really has

"""Count a real source tree against real omission patterns and bind the committed document."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tools.coverage_scope import BEGIN, END, main, read_scope, render, replace_block

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION = (
    "[tool.coverage.run]\n"
    'source_dirs = ["src/pkg"]\n'
    'omit = ["*/heavy/*", "*/pkg/cli.py", "*/absent/*", "*/heavy/big.py"]\n'
)


def _tree(root: Path, configuration: str = CONFIGURATION) -> Path:
    """Create a small package with a measured module, an omitted folder and an omitted file."""
    (root / "src/pkg/heavy").mkdir(parents=True)
    (root / "src/pkg/__pycache__").mkdir()
    (root / "pyproject.toml").write_text(configuration, encoding="utf-8")
    (root / "src/pkg/core.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    (root / "src/pkg/cli.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src/pkg/heavy/big.py").write_text("y = 1\n" * 10, encoding="utf-8")
    (root / "src/pkg/heavy/small.py").write_text("z = 1\nz = 2\n", encoding="utf-8")
    (root / "src/pkg/__pycache__/core.py").write_text("ignored = 1\n", encoding="utf-8")
    (root / "src/pkg/notes.txt").write_text("not source\n", encoding="utf-8")
    (root / "VALIDATION.md").write_text(
        f"# V\n\n{BEGIN}\nSTALE-BLOCK\n{END}\n\ntail\n", encoding="utf-8"
    )
    return root


def test_scope_counts_files_and_lines_once_however_many_patterns_match(tmp_path: Path) -> None:
    """Totals count an omitted file once; each pattern row counts what it matches.

    Parameters
    ----------
    tmp_path : Path
        Root of the small package.
    """
    scope = read_scope(_tree(tmp_path))
    assert (scope.package, scope.files, scope.lines) == ("src/pkg", 4, 16)
    assert (scope.omitted_files, scope.omitted_lines) == (3, 13)
    assert [(row.pattern, row.files, row.lines) for row in scope.omissions] == [
        ("*/heavy/*", 2, 12),
        ("*/pkg/cli.py", 1, 1),
        ("*/absent/*", 0, 0),
        ("*/heavy/big.py", 1, 10),
    ]
    block = render(scope)
    assert block.startswith(BEGIN) and block.endswith(END)
    assert "applies to 1 of 4 Python files" in block
    assert "(3 of 16 source lines, 18.8 %)" in block
    assert "| `*/heavy/*` | 2 | 12 |" in block


def test_the_tool_writes_the_block_and_then_finds_it_current(tmp_path: Path) -> None:
    """Writing replaces only the block; a later change of the tree is reported.

    Parameters
    ----------
    tmp_path : Path
        Root of the small package.
    """
    root = _tree(tmp_path)
    assert main(["--repo", str(root), "--check"]) == 1
    assert main(["--repo", str(root)]) == 0
    written = (root / "VALIDATION.md").read_text(encoding="utf-8")
    assert written.startswith("# V\n\n" + BEGIN) and written.endswith(END + "\n\ntail\n")
    assert "STALE-BLOCK" not in written
    assert main(["--repo", str(root), "--check"]) == 0
    (root / "src/pkg/heavy/more.py").write_text("w = 1\n", encoding="utf-8")
    assert main(["--repo", str(root), "--check"]) == 1


@pytest.mark.parametrize(
    "configuration",
    [
        "[tool.other]\n",
        '[tool.coverage.run]\nsource_dirs = ["a", "b"]\nomit = []\n',
        '[tool.coverage.run]\nsource_dirs = ["src/pkg"]\n',
        '[tool.coverage.run]\nsource_dirs = ["src/pkg"]\nomit = [1]\n',
        '[tool.coverage.run]\nsource_dirs = ["src/empty"]\nomit = []\n',
    ],
)
def test_an_unusable_configuration_is_reported_not_guessed(
    tmp_path: Path, configuration: str
) -> None:
    """A missing source directory or omission list stops the tool with exit status two.

    Parameters
    ----------
    tmp_path : Path
        Root of the small package.
    configuration : str
        Coverage configuration that does not determine a scope.
    """
    root = _tree(tmp_path, configuration)
    with pytest.raises(ValueError):
        read_scope(root)
    assert main(["--repo", str(root)]) == 2


def test_a_document_without_one_ordered_block_is_refused(tmp_path: Path) -> None:
    """No marker, two blocks, reversed markers and a missing document are all refused.

    Parameters
    ----------
    tmp_path : Path
        Root of the small package.
    """
    for document in ("no markers", f"{BEGIN}{END}{BEGIN}{END}", f"{END} text {BEGIN}"):
        with pytest.raises(ValueError):
            replace_block(document, "block")
    root = _tree(tmp_path)
    (root / "VALIDATION.md").unlink()
    assert main(["--repo", str(root), "--check"]) == 2


def test_the_committed_validation_document_states_the_current_scope() -> None:
    """The repository's own document is current, and names its largest omissions."""
    completed = subprocess.run(
        [sys.executable, "-m", "tools.coverage_scope", "--repo", str(ROOT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    document = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
    block = document.split(BEGIN, 1)[1].split(END, 1)[0]
    for pattern in ("*/neurons/models/*", "*/training/*", "*/studio/*", "*/conversion/*"):
        assert f"| `{pattern}` |" in block
    for stale in ("drivers/", "ccw_bridge", "sleep/"):
        assert stale not in block


def test_in_a_work_tree_only_tracked_files_are_counted(tmp_path: Path) -> None:
    """A build product or leftover beside the tracked sources does not change the scope.

    Parameters
    ----------
    tmp_path : Path
        Root of the small package, made a Git work tree for this test.
    """
    root = _tree(tmp_path)
    untracked = read_scope(root)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, timeout=30)
    subprocess.run(
        ["git", "-C", str(root), "add", "src/pkg/core.py", "src/pkg/cli.py", "src/pkg/heavy"],
        check=True,
        timeout=30,
    )
    (root / "src/pkg/generated.py").write_text("g = 1\n" * 50, encoding="utf-8")
    tracked = read_scope(root)
    assert (tracked.files, tracked.lines) == (untracked.files, untracked.lines) == (4, 16)
    assert tracked.omissions == untracked.omissions
