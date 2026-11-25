# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the whole-tree star-import gate in CI and pre-commit

"""No star import may come back, suppressed or not.

The test modules once used ``from …support import *`` under a ruff exemption for
``tests/**/*.py`` and ``crates/**/test_*.py``. Every such import has been replaced with an
explicit one, so the exemption is gone. CI and pre-commit now run
``ruff check --isolated --ignore-noqa --select F403,F405`` over the whole tree: ``--isolated``
also covers ``research/``, which the project configuration excludes, and ``--ignore-noqa``
keeps a ``# noqa: F403`` from hiding a new star import.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_COMMAND = "ruff check --isolated --no-cache --ignore-noqa --select F403,F405 ."
GATE_ARGS = ["--isolated", "--ignore-noqa", "--select", "F403,F405"]


def test_no_ruff_exemption_for_star_imports_remains() -> None:
    """The project ruff configuration exempts no path from F403 or F405."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lint = configuration["tool"]["ruff"]["lint"]
    per_file = cast(dict[str, list[str]], lint.get("per-file-ignores", {}))
    exempted = {path: codes for path, codes in per_file.items() if {"F403", "F405"} & set(codes)}
    assert exempted == {}
    assert not {"F403", "F405"} & set(cast(list[str], lint.get("ignore", [])))


def test_ci_runs_the_whole_tree_gate_next_to_ruff_check() -> None:
    """CI runs the isolated, noqa-blind F403/F405 check over the whole tree in the ruff job."""
    workflow = cast(dict[str, Any], yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text()))
    jobs = cast(dict[str, dict[str, Any]], workflow["jobs"])
    holders = [
        name
        for name, job in jobs.items()
        if any(
            step.get("run") == CI_COMMAND
            for step in cast(list[dict[str, Any]], job.get("steps", []))
        )
    ]
    assert len(holders) == 1, holders
    names = [step.get("name") for step in jobs[holders[0]]["steps"]]
    assert "ruff check" in names
    assert names.index("No star imports, suppressed or not") == names.index("ruff check") + 1


def test_pre_commit_runs_the_same_gate_on_staged_files() -> None:
    """A second ruff hook runs the same isolated, noqa-blind F403/F405 check."""
    configuration = cast(
        dict[str, Any], yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    )
    hooks = [
        hook
        for repository in cast(list[dict[str, Any]], configuration["repos"])
        for hook in cast(list[dict[str, Any]], repository.get("hooks", []))
        if hook.get("id") == "ruff"
    ]
    gates = [hook for hook in hooks if hook.get("args") == GATE_ARGS]
    assert len(gates) == 1, hooks
    assert gates[0]["name"] == "ruff (no star imports, suppressed or not)"


def _ruff(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the installed ruff on ``cwd`` with ``arguments``."""
    return subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", *arguments, "."],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_the_gate_catches_a_suppressed_star_import_that_plain_ruff_lets_through(
    tmp_path: Path,
) -> None:
    """A ``# noqa: F403`` star import passes an ordinary check but fails the gate's arguments."""
    (tmp_path / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_module.py").write_text(
        "from helper import *  # noqa: F403\n\nprint(VALUE)  # noqa: F405\n", encoding="utf-8"
    )
    ordinary = _ruff(["--isolated", "--select", "F403,F405"], tmp_path)
    gate = _ruff(GATE_ARGS, tmp_path)
    assert ordinary.returncode == 0, ordinary.stdout
    assert gate.returncode == 1
    assert "F403" in gate.stdout and "F405" in gate.stdout


def test_the_repository_tree_passes_the_gate() -> None:
    """The whole tree, research/ included, holds no star import."""
    result = _ruff(GATE_ARGS, ROOT)
    assert result.returncode == 0, result.stdout[-2000:]
