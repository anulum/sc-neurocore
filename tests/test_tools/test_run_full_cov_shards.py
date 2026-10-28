# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sharded full-suite runner covers every batch once and gates the union

"""Run the real batch runner on a small repository and gate the shard union.

The hosted coverage leg is split into shards because a single instrumented run
outgrew the runner's time limit. These tests run ``tools/run_full_cov.sh``
itself, with real pytest and coverage subprocesses, on a three-module project.
They check three things:

- the shards partition the batches;
- the union reaches the exact gate only when every shard's data is present;
- a missing shard or an uncovered branch fails.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "tools" / "run_full_cov.sh"

_PYPROJECT = """\
[tool.coverage.run]
source = ["sc_neurocore"]
source_dirs = ["src/sc_neurocore"]
parallel = true
relative_files = true
branch = true

[tool.coverage.report]
include = ["src/sc_neurocore/*"]
"""

_PACKAGE = """\
def increment(value):
    return value + 1


def sign(value):
    if value:
        return 1
    return 0
"""

_TESTS = {
    "test_alpha.py": "from sc_neurocore import increment\n\n\n"
    "def test_increment():\n    assert increment(1) == 2\n",
    "test_beta.py": "from sc_neurocore import sign\n\n\n"
    "def test_sign():\n    assert sign(1) == 1\n    assert sign(0) == 0\n",
    "test_gamma.py": "def test_independent():\n    assert True\n",
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Build a three-module project that uses the repository's runner unchanged."""
    if shutil.which("bash") is None:  # pragma: no cover - every supported host has bash
        pytest.skip("bash is not available")
    (tmp_path / "tools").mkdir()
    shutil.copy2(_RUNNER, tmp_path / "tools" / "run_full_cov.sh")
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    package = tmp_path / "src" / "sc_neurocore"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_PACKAGE, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    for name, body in _TESTS.items():
        (tests / name).write_text(body, encoding="utf-8")
    return tmp_path


def _run(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the copied runner with this interpreter first on PATH and no outer coverage.

    The outer coverage session's process hooks are removed from the environment,
    so the child runs measure only the small project and never write into the
    repository's own coverage data.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_", "PYTEST_"))
    }
    environment["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), environment["PATH"]])
    return subprocess.run(
        ["bash", "tools/run_full_cov.sh", "--batch-size=1", *args],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _batches(output: str) -> list[int]:
    """Return the batch numbers a run announced, in order."""
    return [int(number) for number in re.findall(r"batch (\d+)/3 \(", output)]


def _run_shards(project: Path, union: Path, count: int) -> list[list[int]]:
    """Run every shard and keep each shard's data as the hosted job publishes it."""
    union.mkdir(exist_ok=True)
    ran = []
    for index in range(1, count + 1):
        result = _run(project, f"--shard={index}/{count}")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "coverage data left in" in result.stdout
        assert "TOTAL" not in result.stdout
        ran.append(_batches(result.stdout))
        shutil.copy2(project / ".coverage", union / f".coverage.shard-{index}")
    return ran


def test_shards_partition_the_batches_and_the_union_passes_the_exact_gate(
    project: Path,
) -> None:
    """Two shards run batches 1 and 3, then 2; their union reaches exactly 100 %."""
    union = project / "union"
    assert _run_shards(project, union, 2) == [[1, 3], [2]]

    result = _run(project, f"--combine-from={union}", "--expect-shards=2", "--cov-fail-under=100")

    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"^TOTAL\s.*\s100\.00%$", result.stdout, re.MULTILINE)
    assert (project / "coverage.xml").is_file()
    assert sorted(path.name for path in union.iterdir()) == [
        ".coverage.shard-1",
        ".coverage.shard-2",
    ]


def test_a_missing_shard_or_an_uncovered_line_fails_the_union(project: Path) -> None:
    """The union refuses the wrong shard count, and one shard alone stays below 100 %."""
    union = project / "union"
    _run_shards(project, union, 2)

    wrong_count = _run(project, f"--combine-from={union}", "--expect-shards=3")
    assert wrong_count.returncode == 1
    assert "expected 3 shard coverage files" in wrong_count.stderr

    alone = project / "alone"
    alone.mkdir()
    shutil.copy2(union / ".coverage.shard-1", alone / ".coverage.shard-1")
    partial = _run(project, f"--combine-from={alone}", "--expect-shards=1", "--cov-fail-under=100")
    assert partial.returncode != 0
    assert "less than fail-under=100" in partial.stdout + partial.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--shard=3/2",), "shard must be K/N"),
        (("--shard=0/2",), "shard must be K/N"),
        (("--shard=1-2",), "shard must be K/N"),
        (("--combine-from=union",), "needs --expect-shards"),
        (("--expect-shards=2",), "needs --combine-from"),
        (("--combine-from=union", "--expect-shards=2", "--shard=1/2"), "excludes --shard"),
        (
            ("--combine-from=union", "--expect-shards=2", "--no-cov"),
            "excludes --shard and --no-cov",
        ),
    ],
)
def test_inconsistent_shard_arguments_are_refused_before_any_test_runs(
    project: Path, arguments: tuple[str, ...], message: str
) -> None:
    """A malformed shard request exits 2 without starting pytest."""
    result = _run(project, *arguments)

    assert result.returncode == 2
    assert message in result.stderr
    assert "batch 1/3" not in result.stdout
