# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Strict typing and docstring policy contract

"""Tests for the strict typing and NumPy docstring policy wiring."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCSTRING_POLICY = "docs/docstring_policy.toml"
DOCSTRING_TEST = "tests/test_public_docstring_policy.py"


def _read(relative_path: str) -> str:
    """Return repository text for a committed policy surface."""

    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _toml_section(text: str, header: str) -> str:
    """Extract one top-level TOML section by exact header."""

    lines = text.splitlines()
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise AssertionError(f"missing TOML section {header}") from exc

    collected: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("[") and line.endswith("]"):
            break
        collected.append(line)
    return "\n".join(collected)


def _file_entries(policy_text: str) -> list[str]:
    """Return file paths listed in the scoped docstring policy."""

    return re.findall(r'^path = "([^"]+)"$', policy_text, flags=re.MULTILINE)


def test_toml_section_reports_missing_headers() -> None:
    """Surface malformed policy files with a clear missing-section failure."""

    with pytest.raises(AssertionError, match=r"missing TOML section \[tool\.mypy\]"):
        _toml_section("[tool.ruff]\nline-length = 100\n", "[tool.mypy]")


def test_strict_mypy_policy_is_global_and_ci_gated() -> None:
    """Keep the repository-wide strict Mypy gate aligned with the broadcast."""

    pyproject = _read("pyproject.toml")
    mypy_section = _toml_section(pyproject, "[tool.mypy]")
    ci_workflow = _read(".github/workflows/ci.yml")
    preflight = _read("tools/preflight.py")

    assert re.search(r"^strict = true$", mypy_section, flags=re.MULTILINE)
    assert re.search(r"^ignore_missing_imports = false$", mypy_section, flags=re.MULTILINE)
    assert "mypy --strict src/sc_neurocore/" in ci_workflow
    assert '"mypy", ["python", "-m", "mypy", "--strict", "src/sc_neurocore/"]' in preflight

    forbidden_relaxations = (
        "disallow_untyped_defs = false",
        "check_untyped_defs = false",
        "warn_return_any = false",
        "ignore_errors = true",
    )
    for relaxation in forbidden_relaxations:
        assert relaxation not in pyproject

    assert "Strict-typing burn-down ledger — COMPLETE" in pyproject
    assert "No general per-module strictness relaxations remain" in pyproject


def test_numpy_docstring_policy_is_scoped_and_enforced() -> None:
    """Lock NumPy-convention docstring enforcement to its maintained surface."""

    pyproject = _read("pyproject.toml")
    pydocstyle_section = _toml_section(pyproject, "[tool.ruff.lint.pydocstyle]")
    policy = _read(DOCSTRING_POLICY)
    policy_test = _read(DOCSTRING_TEST)
    preflight = _read("tools/preflight.py")

    assert re.search(r'^convention = "numpy"$', pydocstyle_section, flags=re.MULTILINE)
    assert "`D` is not yet in the global" in pyproject
    assert "tests/test_public_docstring_policy.py" in pyproject
    assert "ruff check --select D" in pyproject
    assert f'["python", "-m", "pytest", "{DOCSTRING_TEST}", "-q"]' in preflight
    assert '"--select", "D", "--no-cache", *files' in policy_test
    assert "docs/docstring_policy.toml" in policy_test

    file_entries = _file_entries(policy)
    expected_match = re.search(r"^expected_file_count = ([0-9]+)$", policy, flags=re.MULTILINE)
    assert expected_match is not None
    assert len(file_entries) == int(expected_match.group(1))
    assert len(file_entries) == len(set(file_entries))
    assert file_entries
    for relative_path in file_entries:
        assert (REPO_ROOT / relative_path).is_file()


def test_maintenance_docs_describe_typing_docstring_boundary() -> None:
    """Keep public maintenance docs aligned with the current enforcement model."""

    docs = _read("docs/development/maintenance_tools.md")

    assert "Strict typing and docstring policy" in docs
    assert "mypy --strict src/sc_neurocore/" in docs
    assert "pytest tests/test_public_docstring_policy.py -q" in docs
    assert "docs/docstring_policy.toml" in docs
    assert "NumPy-convention" in docs
    assert "`D` can be promoted" in docs
    assert "global\nRuff select" in docs
