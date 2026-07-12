# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Systematic audit rerun contract

"""Regression tests for the 2026-07-04 systematic audit rerun."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import snn_memory_discipline_audit, spdx_header_audit


REPO_ROOT = Path(__file__).resolve().parents[2]
MONOREPO_ROOT = REPO_ROOT.parents[1]
SC_NEUROCORE_STIMULI = MONOREPO_ROOT / "04_ARCANE_SAPIENCE" / "snn_stimuli" / "SC-NEUROCORE"


def _check_ignore(path: str) -> str:
    """Return the gitignore rule that covers one audited path."""

    completed = subprocess.run(
        ["git", "check-ignore", "-v", path],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_systematic_audit_gitignore_and_todo_findings_stay_closed() -> None:
    """Keep the old gitignore and TODO-placement findings closed."""

    assert ".env" in _check_ignore(".env")
    assert "TODO" in _check_ignore("TODO")
    assert "docs/internal/" in _check_ignore("docs/internal/TODO.md")
    assert not (REPO_ROOT / "TODO").exists()
    # The canonical TODO lives at docs/internal/TODO.md, which is git-ignored and
    # therefore absent from a clean CI checkout; only assert its placement when the
    # working tree actually carries the ignored internal docs.
    if not (REPO_ROOT / "docs/internal/TODO.md").is_file():
        pytest.skip("git-ignored docs/internal/TODO.md absent in a clean/CI checkout")


def test_systematic_audit_spdx_and_memory_findings_stay_closed() -> None:
    """Rerun the old SPDX and SNN-memory audit classes deterministically."""

    missing_headers = spdx_header_audit.missing_direct_header_paths(REPO_ROOT)
    assert missing_headers == []

    # The SNN-memory audit reads stimulus records from the monorepo sibling tree
    # (04_ARCANE_SAPIENCE/snn_stimuli), which a standalone/CI checkout does not
    # contain; only rerun it when that tree is present.
    if not SC_NEUROCORE_STIMULI.is_dir():
        pytest.skip("monorepo SNN stimulus tree absent in a standalone/CI checkout")
    memory_audit = snn_memory_discipline_audit.audit_memory_discipline(
        REPO_ROOT,
        SC_NEUROCORE_STIMULI,
        "SC-NEUROCORE",
    )

    assert memory_audit.passed
    assert memory_audit.violations == ()
    assert memory_audit.to_json()["violation_count"] == 0
    assert memory_audit.checked_records > 0
