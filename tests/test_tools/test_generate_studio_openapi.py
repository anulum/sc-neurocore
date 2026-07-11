# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for the Studio OpenAPI reference generator

"""Test deterministic generation of the public Studio OpenAPI contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sc_neurocore.studio.app import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "tools" / "generate_studio_openapi.py"
COMMITTED_REFERENCE = REPO_ROOT / "docs" / "_generated" / "studio_openapi.json"


def _run_generator(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_generated_reference_matches_runtime_contract(tmp_path: Path) -> None:
    output = tmp_path / "studio_openapi.json"

    first = _run_generator("--output", str(output))
    first_bytes = output.read_bytes()
    second = _run_generator("--output", str(output))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    assert json.loads(first_bytes) == create_app().openapi()


def test_committed_reference_is_current() -> None:
    result = _run_generator("--check", "--output", str(COMMITTED_REFERENCE))

    assert result.returncode == 0, result.stderr
