# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI Julia runtime provisioning contract

"""Guard deterministic Julia provisioning for the juliacall test surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml


ROOT = Path(__file__).resolve().parents[2]
JULIAPKG_PATH = ROOT / "src" / "sc_neurocore" / "juliapkg.json"
SETUP_JULIA = "julia-actions/setup-julia@fa02766e078afaaf09b14210362cee14137e6a32"
JULIA_VERSION = "1.11.9"


def _test_steps() -> list[dict[str, Any]]:
    """Return the parsed production test-matrix steps."""

    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow = cast(
        dict[str, Any],
        yaml.safe_load(workflow_path.read_text(encoding="utf-8")),
    )
    return cast(list[dict[str, Any]], workflow["jobs"]["test"]["steps"])


def _step_index(steps: list[dict[str, Any]], *, name: str) -> int:
    """Return the unique step index for ``name``."""

    matches = [index for index, step in enumerate(steps) if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def test_ci_pins_compatible_julia_before_package_installation() -> None:
    """Provision the exact juliacall-compatible runtime before Python install."""

    steps = _test_steps()
    setup_matches = [
        (index, step) for index, step in enumerate(steps) if step.get("uses") == SETUP_JULIA
    ]
    assert len(setup_matches) == 1
    setup_index, setup_step = setup_matches[0]
    assert setup_step["with"] == {
        "version": JULIA_VERSION,
        "show-versioninfo": "never",
    }
    assert setup_index < _step_index(steps, name="Install package")


def test_ci_fails_closed_on_wrong_julia_before_juliacall_import() -> None:
    """Assert the selected executable version before warming the real bridge."""

    steps = _test_steps()
    warm_index = _step_index(steps, name="Verify and warm up juliacall (Julia depot)")
    warm_step = steps[warm_index]
    run_text = cast(str, warm_step["run"])

    assert warm_step["shell"] == "bash"
    assert "set -euo pipefail" in run_text
    assert "julia --startup-file=no -e 'print(VERSION)'" in run_text
    assert f'= "{JULIA_VERSION}"' in run_text
    assert "python -c \"import juliacall; juliacall.Main.seval('1+1')\"" in run_text
    assert _step_index(steps, name="Install package") < warm_index
    assert warm_index < _step_index(steps, name="Test + coverage")


def test_python_package_pins_the_same_julia_runtime() -> None:
    """Ship the CI Julia patch pin with source and wheel installations."""

    contract = json.loads(JULIAPKG_PATH.read_text(encoding="utf-8"))
    assert contract == {"julia": f"={JULIA_VERSION}"}

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '  "juliapkg.json",' in pyproject
