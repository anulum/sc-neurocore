# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR graph CI coverage gate contract

"""Lock the module-specific CI coverage gate to the production graph surface."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def test_ci_enforces_exact_neuron_graph_coverage_on_python_312() -> None:
    """Keep the focused NIR graph gate exact and isolated from global omits."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: NIR graph exact coverage", maxsplit=1)[1].split(
        "- name: Upload test results", maxsplit=1
    )[0]

    assert "if: matrix.python-version == '3.12'" in step
    assert "COVERAGE_FILE=.coverage-nir-graph" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/nir_bridge/neuron_graph*.py'" in step
    assert "-m pytest tests/test_nir_bridge -q" in step
    assert "--fail-under=100 --show-missing" in step
