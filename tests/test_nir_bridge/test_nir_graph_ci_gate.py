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

    # The coverage leg is gated on the `primary` matrix flag (the single leg
    # that also runs the aggregate --cov), which is the exact-patch 3.12 entry.
    assert "if: matrix.primary" in step
    assert "COVERAGE_FILE=.coverage-nir-graph" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/nir_bridge/neuron_graph*.py'" in step
    assert "-m pytest tests/test_nir_bridge -q" in step
    assert "--fail-under=100 --show-missing" in step


def test_framework_profiles_require_installed_real_boundaries_and_no_skips() -> None:
    """Keep the required profiles pinned, installed and independently qualified."""
    import yaml

    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["test-nir-interoperability"]
    assert job["strategy"]["matrix"]["profile"] == ["snntorch", "sinabs"]
    assert "test-nir-interoperability" in workflow["jobs"]["build"]["needs"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "--require-hashes --only-binary :all:" in commands
    assert "python -m installer dist/sc_neurocore-*.whl" in commands
    assert "metadata.version(name) == version" in commands
    assert 'case.find("skipped") is None' in commands
    assert "assert cases" in commands
    for name in (
        "test_snntorch_bridge.py",
        "test_snntorch_bridge_export.py",
        "test_sinabs_runtime.py",
    ):
        assert name in commands
    assert "--confcutdir=" in commands and "-o pythonpath=" in commands
