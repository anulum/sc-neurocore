# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic backpropagation documentation boundary tests

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "docs" / "guides" / "stochastic_backprop.md"
TRAINING_API = REPO_ROOT / "docs" / "api" / "training.md"
GENERATED_API = REPO_ROOT / "docs" / "API_REFERENCE.md"
MKDOCS = REPO_ROOT / "mkdocs.yml"


def test_stochastic_backprop_guide_documents_evidence_boundary_and_artifacts() -> None:
    text = GUIDE.read_text(encoding="utf-8")

    assert "local_simulation_and_executable_hdl_parity" in text
    assert "no physical hardware measurement" in text
    assert "no Vivado timing closure claim" in text
    assert "no PYNQ deployment claim" in text
    assert "stochastic_backprop_benchmark.json" in text
    assert "stochastic_backprop_export_manifest.json" in text
    assert "stochastic_backprop_estimator_regression_manifest.json" in text
    assert "stochastic_backprop_trained_design.v" in text
    assert "stochastic_backprop_trained_design_parity.json" in text
    assert "stochastic_backprop_joint_objective" in text
    assert "--estimator-regression-manifest" in text
    assert "hardware proof" not in text.lower()


def test_stochastic_backprop_guide_is_linked_from_mkdocs_nav() -> None:
    nav = MKDOCS.read_text(encoding="utf-8")

    assert "Stochastic Backpropagation: guides/stochastic_backprop.md" in nav


def test_training_api_references_joint_stochastic_backprop_surface() -> None:
    text = TRAINING_API.read_text(encoding="utf-8")

    assert "SCBackpropDesignSpace" in text
    assert "SCBackpropJointReport" in text
    assert "SCTrainingObjectiveConfig" in text
    assert "stochastic_backprop_joint_objective" in text
    assert "stochastic_backprop_benchmark.py" in text
    assert "local_simulation_and_executable_hdl_parity" in text


def test_generated_api_reference_exposes_joint_stochastic_backprop_surface() -> None:
    text = GENERATED_API.read_text(encoding="utf-8")

    assert "SCBackpropDesignSpace" in text
    assert "SCBackpropJointReport" in text
    assert "SCTrainingObjectiveConfig" in text
    assert "stochastic_backprop_joint_objective" in text
    assert "build_stochastic_backprop_estimator_regression_manifest" in text
    assert "write_stochastic_backprop_estimator_regression_manifest" in text
