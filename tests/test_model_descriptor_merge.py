# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (merge) from former test_model_descriptor.py

from __future__ import annotations

from tests.model_descriptor_support import *  # noqa: F403

def test_merge_descriptor_payloads_preserves_curation_without_structural_drift() -> None:
    """The corpus merge keeps human curation without accepting stale structure."""

    regenerated = generate_descriptor_payload("AdExNeuron")
    curated: dict[str, Any] = {
        "metadata": {
            "schema_version": 999,
            "class_name": "WrongNeuron",
            "name": "Curated AdEx Descriptive Name",
            "display_name": "Adaptive exponential IF",
            "summary": "Curated AdEx descriptor.",
            "maturity": "validated",
            "intended_use": ["adaptive-spiking-reference"],
        },
        "parameters": {
            "tau": {
                "default": -1.0,
                "unit": "ms",
                "range": [1.0, 100.0],
                "meaning": "membrane time constant",
            },
            "stale_parameter": {"unit": "arb"},
        },
        "state": {"v": {"init": 0.0, "unit": "mV", "meaning": "membrane potential"}},
        "provenance": {"authors": ["Brette", "Gerstner"], "year": 2005},
        "dynamics": {"v": "curated membrane equation"},
        "backends": {"python": {"status": "implemented"}, "rust": {"status": "implemented"}},
        "reproducibility": {"reference_config": "golden/adex.json"},
        "validation": {
            "dynamics_faithful": True,
            "metric": "parity",
            "operating_point": "schema-DSL cosim Q16.16",
            "tolerance": "class-correct spike-count band",
            "evidence": "tests/test_cosimulation.py::TestQ1616Precision::test_adex_q1616_parity",
        },
        "silicon": {
            "compiles": True,
            "cosim_validated": True,
            "cosim_evidence": "tests/test_cosimulation.py::TestQ1616Precision::test_adex_q1616_parity",
            "target_tier": "H1",
            "terminal_reason": "point-neuron SC/RTL path; signed PPA out of scope",
        },
        "documentation": {
            "notes": "Preserved reviewer note.",
            "slug": "models/curated_adex_slug",
        },
    }

    merged = merge_descriptor_payloads(curated, regenerated)

    assert merged["metadata"]["schema_version"] == MODEL_DESCRIPTOR_SCHEMA_VERSION
    assert merged["metadata"]["class_name"] == "AdExNeuron"
    # The curated descriptive name and documentation slug are authoritative
    # overlays: a hand-written name/slug is never overwritten by the generic
    # generator default derived from the class/module name.
    assert merged["metadata"]["name"] == "Curated AdEx Descriptive Name"
    assert merged["metadata"]["name"] != regenerated["metadata"]["name"]
    assert merged["documentation"]["slug"] == "models/curated_adex_slug"
    assert merged["documentation"]["slug"] != regenerated["documentation"]["slug"]
    assert merged["metadata"]["display_name"] == "Adaptive exponential IF"
    assert merged["metadata"]["summary"] == "Curated AdEx descriptor."
    assert merged["metadata"]["intended_use"] == ["adaptive-spiking-reference"]
    assert merged["parameters"]["tau"]["default"] == regenerated["parameters"]["tau"]["default"]
    assert merged["parameters"]["tau"]["unit"] == "ms"
    assert "stale_parameter" not in merged["parameters"]
    assert merged["state"]["v"]["init"] == regenerated["state"]["v"]["init"]
    assert merged["state"]["v"]["unit"] == "mV"
    assert merged["provenance"]["authors"] == ["Brette", "Gerstner"]
    assert merged["dynamics"]["v"] == "curated membrane equation"
    assert "rust" in merged["backends"]
    assert merged["reproducibility"]["reference_config"] == "golden/adex.json"
    assert merged["documentation"]["notes"] == "Preserved reviewer note."
    assert merged["validation"]["dynamics_faithful"] is True
    assert merged["validation"]["metric"] == "parity"
    assert merged["validation"]["evidence"].endswith("test_adex_q1616_parity")
    assert merged["silicon"]["compiles"] is True
    assert merged["silicon"]["cosim_validated"] is True
    assert merged["silicon"]["target_tier"] == "H1"


