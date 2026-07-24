# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_descriptor_tiers.py

from __future__ import annotations

"""Tests for the science (S0-S5) and silicon (H0-H5) tier scoring.

The scoring must be a faithful, evidence-gated derivation: S0-S3 track the
curation kernel, S4/S5 open only on recorded validation evidence, and every
silicon rung is credited only alongside its proof anchor so a tier can never be
inflated ahead of the committed evidence (master plan invariant I7)."""

from typing import Any


import pytest


from sc_neurocore.neurons.descriptor_tiers import (
    SILICON_RUNGS,
    CompletenessTiers,
    completeness_tiers,
    is_perfect,
    science_tier,
    silicon_tier,
)


from sc_neurocore.neurons.model_descriptor import (
    descriptor_completeness_tier,
    parse_model_descriptor,
)


def _metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "schema_version": 2,
        "name": "AdEx",
        "class_name": "AdExNeuron",
        "module": "adex",
    }
    base.update(extra)
    return base


def _s0_payload() -> dict[str, Any]:
    """Structure only: identifies a real model, no taxonomy (science tier 0)."""
    return {
        "metadata": _metadata(),
        "state": {"v": {"init": -65.0}},
        "parameters": {"tau": {"default": 20.0}},
        "integration": {"dt": 0.1},
    }


def _s3_payload() -> dict[str, Any]:
    """A descriptor curated and engineering-verified to exactly science tier 3.

    Carries dynamics as well, so adding validation evidence can lift it to S4/S5.
    """
    return {
        "metadata": _metadata(family="Integrate-and-Fire", category="adaptive"),
        "provenance": {"authors": ["Brette"], "year": 2005, "doi": "10.1152/jn.00686.2005"},
        "state": {"v": {"init": -65.0}},
        "parameters": {
            "tau": {
                "default": 20.0,
                "unit": "ms",
                "range": [1.0, 100.0],
                "meaning": "time constant",
            }
        },
        "integration": {"dt": 0.1, "method": "euler"},
        "dynamics": {"v": "(-v + tau * I) / tau"},
        "backends": {
            "python": {"status": "implemented"},
            "rust": {"status": "implemented", "parity": "ulp-bounded"},
        },
        "reproducibility": {
            "reference_config": "golden/adex.json",
            "golden_trace_sha256": "a" * 64,
        },
    }


def _validated_payload() -> dict[str, Any]:
    """An S3 payload with full faithful-dynamics + class-validation evidence (S5)."""
    payload = _s3_payload()
    payload["validation"] = {
        "dynamics_faithful": True,
        "metric": "parity",
        "operating_point": "I = 5",
        "tolerance": "0 spikes",
        "evidence": "tests/test_cosimulation.py::TestRK4Emitter",
    }
    return payload


def _silicon(**flags: Any) -> Any:
    payload = _validated_payload()
    payload["silicon"] = flags
    return parse_model_descriptor(payload)


def _perfect_payload(target_tier: str = "H2") -> dict[str, Any]:
    payload = _validated_payload()
    payload["silicon"] = {
        "compiles": True,
        "cosim_validated": True,
        "cosim_evidence": "cosim.log",
        "synthesised": True,
        "synth_report": "yosys.json",
        "target_tier": target_tier,
        "terminal_reason": "point neuron; H2 is the deployable terminal",
    }
    return payload



__all__ = ['Any', 'pytest', 'SILICON_RUNGS', 'CompletenessTiers', 'completeness_tiers', 'is_perfect', 'science_tier', 'silicon_tier', 'descriptor_completeness_tier', 'parse_model_descriptor', '_metadata', '_s0_payload', '_s3_payload', '_validated_payload', '_silicon', '_perfect_payload']
