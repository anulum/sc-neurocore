# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler-service response contracts

"""Response serialisation and diagnostic tests for the compiler boundary."""

from __future__ import annotations

from sc_neurocore.compiler_service import build_compiler_service_response
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
)

from .compiler_service_support import _request


def test_build_response_serialises_optimizer_report() -> None:
    report = SurrogateOptimizerReport(
        config={
            "hidden": SurrogateLayerConfig(
                bitstream_length=256,
                decorrelator="LFSR",
                mode="SC",
                precision_bits=8,
                lfsr_polynomial="x16+x14+x13+x11+1",
                luts_used=512,
                power_used=4.5,
                latency_cycles=256,
                accuracy_score=0.98,
                utility_score=0.91,
            )
        },
        total_luts=512,
        total_power_mw=4.5,
        total_latency_cycles=256,
        mean_accuracy=0.98,
        training_points=32,
        target_name="pynq-z2",
    )

    response = build_compiler_service_response(
        _request("bitstream_length"), optimiser_report=report
    )
    payload = response.to_dict()

    assert payload["accepted"] is True
    assert payload["optimiser_report"]["feasible"] is True
    assert payload["optimiser_report"]["config"]["hidden"]["bitstream_length"] == 256
    assert payload["update_package"]["kind"] == "hot_swap"


def test_response_diagnostics_vary_by_update_kind() -> None:
    """Full-resynthesis and partial-reconfiguration packages each carry their
    own dedicated diagnostic message, distinct from the hot-swap default."""
    full = build_compiler_service_response(_request("clock")).to_dict()
    partial = build_compiler_service_response(_request("aer_route_overlay")).to_dict()

    assert full["update_package"]["kind"] == "full_resynthesis_required"
    assert "full synthesis toolchain" in full["diagnostics"][0]
    assert partial["update_package"]["kind"] == "partial_reconfiguration"
    assert "partial reconfiguration" in partial["diagnostics"][0]


def test_response_without_optimiser_report_serialises_null() -> None:
    """A response built without an optimiser report serialises the report slot
    as null rather than an empty object."""
    payload = build_compiler_service_response(_request("weights")).to_dict()
    assert payload["optimiser_report"] is None
