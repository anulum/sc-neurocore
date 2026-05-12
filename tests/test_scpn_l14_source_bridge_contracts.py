# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L14 Source-bridge contract tests

"""Production contracts for L14 consumption of structured L13 Source-field diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l14_integration import (
    L14_IntegrationLayer,
    L14_StochasticParameters,
)


def test_l14_l13_source_sampling_opens_resonance_bridge() -> None:
    params = L14_StochasticParameters(
        n_dimensions=3,
        bitstream_length=16,
        integration_weights=np.ones(3, dtype=np.float64),
        temporal_coupling=1.0,
        bridge_decoherence_coupling=0.0,
        rng_seed=51,
    )
    layer = L14_IntegrationLayer(params)

    baseline = layer.step(0.01, {"l1": 0.2, "l2": 0.2, "l3": 0.2})
    driven = layer.step(
        0.01,
        {"l1": 0.2, "l2": 0.2, "l3": 0.2},
        l13_input={
            "source_sampling_signal": np.full(3, 0.6, dtype=np.float64),
            "source_sampling_gain": 0.2,
            "binding_strength": 0.3,
            "temporal_decoherence_load": 0.0,
        },
    )

    assert driven["transdimensional_bridge_drive"] > baseline["transdimensional_bridge_drive"]
    assert driven["layer_metrics"][-1] == pytest.approx(1.0)
    assert driven["integrated_coherence"] > baseline["integrated_coherence"]


def test_l14_l13_decoherence_load_reduces_bridge_integrity() -> None:
    params = L14_StochasticParameters(
        n_dimensions=3,
        bitstream_length=16,
        integration_weights=np.ones(3, dtype=np.float64),
        temporal_coupling=1.0,
        bridge_decoherence_coupling=0.5,
        rng_seed=52,
    )
    protected = L14_IntegrationLayer(params)
    decohered = L14_IntegrationLayer(params)

    common_l13 = {
        "source_sampling_signal": np.full(3, 0.6, dtype=np.float64),
        "source_sampling_gain": 0.1,
        "binding_strength": 0.1,
    }
    protected_out = protected.step(
        0.01,
        {"l1": 0.2, "l2": 0.2, "l3": 0.2},
        l13_input={**common_l13, "temporal_decoherence_load": 0.0},
    )
    decohered_out = decohered.step(
        0.01,
        {"l1": 0.2, "l2": 0.2, "l3": 0.2},
        l13_input={**common_l13, "temporal_decoherence_load": 0.8},
    )

    assert decohered_out["holographic_protection_load"] == pytest.approx(0.8)
    assert decohered_out["layer_metrics"][-1] < protected_out["layer_metrics"][-1]
    assert decohered_out["integrated_coherence"] < protected_out["integrated_coherence"]


def test_l14_rejects_invalid_l13_source_bridge_contracts() -> None:
    with pytest.raises(ValueError, match="bridge_decoherence_coupling"):
        L14_IntegrationLayer(L14_StochasticParameters(bridge_decoherence_coupling=-0.1))

    layer = L14_IntegrationLayer(L14_StochasticParameters(n_dimensions=3, bitstream_length=16))
    invalid_payloads = [
        {"source_sampling_signal": np.array([0.2, np.nan, 0.4])},
        {"source_sampling_signal": np.array([1.2, 0.0, 0.0])},
        {"source_sampling_signal": np.ones(3), "source_sampling_gain": np.array([0.1, 0.2])},
        {"source_sampling_signal": np.ones(3), "binding_strength": 1.2},
        {"source_sampling_signal": np.ones(3), "temporal_decoherence_load": -0.1},
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            layer.step(0.01, {"l1": 0.2}, l13_input=payload)
