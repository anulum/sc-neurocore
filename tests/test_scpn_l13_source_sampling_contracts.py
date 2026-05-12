# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L13 Source-field sampling contract tests

"""Production contracts for L13 consumption of structured L12 Gaian diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l13_temporal import L13_StochasticParameters, L13_TemporalLayer


def test_l13_l12_gaian_stabilisation_raises_source_sampling_signal() -> None:
    params = L13_StochasticParameters(
        n_channels=3,
        bitstream_length=16,
        binding_window=3,
        binding_threshold=0.5,
        quantum_info_coupling=0.5,
        source_decoherence_coupling=0.0,
        rng_seed=41,
    )
    layer = L13_TemporalLayer(params)

    out = layer.step(
        0.5,
        {
            "coherence": np.full(3, 0.2, dtype=np.float64),
            "gaian_stabilization_drive": 0.4,
            "noospheric_entropy_load": 0.0,
            "effective_dephasing_gamma": 0.0,
        },
    )

    np.testing.assert_allclose(out["source_sampling_signal"], np.full(3, 0.4))
    assert out["source_sampling_gain"] == pytest.approx(0.2)
    assert out["temporal_decoherence_load"] == pytest.approx(0.0)


def test_l13_l12_entropy_and_dephasing_reduce_source_sampling_signal() -> None:
    params = L13_StochasticParameters(
        n_channels=4,
        bitstream_length=16,
        binding_window=3,
        binding_threshold=0.5,
        quantum_info_coupling=0.5,
        source_decoherence_coupling=0.25,
        rng_seed=42,
    )
    stable = L13_TemporalLayer(params)
    decohered = L13_TemporalLayer(params)

    stable_out = stable.step(
        0.5,
        {
            "coherence": np.full(4, 0.6, dtype=np.float64),
            "gaian_stabilization_drive": 0.2,
            "noospheric_entropy_load": 0.0,
            "effective_dephasing_gamma": 0.0,
        },
    )
    decohered_out = decohered.step(
        0.5,
        {
            "coherence": np.full(4, 0.6, dtype=np.float64),
            "gaian_stabilization_drive": 0.2,
            "noospheric_entropy_load": 0.4,
            "effective_dephasing_gamma": 0.4,
        },
    )

    assert decohered_out["temporal_decoherence_load"] == pytest.approx(0.8)
    assert np.mean(decohered_out["source_sampling_signal"]) < np.mean(
        stable_out["source_sampling_signal"]
    )


def test_l13_rejects_invalid_l12_source_sampling_contracts() -> None:
    with pytest.raises(ValueError, match="quantum_info_coupling"):
        L13_TemporalLayer(L13_StochasticParameters(quantum_info_coupling=-0.1))
    with pytest.raises(ValueError, match="source_decoherence_coupling"):
        L13_TemporalLayer(L13_StochasticParameters(source_decoherence_coupling=-0.1))

    layer = L13_TemporalLayer(L13_StochasticParameters(n_channels=2, rng_seed=43))
    invalid_payloads = [
        {"coherence": np.ones(2), "gaian_stabilization_drive": np.array([0.1, 0.2])},
        {"coherence": np.ones(2), "noospheric_entropy_load": -0.1},
        {"coherence": np.ones(2), "effective_dephasing_gamma": -0.1},
        {"coherence": np.array([1.2, 0.0])},
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            layer.step(0.001, payload)
