# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L12 noospheric-Gaian contract tests

"""Production contracts for L12 consumption of structured L11 noospheric diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l12_quantum_info import (
    L12_QuantumInfoLayer,
    L12_StochasticParameters,
)


def test_l12_damps_fragmented_noospheric_entropy_load() -> None:
    params = L12_StochasticParameters(
        n_sites=4,
        bitstream_length=16,
        transport_rate=0.0,
        dephasing_gamma=0.0,
        morphic_coupling=0.2,
        noospheric_entropy_coupling=0.5,
        rng_seed=31,
    )
    coherent = L12_QuantumInfoLayer(params)
    fragmented = L12_QuantumInfoLayer(params)

    coherent_out = coherent.step(
        0.5,
        {
            "info_saturation": 0.8,
            "boundary_shielding": 1.0,
            "boundary_fragmentation_pressure": 0.0,
            "polarization": 0.0,
        },
    )
    fragmented_out = fragmented.step(
        0.5,
        {
            "info_saturation": 0.8,
            "boundary_shielding": 0.0,
            "boundary_fragmentation_pressure": 0.4,
            "polarization": 0.2,
        },
    )

    assert fragmented_out["noospheric_entropy_load"] > coherent_out["noospheric_entropy_load"]
    assert fragmented_out["gaian_stabilization_drive"] < coherent_out["gaian_stabilization_drive"]
    assert np.mean(fragmented_out["coherence"]) < np.mean(coherent_out["coherence"])


def test_l12_noospheric_entropy_load_raises_effective_dephasing() -> None:
    params = L12_StochasticParameters(
        n_sites=3,
        bitstream_length=16,
        transport_rate=0.0,
        dephasing_gamma=0.1,
        morphic_coupling=0.0,
        noospheric_entropy_coupling=0.4,
        rng_seed=32,
    )
    quiet = L12_QuantumInfoLayer(params)
    volatile = L12_QuantumInfoLayer(params)

    quiet_out = quiet.step(
        0.5,
        {
            "info_saturation": 0.5,
            "boundary_shielding": 1.0,
            "boundary_fragmentation_pressure": 0.0,
        },
    )
    volatile_out = volatile.step(
        0.5,
        {
            "info_saturation": 0.5,
            "boundary_shielding": 0.0,
            "boundary_fragmentation_pressure": 0.5,
        },
    )

    assert volatile_out["effective_dephasing_gamma"] > quiet_out["effective_dephasing_gamma"]
    assert volatile_out["transport_efficiency"] < quiet_out["transport_efficiency"]


def test_l12_rejects_invalid_noospheric_gaian_contracts() -> None:
    with pytest.raises(ValueError, match="noospheric_entropy_coupling"):
        L12_QuantumInfoLayer(L12_StochasticParameters(noospheric_entropy_coupling=-0.1))

    layer = L12_QuantumInfoLayer(L12_StochasticParameters(n_sites=2, rng_seed=33))
    invalid_payloads = [
        {"info_saturation": -0.1, "boundary_shielding": 0.0},
        {"info_saturation": 0.5, "boundary_shielding": 1.1},
        {"info_saturation": 0.5, "boundary_fragmentation_pressure": -0.1},
        {"info_saturation": 0.5, "polarization": np.array([0.1, 0.2])},
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            layer.step(0.001, payload)
