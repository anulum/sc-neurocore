# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L16 bridge-closure contract tests

"""Production contracts for L16 consumption of L15 SEC bridge diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l16_director import L16_DirectorLayer, L16_StochasticParameters


def test_l16_l15_bridge_alignment_credit_reduces_recursive_correction() -> None:
    params = L16_StochasticParameters(
        n_control_nodes=3,
        bitstream_length=16,
        kp=1.0,
        ki=0.0,
        target_gci=0.8,
        meta_coupling=0.0,
        bridge_alignment_coupling=1.0,
        bridge_protection_coupling=0.0,
        rng_seed=71,
    )
    unaligned = L16_DirectorLayer(params)
    aligned = L16_DirectorLayer(params)

    unaligned_out = unaligned.step(
        0.1,
        l15_input={
            "gci": 0.6,
            "ethical_dissonance": 0.0,
            "free_energy": 0.0,
            "bridge_alignment_credit": 0.0,
            "bridge_protection_penalty": 0.0,
        },
    )
    aligned_out = aligned.step(
        0.1,
        l15_input={
            "gci": 0.6,
            "ethical_dissonance": 0.0,
            "free_energy": 0.0,
            "bridge_alignment_credit": 0.15,
            "bridge_protection_penalty": 0.0,
        },
    )

    assert aligned_out["closure_bridge_alignment_credit"] == pytest.approx(0.15)
    assert aligned_out["control_signal"] < unaligned_out["control_signal"]
    assert aligned_out["recursive_hamiltonian"] < unaligned_out["recursive_hamiltonian"]


def test_l16_l15_bridge_protection_penalty_drives_entropy_qecc() -> None:
    params = L16_StochasticParameters(
        n_control_nodes=4,
        bitstream_length=16,
        kp=1.0,
        ki=0.0,
        veto_threshold=0.5,
        target_gci=0.8,
        meta_coupling=0.0,
        bridge_alignment_coupling=0.0,
        bridge_protection_coupling=1.0,
        rng_seed=72,
    )
    protected = L16_DirectorLayer(params)
    overloaded = L16_DirectorLayer(params)

    protected_out = protected.step(
        0.1,
        l15_input={
            "gci": 0.8,
            "ethical_dissonance": 0.0,
            "free_energy": 0.0,
            "bridge_protection_penalty": 0.0,
        },
    )
    overloaded_out = overloaded.step(
        0.1,
        l15_input={
            "gci": 0.8,
            "ethical_dissonance": 0.0,
            "free_energy": 0.0,
            "bridge_protection_penalty": 0.7,
        },
    )

    assert overloaded_out["closure_bridge_protection_penalty"] == pytest.approx(0.7)
    assert overloaded_out["entropy_flux"] > protected_out["entropy_flux"]
    assert overloaded_out["qecc_syndrome"].sum() > protected_out["qecc_syndrome"].sum()


def test_l16_rejects_invalid_l15_bridge_closure_contracts() -> None:
    with pytest.raises(ValueError, match="bridge_alignment_coupling"):
        L16_DirectorLayer(L16_StochasticParameters(bridge_alignment_coupling=-0.1))
    with pytest.raises(ValueError, match="bridge_protection_coupling"):
        L16_DirectorLayer(L16_StochasticParameters(bridge_protection_coupling=-0.1))

    layer = L16_DirectorLayer(L16_StochasticParameters(n_control_nodes=2, bitstream_length=8))
    invalid_inputs = [
        {"gci": 0.8, "bridge_alignment_credit": -0.1},
        {"gci": 0.8, "bridge_alignment_credit": 1.1},
        {"gci": 0.8, "bridge_protection_penalty": -0.1},
        {"gci": 0.8, "bridge_protection_penalty": np.array([0.1, 0.2])},
    ]

    for l15_input in invalid_inputs:
        with pytest.raises(ValueError):
            layer.step(0.01, l15_input=l15_input)
