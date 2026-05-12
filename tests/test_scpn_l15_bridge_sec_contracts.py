# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L15 bridge SEC contract tests

"""Production contracts for L15 consumption of L14 bridge/protection diagnostics."""

from __future__ import annotations

import pytest

from sc_neurocore.scpn.layers.l15_meta import L15_MetaLayer, L15_StochasticParameters


def test_l15_l14_bridge_drive_reduces_sec_dissonance() -> None:
    params = L15_StochasticParameters(
        n_monitors=4,
        bitstream_length=16,
        target_coherence=0.8,
        smoothing_alpha=1.0,
        integration_coupling=0.0,
        bridge_alignment_coupling=0.5,
        bridge_protection_coupling=0.0,
        rng_seed=61,
    )
    weak_bridge = L15_MetaLayer(params)
    strong_bridge = L15_MetaLayer(params)

    weak = weak_bridge.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.6,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
            "transdimensional_bridge_drive": 0.0,
            "holographic_protection_load": 0.0,
        },
    )
    strong = strong_bridge.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.6,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
            "transdimensional_bridge_drive": 0.4,
            "holographic_protection_load": 0.0,
        },
    )

    assert strong["bridge_alignment_credit"] == pytest.approx(0.2)
    assert strong["ethical_dissonance"] < weak["ethical_dissonance"]
    assert strong["oversoul_attractor"] > weak["oversoul_attractor"]


def test_l15_l14_holographic_protection_load_penalises_sec() -> None:
    params = L15_StochasticParameters(
        n_monitors=4,
        bitstream_length=16,
        target_coherence=0.8,
        smoothing_alpha=1.0,
        integration_coupling=1.0,
        bridge_alignment_coupling=0.0,
        bridge_protection_coupling=0.5,
        rng_seed=62,
    )
    protected = L15_MetaLayer(params)
    overloaded = L15_MetaLayer(params)

    protected_out = protected.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.8,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
            "holographic_protection_load": 0.0,
        },
    )
    overloaded_out = overloaded.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.8,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
            "holographic_protection_load": 0.6,
        },
    )

    assert overloaded_out["bridge_protection_penalty"] == pytest.approx(0.3)
    assert overloaded_out["free_energy"] > protected_out["free_energy"]
    assert overloaded_out["gci"] < protected_out["gci"]


def test_l15_uses_ebs_t2_t4_t5_t6_terminals_as_consilium_bandwidth() -> None:
    params = L15_StochasticParameters(
        n_monitors=4,
        bitstream_length=16,
        target_coherence=0.8,
        smoothing_alpha=1.0,
        integration_coupling=0.0,
        bridge_alignment_coupling=0.5,
        bridge_protection_coupling=0.0,
        rng_seed=63,
    )
    no_consilium_interface = L15_MetaLayer(params)
    full_consilium_interface = L15_MetaLayer(params)

    blocked = no_consilium_interface.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.6,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
            "transdimensional_bridge_drive": 0.4,
            "boundary_context_id": "ebs-l15",
            "boundary_terminals": ("T1", "T3"),
        },
    )
    admitted = full_consilium_interface.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.6,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
            "transdimensional_bridge_drive": 0.4,
            "boundary_context_id": "ebs-l15",
            "boundary_terminals": ("T2", "T4", "T5", "T6"),
        },
    )

    assert blocked["consilium_terminal_set"] == ()
    assert blocked["consilium_terminal_bandwidth"] == pytest.approx(0.0)
    assert admitted["boundary_context_id"] == "ebs-l15"
    assert admitted["boundary_terminals"] == ("T2", "T4", "T5", "T6")
    assert admitted["consilium_terminal_set"] == ("T2", "T4", "T5", "T6")
    assert admitted["consilium_terminal_bandwidth"] == pytest.approx(1.0)
    assert admitted["bridge_alignment_credit"] > blocked["bridge_alignment_credit"]
    assert admitted["ethical_dissonance"] < blocked["ethical_dissonance"]


def test_l15_rejects_invalid_l14_bridge_sec_contracts() -> None:
    with pytest.raises(ValueError, match="bridge_alignment_coupling"):
        L15_MetaLayer(L15_StochasticParameters(bridge_alignment_coupling=-0.1))
    with pytest.raises(ValueError, match="bridge_protection_coupling"):
        L15_MetaLayer(L15_StochasticParameters(bridge_protection_coupling=-0.1))

    layer = L15_MetaLayer(L15_StochasticParameters(n_monitors=2, bitstream_length=8))
    invalid_inputs = [
        {"integrated_coherence": 0.8, "transdimensional_bridge_drive": -0.1},
        {"integrated_coherence": 0.8, "transdimensional_bridge_drive": 1.1},
        {"integrated_coherence": 0.8, "holographic_protection_load": -0.1},
        {"integrated_coherence": 0.8, "holographic_protection_load": [0.1, 0.2]},
        {"integrated_coherence": 0.8, "boundary_context_id": "ebs"},
        {"integrated_coherence": 0.8, "boundary_terminals": ("T4",)},
        {"integrated_coherence": 0.8, "boundary_context_id": "ebs", "boundary_terminals": ("T8",)},
    ]

    for l14_input in invalid_inputs:
        with pytest.raises(ValueError):
            layer.step(0.01, l14_input=l14_input)
