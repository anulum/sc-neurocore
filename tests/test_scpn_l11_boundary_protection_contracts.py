# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L11 boundary-protection contract tests

"""Production contracts for L11 consumption of L10 boundary-protection diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l11_morphic import L11_MorphicLayer, L11_StochasticParameters


def test_l11_l10_shielding_inhibits_memetic_percolation() -> None:
    params = L11_StochasticParameters(
        n_nodes=4,
        bitstream_length=16,
        j_coupling=0.0,
        h_bias=1.0,
        beta_infection=1.0,
        gamma_recovery=0.0,
        boundary_coupling=0.0,
        boundary_shielding_coupling=0.75,
        rng_seed=21,
    )
    weakly_shielded = L11_MorphicLayer(params)
    strongly_shielded = L11_MorphicLayer(params)

    weak = weakly_shielded.step(
        0.5,
        {
            "integrity": 0.9,
            "firewall_strength": np.zeros(4, dtype=np.float64),
            "topological_rejection_mask": np.zeros(4, dtype=bool),
            "boundary_complexity": 0.0,
            "qec_residual_load": 0.0,
            "memory_complexity_flux": 0.0,
        },
    )
    strong = strongly_shielded.step(
        0.5,
        {
            "integrity": 0.9,
            "firewall_strength": np.ones(4, dtype=np.float64),
            "topological_rejection_mask": np.array([True, True, False, False]),
            "boundary_complexity": 0.4,
            "qec_residual_load": 0.2,
            "memory_complexity_flux": 0.1,
        },
    )

    assert strong["boundary_shielding"] > weak["boundary_shielding"]
    assert strong["l10_rejection_fraction"] == pytest.approx(0.5)
    assert strong["info_saturation"] < weak["info_saturation"]


def test_l11_l10_residual_boundary_pressure_reduces_noospheric_alignment() -> None:
    params = L11_StochasticParameters(
        n_nodes=3,
        bitstream_length=16,
        j_coupling=0.0,
        h_bias=0.0,
        beta_infection=0.0,
        gamma_recovery=0.0,
        boundary_coupling=0.4,
        boundary_pressure_coupling=0.5,
        rng_seed=22,
    )
    coherent = L11_MorphicLayer(params)
    fragmented = L11_MorphicLayer(params)

    coherent_out = coherent.step(
        0.5,
        {
            "integrity": 0.8,
            "firewall_strength": np.zeros(3, dtype=np.float64),
            "topological_rejection_mask": np.zeros(3, dtype=bool),
            "boundary_complexity": 0.0,
            "qec_residual_load": 0.0,
        },
    )
    fragmented_out = fragmented.step(
        0.5,
        {
            "integrity": 0.8,
            "firewall_strength": np.zeros(3, dtype=np.float64),
            "topological_rejection_mask": np.array([True, False, False]),
            "boundary_complexity": 0.3,
            "qec_residual_load": 0.3,
        },
    )

    assert fragmented_out["boundary_fragmentation_pressure"] > 0.0
    assert np.mean(fragmented_out["spins"]) < np.mean(coherent_out["spins"])


def test_l11_uses_ebs_t3_t6_terminals_as_noospheric_bandwidth() -> None:
    params = L11_StochasticParameters(
        n_nodes=4,
        bitstream_length=16,
        j_coupling=0.0,
        h_bias=1.0,
        beta_infection=1.0,
        gamma_recovery=0.0,
        boundary_coupling=0.0,
        rng_seed=24,
    )
    no_noospheric_interface = L11_MorphicLayer(params)
    t3_t6_interface = L11_MorphicLayer(params)

    blocked = no_noospheric_interface.step(
        0.5,
        {
            "integrity": 0.9,
            "boundary_context_id": "ebs-l11",
            "boundary_terminals": ("T2", "T5"),
        },
    )
    admitted = t3_t6_interface.step(
        0.5,
        {
            "integrity": 0.9,
            "boundary_context_id": "ebs-l11",
            "boundary_terminals": ("T3", "T6"),
        },
    )

    assert blocked["noospheric_terminal_set"] == ()
    assert blocked["noospheric_terminal_bandwidth"] == pytest.approx(0.0)
    assert admitted["boundary_context_id"] == "ebs-l11"
    assert admitted["boundary_terminals"] == ("T3", "T6")
    assert admitted["noospheric_terminal_set"] == ("T3", "T6")
    assert admitted["noospheric_terminal_bandwidth"] == pytest.approx(1.0)
    assert admitted["info_saturation"] > blocked["info_saturation"]


def test_l11_rejects_invalid_l10_boundary_diagnostics() -> None:
    with pytest.raises(ValueError, match="boundary_shielding_coupling"):
        L11_MorphicLayer(L11_StochasticParameters(boundary_shielding_coupling=-0.1))
    with pytest.raises(ValueError, match="boundary_pressure_coupling"):
        L11_MorphicLayer(L11_StochasticParameters(boundary_pressure_coupling=-0.1))

    layer = L11_MorphicLayer(L11_StochasticParameters(n_nodes=3, rng_seed=23))
    invalid_payloads = [
        {"integrity": 0.5, "firewall_strength": np.array([0.0, np.nan, 1.0])},
        {"integrity": 0.5, "firewall_strength": np.array([], dtype=np.float64)},
        {"integrity": 0.5, "topological_rejection_mask": np.array([0.0, 0.5, 1.0])},
        {"integrity": 0.5, "boundary_complexity": -0.1},
        {"integrity": 0.5, "qec_residual_load": 1.1},
        {"integrity": 0.5, "memory_complexity_flux": np.array([0.1, 0.2])},
        {"integrity": 0.5, "boundary_context_id": "ebs"},
        {"integrity": 0.5, "boundary_terminals": ("T3",)},
        {"integrity": 0.5, "boundary_context_id": "ebs", "boundary_terminals": ("T8",)},
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            layer.step(0.001, payload)
