# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L10 holographic QEC boundary contract tests

"""Production contracts for L10 consumption of L9 holographic QEC diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l10_boundary import L10_BoundaryLayer, L10_StochasticParameters


def test_l10_l9_qec_residual_drives_topological_rejection() -> None:
    params = L10_StochasticParameters(
        n_boundary_nodes=4,
        bitstream_length=16,
        rejection_threshold=0.1,
        shielding_strength=1.0,
        steering_gain=0.0,
        memory_coupling=0.0,
        qec_coupling=1.0,
        rng_seed=10,
    )
    residual = L10_BoundaryLayer(params)
    recovered = L10_BoundaryLayer(params)

    l9_residual = {
        "retrieval_quality": 0.0,
        "qec_syndrome": np.array([1, 0, 1, 0], dtype=np.uint8),
        "recovery_operator": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        "memory_free_energy": 0.0,
    }
    l9_recovered = {
        "retrieval_quality": 0.0,
        "qec_syndrome": np.array([1, 0, 1, 0], dtype=np.uint8),
        "recovery_operator": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        "memory_free_energy": 0.0,
    }

    residual_out = residual.step(0.1, l9_input=l9_residual)
    recovered_out = recovered.step(0.1, l9_input=l9_recovered)

    assert residual_out["qec_residual_load"] == pytest.approx(0.25)
    assert recovered_out["qec_residual_load"] == pytest.approx(0.0)
    np.testing.assert_array_equal(
        residual_out["topological_rejection_mask"], np.array([True, False, False, False])
    )
    assert residual_out["boundary_complexity"] > recovered_out["boundary_complexity"]
    assert residual_out["integrity"] < recovered_out["integrity"]


def test_l10_memory_free_energy_adds_bounded_complexity_flux() -> None:
    layer = L10_BoundaryLayer(
        L10_StochasticParameters(
            n_boundary_nodes=3,
            bitstream_length=16,
            rejection_threshold=0.0,
            shielding_strength=2.0,
            steering_gain=0.0,
            memory_coupling=0.0,
            qec_coupling=0.5,
            rng_seed=4,
        )
    )

    out = layer.step(
        0.1,
        l9_input={
            "retrieval_quality": 0.0,
            "qec_syndrome": np.zeros(3, dtype=np.uint8),
            "recovery_operator": np.zeros(3, dtype=np.float64),
            "memory_free_energy": 0.6,
        },
    )

    assert out["memory_complexity_flux"] == pytest.approx(0.3)
    assert out["dissonance"] == pytest.approx(0.3)
    assert out["boundary_complexity"] == pytest.approx(0.3)


def test_l10_projects_l9_qec_vectors_into_boundary_space() -> None:
    layer = L10_BoundaryLayer(
        L10_StochasticParameters(
            n_boundary_nodes=5,
            bitstream_length=16,
            rejection_threshold=1.0,
            steering_gain=0.0,
            memory_coupling=0.0,
            qec_coupling=1.0,
            rng_seed=6,
        )
    )

    out = layer.step(
        0.1,
        l9_input={
            "retrieval_quality": 0.0,
            "qec_syndrome": np.array([1.0, 1.0, 0.0], dtype=np.float64),
            "recovery_operator": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        },
    )

    assert out["qec_residual_load"] == pytest.approx(0.2)


def test_l10_rejects_invalid_l9_qec_payloads() -> None:
    with pytest.raises(ValueError, match="qec_coupling"):
        L10_BoundaryLayer(L10_StochasticParameters(qec_coupling=-0.1))
    with pytest.raises(ValueError, match="qec_coupling"):
        L10_BoundaryLayer(L10_StochasticParameters(qec_coupling=1.1))

    layer = L10_BoundaryLayer(L10_StochasticParameters(n_boundary_nodes=3, rng_seed=3))

    invalid_payloads = [
        {"retrieval_quality": 0.0, "qec_syndrome": np.array([1.0, np.nan, 0.0])},
        {"retrieval_quality": 0.0, "qec_syndrome": np.array([], dtype=np.float64)},
        {
            "retrieval_quality": 0.0,
            "qec_syndrome": np.zeros(3),
            "recovery_operator": np.array([0.0, 2.0, 0.0]),
        },
        {"retrieval_quality": 0.0, "memory_free_energy": -0.1},
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            layer.step(0.001, l9_input=payload)
