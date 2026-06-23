# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN L10 boundary firewall layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l10_boundary import L10_BoundaryLayer, L10_StochasticParameters


def test_l10_memory_coupling_parameter_controls_l9_drive() -> None:
    common = dict(
        n_boundary_nodes=4,
        bitstream_length=16,
        rejection_threshold=1.0,
        shielding_strength=1.0,
        steering_gain=0.2,
        rng_seed=10,
    )
    uncoupled = L10_BoundaryLayer(L10_StochasticParameters(**common, memory_coupling=0.0))
    coupled = L10_BoundaryLayer(L10_StochasticParameters(**common, memory_coupling=0.5))

    base = uncoupled.step(0.5, {"retrieval_quality": 0.8})["firewall_strength"]
    driven = coupled.step(0.5, {"retrieval_quality": 0.8})["firewall_strength"]

    np.testing.assert_allclose(driven - base, np.full(4, 0.04))


def test_l10_rejection_threshold_and_shielding_strength_control_noise_loss() -> None:
    common = dict(
        n_boundary_nodes=3,
        bitstream_length=16,
        steering_gain=0.0,
        memory_coupling=0.0,
        rng_seed=21,
    )
    sensitive = L10_BoundaryLayer(
        L10_StochasticParameters(
            **common,
            rejection_threshold=0.1,
            shielding_strength=1.0,
        )
    )
    tolerant = L10_BoundaryLayer(
        L10_StochasticParameters(
            **common,
            rejection_threshold=0.8,
            shielding_strength=1.0,
        )
    )
    hardened = L10_BoundaryLayer(
        L10_StochasticParameters(
            **common,
            rejection_threshold=0.1,
            shielding_strength=4.0,
        )
    )
    noise = np.full(3, 0.9, dtype=np.float64)

    sensitive_strength = sensitive.step(0.5, external_noise=noise)["firewall_strength"]
    tolerant_strength = tolerant.step(0.5, external_noise=noise)["firewall_strength"]
    hardened_strength = hardened.step(0.5, external_noise=noise)["firewall_strength"]

    assert float(np.mean(sensitive_strength)) < float(np.mean(tolerant_strength))
    assert float(np.mean(sensitive_strength)) < float(np.mean(hardened_strength))


def test_l10_seed_scopes_output_bitstreams() -> None:
    params = L10_StochasticParameters(
        n_boundary_nodes=3,
        bitstream_length=64,
        rejection_threshold=1.0,
        steering_gain=0.0,
        memory_coupling=0.0,
        rng_seed=123,
    )
    layer_a = L10_BoundaryLayer(params)
    layer_b = L10_BoundaryLayer(params)

    out_a0 = layer_a.step(0.001)["output_bitstreams"]
    out_b0 = layer_b.step(0.001)["output_bitstreams"]
    out_a1 = layer_a.step(0.001)["output_bitstreams"]
    out_b1 = layer_b.step(0.001)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l10_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_boundary_nodes"):
        L10_BoundaryLayer(L10_StochasticParameters(n_boundary_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L10_BoundaryLayer(L10_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="rejection_threshold"):
        L10_BoundaryLayer(L10_StochasticParameters(rejection_threshold=-0.1))
    with pytest.raises(ValueError, match="shielding_strength"):
        L10_BoundaryLayer(L10_StochasticParameters(shielding_strength=0.0))
    with pytest.raises(ValueError, match="steering_gain"):
        L10_BoundaryLayer(L10_StochasticParameters(steering_gain=-0.1))
    with pytest.raises(ValueError, match="memory_coupling"):
        L10_BoundaryLayer(L10_StochasticParameters(memory_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L10_BoundaryLayer(L10_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L10_BoundaryLayer(L10_StochasticParameters(n_boundary_nodes=2, rng_seed=5))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="retrieval_quality"):
        layer.step(0.001, {"retrieval_quality": np.nan})
    with pytest.raises(ValueError, match="retrieval_quality"):
        layer.step(0.001, {"retrieval_quality": np.array([0.1, 0.2])})
    with pytest.raises(ValueError, match="external_noise"):
        layer.step(0.001, external_noise=np.array([0.0, np.nan]))


def test_l10_short_external_noise_is_padded_before_step() -> None:
    layer = L10_BoundaryLayer()

    layer.step(0.01, external_noise=np.array([0.1]))

    assert layer.time > 0


def test_l10_get_global_metric_returns_integrity() -> None:
    layer = L10_BoundaryLayer(
        L10_StochasticParameters(n_boundary_nodes=4, bitstream_length=16, rng_seed=3)
    )
    assert 0.0 <= layer.get_global_metric() <= 1.0


def test_l10_validate_params_type_guards_and_negative_seed() -> None:
    with pytest.raises(ValueError, match="n_boundary_nodes must be a positive integer"):
        L10_BoundaryLayer(L10_StochasticParameters(n_boundary_nodes=cast(int, True)))
    with pytest.raises(ValueError, match="bitstream_length must be a positive integer"):
        L10_BoundaryLayer(L10_StochasticParameters(bitstream_length=cast(int, True)))
    with pytest.raises(ValueError, match="rng_seed"):
        L10_BoundaryLayer(L10_StochasticParameters(rng_seed=-1))


def test_l10_memory_complexity_flux_rejects_non_scalar() -> None:
    layer = L10_BoundaryLayer(
        L10_StochasticParameters(n_boundary_nodes=4, bitstream_length=16, rng_seed=2)
    )
    with pytest.raises(ValueError, match="memory_free_energy must be a finite scalar"):
        layer._memory_complexity_flux({"memory_free_energy": np.array([1.0, 2.0])})


def test_l10_boundary_context_null_and_blank_branches() -> None:
    empty = L10_BoundaryLayer._boundary_context(
        {"boundary_context_id": None, "boundary_terminals": ()}
    )
    assert empty["ebs_id"] is None
    with pytest.raises(ValueError, match="boundary_context_id must be non-empty"):
        L10_BoundaryLayer._boundary_context(
            {"boundary_context_id": "", "boundary_terminals": ("T1",)}
        )
