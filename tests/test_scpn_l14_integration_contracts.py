# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L14 integration layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l14_integration import (
    L14_IntegrationLayer,
    L14_StochasticParameters,
)


def test_l14_seed_scopes_output_bitstreams() -> None:
    params = L14_StochasticParameters(n_dimensions=4, bitstream_length=128, rng_seed=123)
    layer_a = L14_IntegrationLayer(params)
    layer_b = L14_IntegrationLayer(params)
    metrics = {"l1": 0.2, "l2": 0.4, "l3": 0.6, "l4": 0.8}

    out_a0 = layer_a.step(0.01, metrics)["output_bitstreams"]
    out_b0 = layer_b.step(0.01, metrics)["output_bitstreams"]
    out_a1 = layer_a.step(0.01, metrics)["output_bitstreams"]
    out_b1 = layer_b.step(0.01, metrics)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l14_umo_weights_are_normalised_and_resonance_lock_is_reported() -> None:
    params = L14_StochasticParameters(
        n_dimensions=3,
        bitstream_length=16,
        integration_weights=np.array([1.0, 1.0, 2.0]),
        resonance_lock_tolerance=1e-9,
        rng_seed=456,
    )
    layer = L14_IntegrationLayer(params)

    result = layer.step(0.01, {"l1": 0.25, "l2": 0.5, "l3": 1.0})

    assert result["integrated_coherence"] == pytest.approx(0.6875)

    locked = layer.step(0.01, {"l1": 0.5, "l2": 0.5, "l3": 0.5})
    assert locked["resonance_determinant"] == pytest.approx(0.0)
    assert locked["resonance_lock"] is True


def test_l14_temporal_coupling_uses_validated_l13_signal() -> None:
    params = L14_StochasticParameters(
        n_dimensions=3,
        bitstream_length=16,
        integration_weights=np.ones(3),
        temporal_coupling=0.5,
        rng_seed=789,
    )
    layer = L14_IntegrationLayer(params)

    baseline = layer.step(0.01, {"l1": 0.1, "l2": 0.1, "l3": 0.1})["integrated_coherence"]
    driven = layer.step(
        0.01,
        {"l1": 0.1, "l2": 0.1, "l3": 0.1},
        l13_input={"source_field": np.ones(3)},
    )["integrated_coherence"]

    assert driven > baseline


def test_l14_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_dimensions"):
        L14_IntegrationLayer(L14_StochasticParameters(n_dimensions=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L14_IntegrationLayer(L14_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="integration_weights"):
        L14_IntegrationLayer(
            L14_StochasticParameters(n_dimensions=3, integration_weights=np.ones(2))
        )
    with pytest.raises(ValueError, match="integration_weights"):
        L14_IntegrationLayer(
            L14_StochasticParameters(
                n_dimensions=3, integration_weights=np.array([1.0, np.nan, 1.0])
            )
        )
    with pytest.raises(ValueError, match="integration_weights"):
        L14_IntegrationLayer(
            L14_StochasticParameters(n_dimensions=3, integration_weights=np.array([1.0, -1.0, 1.0]))
        )
    with pytest.raises(ValueError, match="temporal_coupling"):
        L14_IntegrationLayer(L14_StochasticParameters(temporal_coupling=-0.1))
    with pytest.raises(ValueError, match="resonance_lock_tolerance"):
        L14_IntegrationLayer(L14_StochasticParameters(resonance_lock_tolerance=0.0))
    with pytest.raises(ValueError, match="rng_seed"):
        L14_IntegrationLayer(L14_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L14_IntegrationLayer(L14_StochasticParameters(n_dimensions=3, bitstream_length=16))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="layer_metrics"):
        layer.step(0.01, {"l1": 0.2, "l2": np.nan})
    with pytest.raises(ValueError, match="source_field"):
        layer.step(0.01, {"l1": 0.2}, l13_input={"source_field": np.array([0.5, np.nan])})


def test_l14_requires_initialised_integration_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = L14_StochasticParameters(n_dimensions=4, bitstream_length=16, rng_seed=1)
    params.integration_weights = None
    # _validate_params also rejects a null matrix, so suppress it to reach the
    # layer's own defensive None-narrowing guard before _normalised_weights.
    monkeypatch.setattr(L14_IntegrationLayer, "_validate_params", staticmethod(lambda _p: None))
    with pytest.raises(ValueError, match="integration_weights must be initialised"):
        L14_IntegrationLayer(params)


def test_l14_get_global_metric_and_negative_seed() -> None:
    layer = L14_IntegrationLayer(
        L14_StochasticParameters(n_dimensions=4, bitstream_length=16, rng_seed=2)
    )
    layer.step(0.01, {"l1": 0.5, "l2": 0.5})
    assert layer.get_global_metric() == layer.integrated_coherence
    with pytest.raises(ValueError, match="rng_seed"):
        L14_IntegrationLayer(L14_StochasticParameters(rng_seed=-1))


def test_l14_metric_vector_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"layer_metrics must be within \[0, 1\]"):
        L14_IntegrationLayer._metric_vector({"a": 1.5}, 10)


def test_l14_bridge_effect_and_context_neutral_branches() -> None:
    # No source signal/field yields a zero bridge drive.
    effect = L14_IntegrationLayer._l13_bridge_effect({})
    assert effect["bridge_drive"] == 0.0
    # A null context id with no terminals yields the empty bridge context.
    empty = L14_IntegrationLayer._bridge_context(
        {"boundary_context_id": None, "boundary_terminals": ()}
    )
    assert empty["boundary_context_id"] is None
    # A blank context id paired with terminals is rejected.
    with pytest.raises(ValueError, match="boundary_context_id must be non-empty"):
        L14_IntegrationLayer._bridge_context(
            {"boundary_context_id": "", "boundary_terminals": ("T2",)}
        )
