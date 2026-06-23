# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN L12 quantum-information layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l12_quantum_info import (
    L12_QuantumInfoLayer,
    L12_StochasticParameters,
)


def test_l12_morphic_coupling_parameter_controls_info_drive() -> None:
    common = dict(
        n_sites=4,
        bitstream_length=16,
        transport_rate=0.0,
        dephasing_gamma=0.0,
        rng_seed=19,
    )
    uncoupled = L12_QuantumInfoLayer(L12_StochasticParameters(**common, morphic_coupling=0.0))
    coupled = L12_QuantumInfoLayer(L12_StochasticParameters(**common, morphic_coupling=0.25))

    np.testing.assert_allclose(uncoupled.coherence, coupled.coherence)
    base = uncoupled.step(0.5, {"info_saturation": 0.8})["coherence"]
    driven = coupled.step(0.5, {"info_saturation": 0.8})["coherence"]

    np.testing.assert_allclose(driven - base, np.full(4, 0.1))


def test_l12_seed_scopes_initial_state_and_output_bitstreams() -> None:
    params = L12_StochasticParameters(
        n_sites=3,
        bitstream_length=64,
        transport_rate=0.0,
        dephasing_gamma=0.0,
        rng_seed=123,
    )
    layer_a = L12_QuantumInfoLayer(params)
    layer_b = L12_QuantumInfoLayer(params)

    np.testing.assert_allclose(layer_a.coherence, layer_b.coherence)
    out_a0 = layer_a.step(0.001)["output_bitstreams"]
    out_b0 = layer_b.step(0.001)["output_bitstreams"]
    out_a1 = layer_a.step(0.001)["output_bitstreams"]
    out_b1 = layer_b.step(0.001)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l12_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_sites"):
        L12_QuantumInfoLayer(L12_StochasticParameters(n_sites=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L12_QuantumInfoLayer(L12_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="transport_rate"):
        L12_QuantumInfoLayer(L12_StochasticParameters(transport_rate=-0.1))
    with pytest.raises(ValueError, match="dephasing_gamma"):
        L12_QuantumInfoLayer(L12_StochasticParameters(dephasing_gamma=-0.1))
    with pytest.raises(ValueError, match="morphic_coupling"):
        L12_QuantumInfoLayer(L12_StochasticParameters(morphic_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L12_QuantumInfoLayer(L12_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L12_QuantumInfoLayer(L12_StochasticParameters(n_sites=2, rng_seed=4))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="info_saturation"):
        layer.step(0.001, {"info_saturation": np.nan})
    with pytest.raises(ValueError, match="info_saturation"):
        layer.step(0.001, {"info_saturation": np.array([0.1, 0.2])})


def test_l12_get_global_metric_returns_mean_coherence() -> None:
    layer = L12_QuantumInfoLayer(
        L12_StochasticParameters(n_sites=4, bitstream_length=16, rng_seed=3)
    )
    assert 0.0 <= layer.get_global_metric() <= 1.0


def test_l12_validate_params_type_guards_and_negative_seed() -> None:
    with pytest.raises(ValueError, match="n_sites must be a positive integer"):
        L12_QuantumInfoLayer(L12_StochasticParameters(n_sites=cast(int, True)))
    with pytest.raises(ValueError, match="bitstream_length must be a positive integer"):
        L12_QuantumInfoLayer(L12_StochasticParameters(bitstream_length=cast(int, True)))
    with pytest.raises(ValueError, match="rng_seed"):
        L12_QuantumInfoLayer(L12_StochasticParameters(rng_seed=-1))


def test_l12_gaian_context_null_and_blank_branches() -> None:
    empty = L12_QuantumInfoLayer._gaian_context(
        {"boundary_context_id": None, "boundary_terminals": ()}
    )
    assert empty["boundary_context_id"] is None
    with pytest.raises(ValueError, match="boundary_context_id must be non-empty"):
        L12_QuantumInfoLayer._gaian_context(
            {"boundary_context_id": "", "boundary_terminals": ("T1",)}
        )
