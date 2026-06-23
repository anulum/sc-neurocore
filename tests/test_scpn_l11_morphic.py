# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN L11 morphic layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l11_morphic import L11_MorphicLayer, L11_StochasticParameters


def test_l11_boundary_coupling_parameter_controls_l10_drive() -> None:
    common = dict(
        n_nodes=4,
        bitstream_length=16,
        j_coupling=0.0,
        h_bias=0.0,
        beta_infection=0.0,
        gamma_recovery=0.0,
        rng_seed=11,
    )
    uncoupled = L11_MorphicLayer(L11_StochasticParameters(**common, boundary_coupling=0.0))
    coupled = L11_MorphicLayer(L11_StochasticParameters(**common, boundary_coupling=0.25))

    base = uncoupled.step(0.4, {"integrity": 0.8})["spins"]
    driven = coupled.step(0.4, {"integrity": 0.8})["spins"]

    np.testing.assert_allclose(driven - base, np.full(4, 0.08))


def test_l11_memetic_parameters_control_information_density() -> None:
    common = dict(
        n_nodes=3,
        bitstream_length=16,
        j_coupling=0.0,
        h_bias=1.0,
        boundary_coupling=0.0,
        rng_seed=12,
    )
    inert = L11_MorphicLayer(
        L11_StochasticParameters(**common, beta_infection=0.0, gamma_recovery=0.0)
    )
    spreading = L11_MorphicLayer(
        L11_StochasticParameters(**common, beta_infection=0.5, gamma_recovery=0.0)
    )

    inert_out = inert.step(0.2)["info_saturation"]
    spreading_out = spreading.step(0.2)["info_saturation"]

    assert inert_out == pytest.approx(0.0)
    assert spreading_out > 0.0

    spreading.info_density[:] = 0.5
    recovered = spreading.step(0.2)["info_saturation"]
    recovering = L11_MorphicLayer(
        L11_StochasticParameters(**common, beta_infection=0.0, gamma_recovery=0.5)
    )
    recovering.info_density[:] = 0.5
    recovered_without_infection = recovering.step(0.2)["info_saturation"]

    assert recovered > recovered_without_infection
    assert recovered_without_infection < 0.5


def test_l11_seed_scopes_output_bitstreams() -> None:
    params = L11_StochasticParameters(
        n_nodes=3,
        bitstream_length=64,
        j_coupling=0.0,
        h_bias=0.0,
        beta_infection=0.0,
        gamma_recovery=0.0,
        rng_seed=123,
    )
    layer_a = L11_MorphicLayer(params)
    layer_b = L11_MorphicLayer(params)

    out_a0 = layer_a.step(0.001)["output_bitstreams"]
    out_b0 = layer_b.step(0.001)["output_bitstreams"]
    out_a1 = layer_a.step(0.001)["output_bitstreams"]
    out_b1 = layer_b.step(0.001)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l11_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        L11_MorphicLayer(L11_StochasticParameters(n_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L11_MorphicLayer(L11_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="j_coupling"):
        L11_MorphicLayer(L11_StochasticParameters(j_coupling=np.inf))
    with pytest.raises(ValueError, match="h_bias"):
        L11_MorphicLayer(L11_StochasticParameters(h_bias=np.nan))
    with pytest.raises(ValueError, match="beta_infection"):
        L11_MorphicLayer(L11_StochasticParameters(beta_infection=-0.1))
    with pytest.raises(ValueError, match="gamma_recovery"):
        L11_MorphicLayer(L11_StochasticParameters(gamma_recovery=-0.1))
    with pytest.raises(ValueError, match="boundary_coupling"):
        L11_MorphicLayer(L11_StochasticParameters(boundary_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L11_MorphicLayer(L11_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L11_MorphicLayer(L11_StochasticParameters(n_nodes=2, rng_seed=5))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="integrity"):
        layer.step(0.001, {"integrity": np.nan})
    with pytest.raises(ValueError, match="integrity"):
        layer.step(0.001, {"integrity": np.array([0.1, 0.2])})


def test_l11_get_global_metric_returns_mean_spin() -> None:
    layer = L11_MorphicLayer(L11_StochasticParameters(n_nodes=4, rng_seed=3))
    assert layer.get_global_metric() == pytest.approx(float(np.mean(layer.spins)))


def test_l11_validate_params_type_guards_and_negative_seed() -> None:
    with pytest.raises(ValueError, match="n_nodes must be a positive integer"):
        L11_MorphicLayer(L11_StochasticParameters(n_nodes=cast(int, True)))
    with pytest.raises(ValueError, match="bitstream_length must be a positive integer"):
        L11_MorphicLayer(L11_StochasticParameters(bitstream_length=cast(int, True)))
    with pytest.raises(ValueError, match="rng_seed"):
        L11_MorphicLayer(L11_StochasticParameters(rng_seed=-1))


def test_l11_l10_boundary_effect_rejects_out_of_range_integrity() -> None:
    layer = L11_MorphicLayer(L11_StochasticParameters(n_nodes=2, rng_seed=5))
    with pytest.raises(ValueError, match=r"integrity must be within \[0, 1\]"):
        layer.step(0.001, {"integrity": 2.0})


def test_l11_noospheric_context_null_and_blank_branches() -> None:
    # A null context id with no terminals yields the empty noospheric context.
    empty = L11_MorphicLayer._noospheric_context(
        {"boundary_context_id": None, "boundary_terminals": ()}
    )
    assert empty["boundary_context_id"] is None
    assert empty["noospheric_terminal_set"] == ()
    # A blank context id paired with terminals is rejected.
    with pytest.raises(ValueError, match="boundary_context_id must be non-empty"):
        L11_MorphicLayer._noospheric_context(
            {"boundary_context_id": "", "boundary_terminals": ("T1",)}
        )


def test_l11_project_nonnegative_vector_negative_and_pad() -> None:
    with pytest.raises(ValueError, match="firewall must be non-negative"):
        L11_MorphicLayer._project_nonnegative_vector([1.0, -1.0], 4, "firewall")
    padded = L11_MorphicLayer._project_nonnegative_vector([1.0, 2.0], 4, "firewall")
    assert padded.shape == (4,)
    assert padded.tolist() == [1.0, 2.0, 0.0, 0.0]


def test_l11_project_rejection_mask_branches() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        L11_MorphicLayer._project_rejection_mask([], 4)
    with pytest.raises(ValueError, match="only finite values"):
        L11_MorphicLayer._project_rejection_mask([0.0, np.nan], 4)
    # A numeric (non-bool) 0/1 mask is coerced to bool and padded to n_nodes.
    mask = L11_MorphicLayer._project_rejection_mask(np.array([1, 0]), 4)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, False, False]
