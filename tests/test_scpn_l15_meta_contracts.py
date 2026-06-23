# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L15 Consilium integration contract tests

"""Production contracts for the SCPN L15 Consilium stochastic layer."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l15_meta import L15_MetaLayer, L15_StochasticParameters


def test_l15_seed_scopes_output_bitstreams() -> None:
    params = L15_StochasticParameters(n_monitors=4, bitstream_length=128, rng_seed=1515)

    first = L15_MetaLayer(params)
    second = L15_MetaLayer(params)

    l14_input = {"integrated_coherence": 0.8, "resonance_lock": True, "resonance_determinant": 0.0}
    first_step = first.step(0.01, l14_input=l14_input)
    second_step = second.step(0.01, l14_input=l14_input)
    next_step = first.step(0.01, l14_input=l14_input)

    np.testing.assert_array_equal(first_step["output_bitstreams"], second_step["output_bitstreams"])
    assert not np.array_equal(first_step["output_bitstreams"], next_step["output_bitstreams"])


def test_l15_uses_l14_resonance_and_integration_coupling() -> None:
    coupled = L15_MetaLayer(
        L15_StochasticParameters(
            n_monitors=3,
            bitstream_length=32,
            target_coherence=0.9,
            smoothing_alpha=0.5,
            integration_coupling=0.8,
            rng_seed=7,
        )
    )
    decoupled = L15_MetaLayer(
        L15_StochasticParameters(
            n_monitors=3,
            bitstream_length=32,
            target_coherence=0.9,
            smoothing_alpha=0.5,
            integration_coupling=0.0,
            rng_seed=7,
        )
    )

    l14_input = {
        "integrated_coherence": 0.7,
        "resonance_lock": False,
        "resonance_determinant": 0.25,
    }
    coupled_out = coupled.step(0.05, l14_input=l14_input)
    decoupled_out = decoupled.step(0.05, l14_input=l14_input)

    assert coupled_out["actual_coherence"] == pytest.approx(0.7)
    assert coupled_out["error"] == pytest.approx(0.2)
    assert coupled_out["ethical_dissonance"] == pytest.approx(0.4)
    assert coupled_out["free_energy"] == pytest.approx(0.16)
    assert coupled_out["oversoul_attractor"] == pytest.approx(0.6)
    assert coupled_out["gci"] == pytest.approx(0.54)
    assert decoupled_out["gci"] == pytest.approx(0.5)


def test_l15_tracks_consilium_trend_with_bounded_metrics() -> None:
    layer = L15_MetaLayer(
        L15_StochasticParameters(
            n_monitors=4,
            bitstream_length=16,
            target_coherence=0.75,
            smoothing_alpha=1.0,
            integration_coupling=1.0,
            rng_seed=21,
        )
    )

    first = layer.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.75,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
        },
    )
    second = layer.step(
        0.1,
        l14_input={
            "integrated_coherence": 0.25,
            "resonance_lock": True,
            "resonance_determinant": 0.0,
        },
    )

    assert first["gci"] == pytest.approx(1.0)
    assert second["gci"] == pytest.approx(0.5)
    assert second["error_trend"] == pytest.approx(0.125)
    assert second["umo_weights"].shape == (4,)
    assert np.all(second["umo_weights"] == pytest.approx(np.full(4, 0.25)))
    assert 0.0 <= layer.get_global_metric() <= 1.0


def test_l15_rejects_invalid_parameters_and_inputs() -> None:
    invalid_params = [
        {"n_monitors": 0},
        {"bitstream_length": 0},
        {"target_coherence": -0.1},
        {"target_coherence": 1.1},
        {"smoothing_alpha": -0.1},
        {"smoothing_alpha": 1.1},
        {"integration_coupling": -0.1},
        {"integration_coupling": 1.1},
        {"rng_seed": -1},
    ]
    for kwargs in invalid_params:
        with pytest.raises(ValueError):
            L15_MetaLayer(L15_StochasticParameters(**kwargs))

    layer = L15_MetaLayer(L15_StochasticParameters(n_monitors=2, bitstream_length=8, rng_seed=1))

    invalid_steps = [
        (0.0, None),
        (float("nan"), None),
        (0.01, {"integrated_coherence": -0.1}),
        (0.01, {"integrated_coherence": 1.1}),
        (0.01, {"integrated_coherence": float("nan")}),
        (0.01, {"integrated_coherence": 0.5, "resonance_lock": "yes"}),
        (0.01, {"integrated_coherence": 0.5, "resonance_determinant": float("inf")}),
    ]
    for dt, l14_input in invalid_steps:
        with pytest.raises(ValueError):
            layer.step(dt, l14_input=l14_input)


def test_l15_validate_params_bridge_coupling_bounds() -> None:
    with pytest.raises(ValueError, match="bridge_alignment_coupling must be finite and in"):
        L15_MetaLayer(L15_StochasticParameters(bridge_alignment_coupling=1.5))
    with pytest.raises(ValueError, match="bridge_protection_coupling must be finite and in"):
        L15_MetaLayer(L15_StochasticParameters(bridge_protection_coupling=1.5))


def test_l15_step_without_l14_input_uses_neutral_defaults() -> None:
    layer = L15_MetaLayer(L15_StochasticParameters(rng_seed=3))
    assert layer.step(0.01) is not None


def test_l15_step_requires_integrated_coherence() -> None:
    layer = L15_MetaLayer(L15_StochasticParameters(rng_seed=3))
    with pytest.raises(ValueError, match="must include integrated_coherence"):
        layer.step(0.01, {"resonance_lock": True})


def test_l15_consilium_context_branches() -> None:
    ctx = L15_MetaLayer._consilium_context
    # A partial boundary context (one key only) is rejected.
    with pytest.raises(ValueError, match="boundary context requires"):
        ctx({"boundary_context_id": "ctx"})
    # A null id with no terminals yields the empty consilium context.
    empty = ctx({"boundary_context_id": None, "boundary_terminals": ()})
    assert empty["boundary_context_id"] is None
    # A blank id is rejected.
    with pytest.raises(ValueError, match="boundary_context_id must be non-empty"):
        ctx({"boundary_context_id": "", "boundary_terminals": ("T2",)})
    # An unrecognised terminal is rejected.
    with pytest.raises(ValueError, match="valid T1-T7"):
        ctx({"boundary_context_id": "ctx", "boundary_terminals": ("T9",)})
    # A valid context computes the consilium terminal subset and bandwidth.
    valid = ctx({"boundary_context_id": "ctx", "boundary_terminals": ("T2", "T4")})
    assert valid["consilium_terminal_set"] == ("T2", "T4")
    assert valid["consilium_terminal_bandwidth"] == 0.5


def test_l15_scalar_helpers_guard_shape_sign_and_unit_range() -> None:
    with pytest.raises(ValueError, match="must be a finite scalar"):
        L15_MetaLayer._nonnegative_scalar(np.array([1.0, 2.0]), "x")
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        L15_MetaLayer._nonnegative_scalar(-1.0, "x")
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        L15_MetaLayer._unit_scalar(1.5, "x")
