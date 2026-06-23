# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN L13 temporal binding layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l13_temporal import L13_StochasticParameters, L13_TemporalLayer


def test_l13_temporal_binding_uses_lagged_correlation() -> None:
    params = L13_StochasticParameters(
        n_channels=2,
        bitstream_length=16,
        binding_window=6,
        binding_threshold=0.7,
    )
    layer = L13_TemporalLayer(params)

    # Channel 1 is channel 0 delayed by one timestep. Zero-lag Pearson
    # stays weak, but max-lag binding should detect the temporal relation.
    inputs = [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    result = {}
    for values in inputs:
        result = layer.step(0.001, {"coherence": np.array(values, dtype=np.float64)})

    binding = result["binding_matrix"]
    assert abs(binding[0, 1]) > 0.9
    assert result["binding_strength"] == pytest.approx(1.0)


def test_l13_temporal_layer_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        L13_TemporalLayer(L13_StochasticParameters(n_channels=0))
    with pytest.raises(ValueError, match="binding_window"):
        L13_TemporalLayer(L13_StochasticParameters(binding_window=0))

    layer = L13_TemporalLayer(L13_StochasticParameters(n_channels=2))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="coherence"):
        layer.step(0.001, {"coherence": np.array([np.nan, 0.0])})


def test_l13_temporal_output_bitstreams_are_seed_scoped() -> None:
    params = L13_StochasticParameters(
        n_channels=3,
        bitstream_length=64,
        binding_window=3,
        rng_seed=1234,
    )
    layer_a = L13_TemporalLayer(params)
    layer_b = L13_TemporalLayer(params)
    drive = {"coherence": np.array([1.0, 0.5, 0.0], dtype=np.float64)}

    out_a0 = layer_a.step(0.001, drive)["output_bitstreams"]
    out_b0 = layer_b.step(0.001, drive)["output_bitstreams"]
    out_a1 = layer_a.step(0.001, drive)["output_bitstreams"]
    out_b1 = layer_b.step(0.001, drive)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)

    with pytest.raises(ValueError, match="rng_seed"):
        L13_TemporalLayer(L13_StochasticParameters(rng_seed=-1))
    with pytest.raises(ValueError, match="rng_seed"):
        L13_TemporalLayer(L13_StochasticParameters(rng_seed=cast(Any, 1.5)))


def test_l13_get_global_metric_off_diagonal_binding_mean() -> None:
    layer = L13_TemporalLayer(L13_StochasticParameters(n_channels=3, rng_seed=4))
    # A freshly-initialised binding matrix is zero, so the off-diagonal mean is 0.
    assert layer.get_global_metric() == 0.0


def test_l13_validate_params_bitstream_and_threshold_guards() -> None:
    with pytest.raises(ValueError, match="bitstream_length must be positive"):
        L13_TemporalLayer(L13_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match=r"binding_threshold must be in \[0, 1\]"):
        L13_TemporalLayer(L13_StochasticParameters(binding_threshold=1.5))


def test_l13_coherence_signal_empty_guard_and_pad() -> None:
    with pytest.raises(ValueError, match="coherence must contain at least one value"):
        L13_TemporalLayer._coherence_signal([], 4)
    padded = L13_TemporalLayer._coherence_signal([0.5], 4)
    assert padded.shape == (4,)
    assert padded.tolist() == [0.5, 0.0, 0.0, 0.0]


def test_l13_source_context_null_and_blank_branches() -> None:
    empty = L13_TemporalLayer._source_context(
        {"boundary_context_id": None, "boundary_terminals": ()}
    )
    assert empty["boundary_context_id"] is None
    assert empty["source_terminal_set"] == ()
    with pytest.raises(ValueError, match="boundary_context_id must be non-empty"):
        L13_TemporalLayer._source_context(
            {"boundary_context_id": "", "boundary_terminals": ("T5",)}
        )


def test_l13_scalar_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="must be a finite scalar"):
        L13_TemporalLayer._scalar(float("inf"), "gaian_stabilization_drive")
