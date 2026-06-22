# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCDenseLayer core behavior and edge cases

"""Tests for SCDenseLayer core behavior and edge cases."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.sc_dense_layer import SCDenseLayer


def _make_layer(**overrides):
    params = dict(
        n_neurons=2,
        x_inputs=[0.2, 0.4],
        weight_values=[0.5, 0.5],
        x_min=0.0,
        x_max=1.0,
        w_min=0.0,
        w_max=1.0,
        length=16,
        dt_ms=1.0,
        neuron_params={"noise_std": 0.0, "tau_mem": 1e9},
        base_seed=123,
    )
    params.update(overrides)
    return SCDenseLayer(**params)


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_dense_init_mismatch_raises():
    """Ensure mismatched input/weight lengths raise ValueError."""
    with pytest.raises(ValueError):
        SCDenseLayer(
            n_neurons=1,
            x_inputs=[0.1],
            weight_values=[0.2, 0.3],
            x_min=0.0,
            x_max=1.0,
            w_min=0.0,
            w_max=1.0,
        )


def test_dense_rejects_negative_neuron_count():
    """Layer size must not silently create invalid negative output shapes."""
    with pytest.raises(ValueError, match="n_neurons"):
        _make_layer(n_neurons=-1)


def test_dense_rejects_non_finite_shared_weight_vector():
    with pytest.raises(ValueError, match="only finite values"):
        _make_layer(n_neurons=1, weight_values=[float("inf"), 0.5])


def test_dense_rejects_2d_weights_without_output_neuron():
    with pytest.raises(ValueError, match="at least one output neuron"):
        _make_layer(n_neurons=0, weight_values=[[0.1, 0.2]])


def test_dense_rejects_non_finite_weight_matrix():
    with pytest.raises(ValueError, match="only finite values"):
        _make_layer(n_neurons=1, weight_values=[[float("inf"), 0.2]])


def test_dense_rejects_three_dimensional_weights():
    with pytest.raises(ValueError, match="1-D shared vector or a 2-D"):
        _make_layer(n_neurons=1, weight_values=[[[0.1, 0.2]]])


def test_dense_rejects_weight_matrix_with_wrong_input_width():
    """Per-neuron weight matrices must keep one weight per input channel."""
    with pytest.raises(ValueError, match="shape"):
        _make_layer(n_neurons=2, weight_values=[[0.5], [0.2]])


def test_dense_rejects_weight_matrix_with_wrong_output_count():
    """Per-neuron weight matrices must keep one row per output neuron."""
    with pytest.raises(ValueError, match="shape"):
        _make_layer(n_neurons=3, weight_values=[[0.5, 0.5], [0.2, 0.2]])


def test_dense_builds_neurons_and_recorders():
    """Verify neuron and recorder counts match n_neurons."""
    layer = _make_layer(n_neurons=3)
    assert len(layer.neurons) == 3
    assert len(layer.recorders) == 3


def test_dense_run_collects_spikes():
    """Run a short simulation and confirm spike matrix shape."""
    layer = _make_layer(n_neurons=3)
    layer.run(5)
    spikes = layer.get_spike_trains()
    assert spikes.shape == (3, 5)


def test_dense_get_spike_trains_empty_returns_zeros():
    """Empty recorder list returns a (0,0) matrix."""
    layer = _make_layer(n_neurons=0)
    spikes = layer.get_spike_trains()
    assert spikes.shape == (0, 0)


def test_dense_reset_clears_recorders():
    """Reset clears recorded spike history."""
    layer = _make_layer()
    layer.run(4)
    layer.reset()
    assert all(len(rec.spikes) == 0 for rec in layer.recorders)


def test_dense_summary_average_matches_mean():
    """Average firing rate in summary matches mean of stats."""
    layer = _make_layer(n_neurons=2)
    layer.run(10)
    summary = layer.summary()
    rates = [s["firing_rate_hz"] for s in summary["stats"]]
    assert np.isclose(summary["avg_firing_rate_hz"], float(np.mean(rates)))


def test_dense_summary_fields_present():
    """Summary includes expected keys and stat entries."""
    layer = _make_layer(n_neurons=2)
    layer.run(6)
    summary = layer.summary()
    assert summary["n_neurons"] == 2
    assert len(summary["stats"]) == 2
    assert {"neuron", "total_spikes", "firing_rate_hz"} <= set(summary["stats"][0].keys())


def test_dense_seed_reproducible():
    """Same seed and params yield identical spike trains."""
    layer_a = _make_layer(base_seed=77)
    layer_b = _make_layer(base_seed=77)
    layer_a.run(8)
    layer_b.run(8)
    assert np.array_equal(layer_a.get_spike_trains(), layer_b.get_spike_trains())


def test_dense_run_longer_than_source_length():
    """Running beyond bitstream length should not crash and keeps T steps."""
    layer = _make_layer(length=4)
    layer.run(6)
    spikes = layer.get_spike_trains()
    assert spikes.shape[1] == 6


def test_dense_layer_passes_bipolar_mode_to_current_source():
    """SCDenseLayer should expose signed XNOR SC inference through its source."""
    layer = _make_layer(
        x_inputs=[1.0],
        x_min=-1.0,
        x_max=1.0,
        weight_values=[-1.0],
        w_min=-1.0,
        w_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        sc_mode="bipolar",
    )

    assert layer.source.sc_mode == "bipolar"
    assert np.isclose(layer.source.full_current_estimate(), -1.0)


def test_dense_layer_accepts_per_neuron_weight_matrix():
    """A dense layer should support distinct SC dot-product weights per neuron."""
    layer = _make_layer(
        n_neurons=2,
        x_inputs=[1.0, 1.0],
        weight_values=[[1.0, 1.0], [0.0, 0.0]],
        y_min=0.0,
        y_max=1.0,
        length=64,
    )

    assert len(layer.sources) == 2
    assert layer.source is layer.sources[0]
    assert layer.sources[0].full_current_estimate() == pytest.approx(1.0)
    assert layer.sources[1].full_current_estimate() == pytest.approx(0.0)


def test_dense_layer_matrix_weights_drive_distinct_neuron_spike_trains():
    """Per-neuron sources should feed their matching neurons during run()."""
    layer = _make_layer(
        n_neurons=2,
        x_inputs=[1.0, 1.0],
        weight_values=[[1.0, 1.0], [0.0, 0.0]],
        y_min=0.0,
        y_max=1.0,
        length=32,
        neuron_params={
            "noise_std": 0.0,
            "tau_mem": 1e9,
            "v_threshold": 0.5,
            "resistance": 1.0,
        },
    )

    layer.run(16)
    spikes = layer.get_spike_trains()

    assert spikes[0].sum() == 16
    assert spikes[1].sum() == 0


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_dense_layer_perf_small():
    """Benchmark a small run for basic performance sanity."""
    layer = _make_layer(n_neurons=8, length=64)
    start = time.perf_counter()
    layer.run(64)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
