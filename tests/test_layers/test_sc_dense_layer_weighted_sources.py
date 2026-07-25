# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCDenseLayer weighted-source contracts

"""Bipolar and per-neuron weighted-source contracts for SCDenseLayer."""

from tests.test_layers.sc_dense_layer_support import *


def test_dense_layer_passes_bipolar_mode_to_current_source():  # type: ignore[no-untyped-def] # Preserved legacy test AST
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


def test_dense_layer_accepts_per_neuron_weight_matrix():  # type: ignore[no-untyped-def] # Preserved legacy test AST
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


def test_dense_layer_matrix_weights_drive_distinct_neuron_spike_trains():  # type: ignore[no-untyped-def] # Preserved legacy test AST
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
