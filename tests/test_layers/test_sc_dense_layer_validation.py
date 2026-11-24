# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCDenseLayer validation contracts

"""Construction and weight-shape validation for SCDenseLayer."""

from sc_neurocore.layers.sc_dense_layer import SCDenseLayer
import pytest
from tests.test_layers.sc_dense_layer_support import (
    _make_layer,
)


def test_dense_init_mismatch_raises() -> None:
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


def test_dense_rejects_negative_neuron_count() -> None:
    """Layer size must not silently create invalid negative output shapes."""
    with pytest.raises(ValueError, match="n_neurons"):
        _make_layer(n_neurons=-1)


def test_dense_rejects_non_finite_shared_weight_vector() -> None:
    """A shared weight vector with an infinite value is rejected."""
    with pytest.raises(ValueError, match="only finite values"):
        _make_layer(n_neurons=1, weight_values=[float("inf"), 0.5])


def test_dense_rejects_2d_weights_without_output_neuron() -> None:
    """A weight matrix with no output neuron is rejected."""
    with pytest.raises(ValueError, match="at least one output neuron"):
        _make_layer(n_neurons=0, weight_values=[[0.1, 0.2]])


def test_dense_rejects_non_finite_weight_matrix() -> None:
    """A weight matrix with an infinite value is rejected."""
    with pytest.raises(ValueError, match="only finite values"):
        _make_layer(n_neurons=1, weight_values=[[float("inf"), 0.2]])


def test_dense_rejects_three_dimensional_weights() -> None:
    """Three-dimensional weights are rejected."""
    with pytest.raises(ValueError, match="1-D shared vector or a 2-D"):
        _make_layer(n_neurons=1, weight_values=[[[0.1, 0.2]]])


def test_dense_rejects_weight_matrix_with_wrong_input_width() -> None:
    """Per-neuron weight matrices must keep one weight per input channel."""
    with pytest.raises(ValueError, match="shape"):
        _make_layer(n_neurons=2, weight_values=[[0.5], [0.2]])


def test_dense_rejects_weight_matrix_with_wrong_output_count() -> None:
    """Per-neuron weight matrices must keep one row per output neuron."""
    with pytest.raises(ValueError, match="shape"):
        _make_layer(n_neurons=3, weight_values=[[0.5, 0.5], [0.2, 0.2]])
