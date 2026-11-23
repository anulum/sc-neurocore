# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (layers) from former test_stochastic_core_properties.py

"""Property-based checks of the dense and vectorised SC layers."""

from __future__ import annotations

import numpy as np
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from sc_neurocore import (
    SCDenseLayer,
    VectorizedSCLayer,
)


@given(
    n_in=st.integers(min_value=2, max_value=8),
    n_out=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=20)
def test_dense_layer_produces_spikes(n_in: int, n_out: int) -> None:
    """A dense layer gives one train of the run length per output."""
    inputs = [0.5] * n_in
    weights = [0.3] * n_in
    layer = SCDenseLayer(
        n_neurons=n_out,
        x_inputs=inputs,
        weight_values=weights,
        x_min=0.0,
        x_max=1.0,
        w_min=-1.0,
        w_max=1.0,
        length=64,
    )
    layer.run(T=50)
    trains = layer.get_spike_trains()
    assert len(trains) == n_out
    assert all(len(t) == 50 for t in trains)


@given(
    n_in=st.integers(min_value=2, max_value=16),
    n_out=st.integers(min_value=2, max_value=16),
)
@settings(max_examples=20)
def test_vectorized_layer_output_shape(n_in: int, n_out: int) -> None:
    """A vectorised layer gives one output per neuron."""
    layer = VectorizedSCLayer(n_inputs=n_in, n_neurons=n_out, length=64)
    inp = np.random.rand(n_in)
    out = layer.forward(inp)
    assert out.shape == (n_out,)


@given(
    n_in=st.integers(min_value=2, max_value=8),
    n_out=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=10)
def test_vectorized_layer_output_bounded(n_in: int, n_out: int) -> None:
    """A vectorised layer gives finite outputs."""
    layer = VectorizedSCLayer(n_inputs=n_in, n_neurons=n_out, length=128)
    inp = np.random.rand(n_in)
    out = layer.forward(inp)
    assert np.all(np.isfinite(out))
