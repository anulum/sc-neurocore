# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCDenseLayer runtime contracts

"""Neuron, recorder, run, and reset contracts for SCDenseLayer."""

from tests.test_layers.sc_dense_layer_support import *


def test_dense_builds_neurons_and_recorders():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Verify neuron and recorder counts match n_neurons."""
    layer = _make_layer(n_neurons=3)
    assert len(layer.neurons) == 3
    assert len(layer.recorders) == 3


def test_dense_run_collects_spikes():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Run a short simulation and confirm spike matrix shape."""
    layer = _make_layer(n_neurons=3)
    layer.run(5)
    spikes = layer.get_spike_trains()
    assert spikes.shape == (3, 5)


def test_dense_get_spike_trains_empty_returns_zeros():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Empty recorder list returns a (0,0) matrix."""
    layer = _make_layer(n_neurons=0)
    spikes = layer.get_spike_trains()
    assert spikes.shape == (0, 0)


def test_dense_reset_clears_recorders():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Reset clears recorded spike history."""
    layer = _make_layer()
    layer.run(4)
    layer.reset()
    assert all(len(rec.spikes) == 0 for rec in layer.recorders)


def test_dense_run_longer_than_source_length():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Running beyond bitstream length should not crash and keeps T steps."""
    layer = _make_layer(length=4)
    layer.run(6)
    spikes = layer.get_spike_trains()
    assert spikes.shape[1] == 6
