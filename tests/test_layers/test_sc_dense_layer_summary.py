# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCDenseLayer summary and reproducibility contracts

"""Summary evidence and seeded reproducibility for SCDenseLayer."""

from tests.test_layers.sc_dense_layer_support import *


def test_dense_summary_average_matches_mean():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Average firing rate in summary matches mean of stats."""
    layer = _make_layer(n_neurons=2)
    layer.run(10)
    summary = layer.summary()
    rates = [s["firing_rate_hz"] for s in summary["stats"]]
    assert np.isclose(summary["avg_firing_rate_hz"], float(np.mean(rates)))


def test_dense_summary_fields_present():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Summary includes expected keys and stat entries."""
    layer = _make_layer(n_neurons=2)
    layer.run(6)
    summary = layer.summary()
    assert summary["n_neurons"] == 2
    assert len(summary["stats"]) == 2
    assert {"neuron", "total_spikes", "firing_rate_hz"} <= set(summary["stats"][0].keys())


def test_dense_seed_reproducible():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Same seed and params yield identical spike trains."""
    layer_a = _make_layer(base_seed=77)
    layer_b = _make_layer(base_seed=77)
    layer_a.run(8)
    layer_b.run(8)
    assert np.array_equal(layer_a.get_spike_trains(), layer_b.get_spike_trains())
