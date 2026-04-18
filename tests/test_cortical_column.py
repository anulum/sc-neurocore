# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for canonical cortical microcircuit

"""Tests for CorticalColumn (Douglas & Martin 2004)."""

import numpy as np

from sc_neurocore.network.cortical_column import CorticalColumn


class TestCorticalColumn:
    def test_step_output_keys(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        spikes = col.step(np.ones(10))
        assert set(spikes.keys()) == {"l23_exc", "l23_inh", "l4", "l5", "l6"}

    def test_step_output_shapes(self):
        n = 8
        col = CorticalColumn(n_per_layer=n, seed=42)
        spikes = col.step(np.ones(n))
        for layer_spikes in spikes.values():
            assert layer_spikes.shape == (n,)

    def test_spikes_are_binary(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        for _ in range(20):
            spikes = col.step(np.ones(10) * 5.0)
        for layer_spikes in spikes.values():
            assert set(np.unique(layer_spikes)).issubset({0.0, 1.0})

    def test_thalamic_input_drives_l4(self):
        """Strong thalamic input should produce L4 spikes."""
        col = CorticalColumn(n_per_layer=10, w_exc=0.5, seed=42)
        total_l4 = 0.0
        for _ in range(50):
            spikes = col.step(np.ones(10) * 10.0)
            total_l4 += spikes["l4"].sum()
        assert total_l4 > 0

    def test_activity_propagates_to_l5(self):
        """Activity should propagate from L4 → L2/3 → L5."""
        col = CorticalColumn(n_per_layer=20, w_exc=1.0, threshold=0.5, seed=42)
        total_l5 = 0.0
        for _ in range(200):
            spikes = col.step(np.ones(20) * 20.0)
            total_l5 += spikes["l5"].sum()
        assert total_l5 > 0

    def test_inhibition_reduces_excitation(self):
        """L2/3 inhibitory activity should suppress L2/3 excitatory."""
        col_strong_inh = CorticalColumn(n_per_layer=10, w_inh=-0.5, w_exc=0.2, seed=42)
        col_weak_inh = CorticalColumn(n_per_layer=10, w_inh=-0.01, w_exc=0.2, seed=42)

        exc_strong = 0.0
        exc_weak = 0.0
        inp = np.ones(10) * 5.0
        for _ in range(50):
            s1 = col_strong_inh.step(inp)
            s2 = col_weak_inh.step(inp)
            exc_strong += s1["l23_exc"].sum()
            exc_weak += s2["l23_exc"].sum()
        # Stronger inhibition → fewer excitatory spikes (or equal)
        assert exc_strong <= exc_weak + 5  # small tolerance

    def test_run_output_shapes(self):
        col = CorticalColumn(n_per_layer=5, seed=42)
        results = col.run(np.ones(5) * 3.0, steps=20)
        for layer_data in results.values():
            assert layer_data.shape == (20, 5)

    def test_reset(self):
        col = CorticalColumn(n_per_layer=5, seed=42)
        col.run(np.ones(5) * 10.0, steps=20)
        col.reset()
        assert np.all(col.v_l4 == 0)
        assert np.all(col.v_l5 == 0)
        assert np.all(col.v_l23_exc == 0)

    def test_no_input_no_spikes(self):
        """Zero input should produce no spikes."""
        col = CorticalColumn(n_per_layer=5, seed=42)
        spikes = col.step(np.zeros(5))
        for layer_spikes in spikes.values():
            assert layer_spikes.sum() == 0

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        inp = np.ones(5) * 5.0
        col_a = CorticalColumn(n_per_layer=5, seed=99)
        col_b = CorticalColumn(n_per_layer=5, seed=99)
        r_a = col_a.run(inp, steps=10)
        r_b = col_b.run(inp, steps=10)
        for k in r_a:
            np.testing.assert_array_equal(r_a[k], r_b[k])
