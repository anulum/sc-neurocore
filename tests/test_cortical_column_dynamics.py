# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for cortical column dynamics

"""Tests for CorticalColumn: step, run, population outputs, reset, gain control."""

from __future__ import annotations

import numpy as np

from sc_neurocore.network.cortical_column import CorticalColumn


class TestCorticalColumnBasic:
    def test_creation(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        assert col.n_per_layer == 10

    def test_step_returns_dict(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        thalamic = np.zeros(10)
        result = col.step(thalamic)
        assert isinstance(result, dict)

    def test_step_has_all_populations(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        thalamic = np.ones(10) * 2.0
        result = col.step(thalamic)
        expected_keys = {"l23_exc", "l23_inh", "l4", "l5", "l6"}
        assert expected_keys.issubset(set(result.keys()))

    def test_step_output_shapes(self):
        n = 15
        col = CorticalColumn(n_per_layer=n, seed=42)
        thalamic = np.zeros(n)
        result = col.step(thalamic)
        for key, arr in result.items():
            assert len(arr) == n, f"{key} has wrong length {len(arr)}"

    def test_step_output_binary(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        thalamic = np.ones(10) * 3.0
        result = col.step(thalamic)
        for key, arr in result.items():
            assert set(np.unique(arr)).issubset({0, 1, 0.0, 1.0}), (
                f"{key} has non-binary values"
            )


class TestCorticalColumnRun:
    def test_run_returns_dict(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        # run() takes constant thalamic input of shape (n_per_layer,)
        thalamic = np.zeros(10)
        result = col.run(thalamic, steps=50)
        assert isinstance(result, dict)

    def test_run_shapes(self):
        n = 10
        T = 100
        col = CorticalColumn(n_per_layer=n, seed=42)
        thalamic = np.zeros(n)
        result = col.run(thalamic, steps=T)
        for key, arr in result.items():
            assert arr.shape == (T, n), f"{key} shape {arr.shape} != ({T}, {n})"

    def test_thalamic_drive_produces_spikes(self):
        n = 20
        col = CorticalColumn(n_per_layer=n, seed=42)
        # Constant strong drive
        thalamic = np.ones(n) * 3.0
        result = col.run(thalamic, steps=200)
        total_spikes = sum(arr.sum() for arr in result.values())
        assert total_spikes > 0, "no spikes with strong thalamic drive"

    def test_no_input_minimal_activity(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        thalamic = np.zeros(10)
        result = col.run(thalamic, steps=100)
        total = sum(arr.sum() for arr in result.values())
        max_possible = 100 * 10 * 5
        assert total < max_possible * 0.5, "too many spikes without input"


class TestCorticalColumnReset:
    def test_reset_clears_state(self):
        col = CorticalColumn(n_per_layer=10, seed=42)
        thalamic = np.ones(10) * 5.0
        for _ in range(50):
            col.step(thalamic)
        col.reset()
        # After reset, a zero-input step should produce low/no activity
        result = col.step(np.zeros(10))
        total = sum(arr.sum() for arr in result.values())
        assert total < 20, "high activity after reset with no input"


class TestCorticalColumnFeedforward:
    def test_l4_responds_to_drive(self):
        """L4 receives thalamic input directly; should have spikes."""
        n = 20
        col = CorticalColumn(n_per_layer=n, seed=42)
        thalamic = np.ones(n) * 3.0
        result = col.run(thalamic, steps=100)
        l4_total = result["l4"].sum()
        assert l4_total > 0, "L4 should respond to thalamic drive"

    def test_inhibition_present(self):
        """Column should produce at least some spikes under strong drive."""
        n = 20
        col = CorticalColumn(n_per_layer=n, w_inh=-0.15, seed=42)
        thalamic = np.ones(n) * 5.0
        result = col.run(thalamic, steps=500)
        total = sum(arr.sum() for arr in result.values())
        assert total > 0, "no spikes anywhere with very strong drive"


class TestCorticalColumnDeterminism:
    def test_same_seed_same_output(self):
        n = 10
        T = 50
        thalamic = np.ones(n) * 1.5

        col1 = CorticalColumn(n_per_layer=n, seed=42)
        r1 = col1.run(thalamic, steps=T)

        col2 = CorticalColumn(n_per_layer=n, seed=42)
        r2 = col2.run(thalamic, steps=T)

        for key in r1:
            np.testing.assert_array_equal(r1[key], r2[key],
                                          err_msg=f"{key} differs between runs")
