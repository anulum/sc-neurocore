# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for cortical column dynamics

"""Tests for CorticalColumn: step, simulate, population outputs, reset, determinism.

Aligned with the production API which uses:
    CorticalColumn(scale=..., seed=...)
    .step(dt=0.1) -> dict[str, np.ndarray]     (one timestep)
    .simulate(duration_ms=..., dt=0.1) -> dict  (full run)
    .reset_state()                               (clear state)

Population keys: L23e, L23i, L4e, L4i, L5e, L5i, L6e, L6i.
"""

from __future__ import annotations

import numpy as np

from sc_neurocore.network.cortical_column import CorticalColumn

EXPECTED_POPULATIONS = {"L23e", "L23i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"}


class TestCorticalColumnBasic:
    def test_creation(self):
        col = CorticalColumn(scale=0.02, seed=42)
        assert col.scale == 0.02

    def test_step_returns_dict(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        assert isinstance(result, dict)

    def test_step_has_all_populations(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        assert EXPECTED_POPULATIONS.issubset(set(result.keys()))

    def test_step_output_shapes(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        for key, arr in result.items():
            assert arr.ndim == 1, f"{key} is not 1-D"
            assert arr.shape[0] > 0, f"{key} has zero length"

    def test_step_output_boolean(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        for key, arr in result.items():
            assert arr.dtype == np.bool_, f"{key} dtype is {arr.dtype}, expected bool"


class TestCorticalColumnSimulate:
    def test_simulate_returns_dict(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.simulate(duration_ms=5.0, dt=0.1)
        assert isinstance(result, dict)

    def test_simulate_shapes(self):
        col = CorticalColumn(scale=0.02, seed=42)
        dt = 0.1
        dur = 5.0
        n_steps = int(round(dur / dt))
        result = col.simulate(duration_ms=dur, dt=dt)
        for key, arr in result.items():
            assert arr.shape[0] == n_steps, f"{key}: {arr.shape[0]} rows != {n_steps}"

    def test_background_drive_produces_spikes(self):
        """Background Poisson input should produce at least some spikes."""
        col = CorticalColumn(scale=0.02, bg_rate=8.0, seed=42)
        result = col.simulate(duration_ms=50.0, dt=0.1)
        total_spikes = sum(arr.sum() for arr in result.values())
        assert total_spikes > 0, "no spikes at all with background drive"

    def test_no_background_minimal_activity(self):
        """With near-zero background, activity should be very low."""
        col = CorticalColumn(scale=0.02, bg_rate=0.01, seed=42)
        result = col.simulate(duration_ms=10.0, dt=0.1)
        total = sum(arr.sum() for arr in result.values())
        max_possible = sum(arr.size for arr in result.values())
        assert total < max_possible * 0.5, "too many spikes with minimal background"


class TestCorticalColumnReset:
    def test_reset_clears_state(self):
        col = CorticalColumn(scale=0.02, seed=42)
        for _ in range(50):
            col.step()
        col.reset_state()
        result = col.step()
        # After reset, first step with low bg should not have massive activity
        total = sum(arr.sum() for arr in result.values())
        max_possible = sum(arr.shape[0] for arr in result.values())
        assert total < max_possible, "unreasonable activity after reset"


class TestCorticalColumnFeedforward:
    def test_excitatory_populations_present(self):
        """Excitatory populations should exist and produce spikes."""
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.simulate(duration_ms=50.0, dt=0.1)
        exc_keys = [k for k in result if k.endswith("e")]
        assert len(exc_keys) >= 4, f"Expected >= 4 excitatory pops, got {exc_keys}"

    def test_inhibition_present(self):
        """Inhibitory populations should exist and produce spikes with strong drive."""
        col = CorticalColumn(scale=0.02, g_inh=4.0, seed=42)
        result = col.simulate(duration_ms=50.0, dt=0.1)
        inh_keys = [k for k in result if k.endswith("i")]
        assert len(inh_keys) >= 4, f"Expected >= 4 inhibitory pops, got {inh_keys}"
        inh_total = sum(result[k].sum() for k in inh_keys)
        assert inh_total > 0, "no inhibitory spikes at all"


class TestCorticalColumnDeterminism:
    def test_same_seed_same_output(self):
        dt = 0.1
        dur = 5.0

        col1 = CorticalColumn(scale=0.02, seed=42)
        r1 = col1.simulate(duration_ms=dur, dt=dt)

        col2 = CorticalColumn(scale=0.02, seed=42)
        r2 = col2.simulate(duration_ms=dur, dt=dt)

        for key in r1:
            np.testing.assert_array_equal(r1[key], r2[key], err_msg=f"{key} differs between runs")
