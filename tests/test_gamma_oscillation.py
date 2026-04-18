# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PING gamma oscillation circuit

"""Tests for PING gamma oscillation circuit."""

import numpy as np

from sc_neurocore.network.gamma_oscillation import PINGCircuit


class TestPINGCircuit:
    def test_creates_default(self):
        ping = PINGCircuit()
        assert ping.n_excitatory == 80
        assert ping.n_inhibitory == 20
        assert ping.v_e.shape == (80,)
        assert ping.v_i.shape == (20,)

    def test_produces_spikes(self):
        np.random.seed(42)
        ping = PINGCircuit()
        total_e, total_i = 0, 0
        for _ in range(500):
            se, si = ping.step(drive=5.0, dt=0.1)
            total_e += se.sum()
            total_i += si.sum()
        assert total_e > 0
        assert total_i > 0

    def test_no_drive_no_spikes(self):
        np.random.seed(42)
        ping = PINGCircuit()
        total = 0
        for _ in range(100):
            se, si = ping.step(drive=0.0, dt=0.1)
            total += se.sum() + si.sum()
        # With zero drive and noise, very few or no spikes
        assert total < 20

    def test_inhibition_suppresses(self):
        np.random.seed(42)
        # Strong inhibition should suppress excitatory firing
        ping_strong = PINGCircuit(w_ie=2.0)
        ping_weak = PINGCircuit(w_ie=0.1)
        e_strong, e_weak = 0, 0
        for _ in range(300):
            se, _ = ping_strong.step(drive=5.0, dt=0.1)
            e_strong += se.sum()
            se2, _ = ping_weak.step(drive=5.0, dt=0.1)
            e_weak += se2.sum()
        assert e_strong < e_weak

    def test_reset(self):
        ping = PINGCircuit()
        for _ in range(100):
            ping.step(drive=5.0, dt=0.1)
        ping.reset_state()
        assert np.all(ping.v_e < 0.5)
        assert np.all(ping.v_i < 0.5)


class TestPINGCircuitDeterminism:
    """Two PINGCircuit instances built with the same seed produce identical output."""

    def test_init_voltages_match_for_same_seed(self):
        a = PINGCircuit(seed=123)
        b = PINGCircuit(seed=123)
        np.testing.assert_array_equal(a.v_e, b.v_e)
        np.testing.assert_array_equal(a.v_i, b.v_i)

    def test_init_voltages_differ_for_different_seeds(self):
        a = PINGCircuit(seed=1)
        b = PINGCircuit(seed=2)
        # At least one of v_e / v_i must differ between distinct seeds
        differ = (not np.array_equal(a.v_e, b.v_e)) or (not np.array_equal(a.v_i, b.v_i))
        assert differ

    def test_step_sequence_identical_for_same_seed(self):
        """500 steps of two same-seed instances yield identical spike trains."""
        a = PINGCircuit(seed=99)
        b = PINGCircuit(seed=99)
        for _ in range(500):
            sa_e, sa_i = a.step(drive=5.0, dt=0.1)
            sb_e, sb_i = b.step(drive=5.0, dt=0.1)
            np.testing.assert_array_equal(sa_e, sb_e)
            np.testing.assert_array_equal(sa_i, sb_i)

    def test_global_numpy_seed_does_not_leak_in(self):
        """Setting np.random.seed must not affect PINGCircuit determinism."""
        np.random.seed(0)
        a = PINGCircuit(seed=42)
        a_spikes_e, a_spikes_i = [], []
        for _ in range(100):
            se, si = a.step(drive=5.0, dt=0.1)
            a_spikes_e.append(se.copy())
            a_spikes_i.append(si.copy())

        # Re-seed global RNG to a different value; build a fresh instance
        # with the SAME PINGCircuit seed; expect identical spike trains.
        np.random.seed(99999)
        b = PINGCircuit(seed=42)
        for t in range(100):
            sb_e, sb_i = b.step(drive=5.0, dt=0.1)
            np.testing.assert_array_equal(a_spikes_e[t], sb_e)
            np.testing.assert_array_equal(a_spikes_i[t], sb_i)

    def test_total_spike_count_constant_across_runs(self):
        """No more 78% spike-count spread across identical-param runs (the v3.14.0 bug)."""
        counts = []
        for _ in range(5):
            ping = PINGCircuit(seed=42)
            total = 0
            for _ in range(500):
                se, si = ping.step(drive=5.0, dt=0.1)
                total += int(se.sum()) + int(si.sum())
            counts.append(total)
        # All five runs must agree exactly
        assert len(set(counts)) == 1, f"non-deterministic: spike totals = {counts}"

    def test_reset_state_uses_per_instance_rng(self):
        """reset_state must not call the global numpy RNG."""
        ping = PINGCircuit(seed=42)
        # Drain a few steps to advance the per-instance RNG
        for _ in range(10):
            ping.step(drive=5.0, dt=0.1)

        # Set a wildly different global seed; reset_state must ignore it
        np.random.seed(9999)
        ping.reset_state()
        v_e_after = ping.v_e.copy()

        # Repeat with the global RNG in a different state — same per-instance
        # state must produce the same reset_state result
        ping2 = PINGCircuit(seed=42)
        for _ in range(10):
            ping2.step(drive=5.0, dt=0.1)
        np.random.seed(1)
        ping2.reset_state()
        np.testing.assert_array_equal(v_e_after, ping2.v_e)
