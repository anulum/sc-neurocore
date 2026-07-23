# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPINGCircuitDeterminism from former test_gamma_oscillation.py

"""Focused suite: TestPINGCircuitDeterminism from former test_gamma_oscillation.py."""

from __future__ import annotations

from tests.gamma_oscillation_support import *  # noqa: F403

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
        differ = (not np.array_equal(a.v_e, b.v_e)) or (not np.array_equal(a.v_i, b.v_i))
        assert differ

    def test_step_sequence_identical_for_same_seed(self):
        a = PINGCircuit(seed=99)
        b = PINGCircuit(seed=99)
        for _ in range(500):
            sa_e, sa_i = a.step(dt=0.1)
            sb_e, sb_i = b.step(dt=0.1)
            np.testing.assert_array_equal(sa_e, sb_e)
            np.testing.assert_array_equal(sa_i, sb_i)

    def test_global_numpy_seed_does_not_leak_in(self):
        np.random.seed(0)
        a = PINGCircuit(seed=42)
        a_spikes_e, a_spikes_i = [], []
        for _ in range(100):
            se, si = a.step(dt=0.1)
            a_spikes_e.append(se.copy())
            a_spikes_i.append(si.copy())

        np.random.seed(99999)
        b = PINGCircuit(seed=42)
        for t in range(100):
            sb_e, sb_i = b.step(dt=0.1)
            np.testing.assert_array_equal(a_spikes_e[t], sb_e)
            np.testing.assert_array_equal(a_spikes_i[t], sb_i)

    def test_total_spike_count_constant_across_runs(self):
        counts = []
        for _ in range(5):
            ping = PINGCircuit(seed=42)
            total = 0
            for _ in range(500):
                se, si = ping.step(dt=0.1)
                total += int(np.count_nonzero(se)) + int(np.count_nonzero(si))
            counts.append(total)
        assert len(set(counts)) == 1, f"non-deterministic: spike totals = {counts}"

    def test_reset_state_uses_per_instance_rng(self):
        ping = PINGCircuit(seed=42)
        for _ in range(10):
            ping.step(dt=0.1)
        np.random.seed(9999)
        ping.reset_state()
        v_e_after = ping.v_e.copy()

        ping2 = PINGCircuit(seed=42)
        for _ in range(10):
            ping2.step(dt=0.1)
        np.random.seed(1)
        ping2.reset_state()
        np.testing.assert_array_equal(v_e_after, ping2.v_e)
