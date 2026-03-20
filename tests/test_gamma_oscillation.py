# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
