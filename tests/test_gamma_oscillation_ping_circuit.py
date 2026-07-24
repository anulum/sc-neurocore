# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPINGCircuit from former test_gamma_oscillation.py

"""Focused suite: TestPINGCircuit from former test_gamma_oscillation.py."""

from __future__ import annotations

from tests.gamma_oscillation_support import *  # noqa: F403


class TestPINGCircuit:
    def test_creates_default(self):
        ping = PINGCircuit()
        assert ping.n_excitatory == 80
        assert ping.n_inhibitory == 20
        assert ping.v_e.shape == (80,)
        assert ping.v_i.shape == (20,)
        # Default initial V is near E_L (-67 mV ± 2 mV jitter).
        assert np.all(ping.v_e >= ping.e_l - 2.5)
        assert np.all(ping.v_e <= ping.e_l + 2.5)

    def test_produces_spikes(self):
        ping = PINGCircuit()  # default drive 1.4 µA/cm² → supra-threshold
        total_e, total_i = 0, 0
        for _ in range(2000):  # 200 ms at dt=0.1
            se, si = ping.step(dt=0.1)
            total_e += int(np.count_nonzero(se))
            total_i += int(np.count_nonzero(si))
        assert total_e > 0
        assert total_i > 0  # E→I gain loop must engage

    def test_no_drive_no_spikes(self):
        ping = PINGCircuit(
            i_drive_e_mean=0.0,
            i_drive_e_sigma=0.0,
            i_drive_i_mean=0.0,
            i_drive_i_sigma=0.0,
            sigma_e=0.0,
            sigma_i=0.0,
        )
        total = 0
        for _ in range(1000):
            se, si = ping.step(dt=0.1)
            total += int(np.count_nonzero(se)) + int(np.count_nonzero(si))
        # Zero drive + zero noise → V relaxes to E_L < threshold → no spikes.
        assert total == 0

    def test_inhibition_suppresses(self):
        # Stronger I→E inhibition should suppress E firing within the
        # published Börgers-Kopell weak-PING regime. Outside this band
        # (w_ie ≫ 0.05) the conductance saturates and rebound bursts
        # dominate, so the assertion is restricted to the realistic span.
        ping_strong = PINGCircuit(w_ie=0.05, seed=7)
        ping_weak = PINGCircuit(w_ie=0.001, seed=7)
        e_strong, e_weak = 0, 0
        for _ in range(1500):  # 150 ms
            se, _ = ping_strong.step(dt=0.1)
            e_strong += int(np.count_nonzero(se))
            se2, _ = ping_weak.step(dt=0.1)
            e_weak += int(np.count_nonzero(se2))
        assert e_strong < e_weak, (
            f"stronger inhibition should suppress (e_strong={e_strong}, e_weak={e_weak})"
        )

    def test_reset_returns_v_near_e_l(self):
        ping = PINGCircuit()
        for _ in range(100):
            ping.step(dt=0.1)
        ping.reset_state()
        assert np.all(ping.v_e >= ping.e_l - 2.5)
        assert np.all(ping.v_e <= ping.e_l + 2.5)
        assert np.all(ping.g_ampa_e == 0.0)
        assert np.all(ping.g_gaba_e == 0.0)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="at least 1 E and 1 I"):
            PINGCircuit(n_excitatory=0)
