# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHodgkinHuxleyValidation from former test_hh_validation.py

"""Focused suite: TestHodgkinHuxleyValidation from former test_hh_validation.py."""

from __future__ import annotations

from tests.hh_validation_support import *  # noqa: F403


class TestHodgkinHuxleyValidation:
    """Validate HH model against 1952 paper expectations."""

    def test_resting_potential(self):
        """No current → resting potential near -65 mV."""
        V, _, _, _ = simulate_hh(0.0, duration_ms=50)
        assert -70 < V[-1] < -60, f"V_rest={V[-1]:.2f} (expect -65±5)"

    def test_ap_peak_positive(self):
        """I=10 → action potential peak above 0 mV."""
        V, _, _, _ = simulate_hh(10.0, duration_ms=50)
        assert np.max(V) > 0

    def test_ap_peak_in_range(self):
        """AP peak between 20 and 60 mV (HH 1952 figure 12)."""
        V, _, _, _ = simulate_hh(10.0, duration_ms=50)
        v_peak = float(np.max(V))
        assert 20 < v_peak < 60, f"V_peak={v_peak:.1f}"

    def test_spike_width(self):
        """Half-max width between 0.5 and 3.0 ms."""
        V, _, _, _ = simulate_hh(10.0, duration_ms=50)
        v_rest = -65.0
        threshold = (float(np.max(V)) + v_rest) / 2
        above = threshold < V
        tr = np.diff(above.astype(int))
        up = np.where(tr == 1)[0]
        dn = np.where(tr == -1)[0]
        assert len(up) > 0 and len(dn) > 0
        fd = dn[dn > up[0]]
        assert len(fd) > 0
        width = float((fd[0] - up[0]) * 0.01)
        assert 0.5 < width < 3.0, f"width={width:.2f}ms"

    def test_m_gate_activates(self):
        """m gate > 0.8 at peak (Na activation)."""
        V, M, _, _ = simulate_hh(10.0, duration_ms=50)
        assert M[np.argmax(V)] > 0.8

    def test_h_gate_inactivates(self):
        """h gate < 0.4 at peak (Na inactivation).

        Euler integration causes slight lag vs continuous model (~0.2).
        """
        V, _, H, _ = simulate_hh(10.0, duration_ms=50)
        assert H[np.argmax(V)] < 0.4

    def test_n_gate_rises(self):
        """n gate at peak > n at rest (K activation)."""
        V, _, _, N = simulate_hh(10.0, duration_ms=50)
        assert N[np.argmax(V)] > N[0]

    def test_afterhyperpolarisation(self):
        """AHP below -65 mV within 10ms after peak."""
        V, _, _, _ = simulate_hh(10.0, duration_ms=50)
        pi = int(np.argmax(V))
        post = V[pi : pi + int(10 / 0.01)]
        assert len(post) > 100
        assert float(np.min(post)) < -65

    def test_fi_monotonic(self):
        """More current → more spikes."""
        rates = []
        for I in [7.0, 10.0, 15.0, 20.0]:
            V, _, _, _ = simulate_hh(I, duration_ms=100)
            rates.append(int(np.sum(np.diff((V > 0).astype(int)) == 1)))
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1], f"rates={rates}"

    def test_subthreshold(self):
        """I=1 (well below rheobase ~6.3) → no spikes."""
        V, _, _, _ = simulate_hh(1.0, duration_ms=100)
        n_spikes = int(np.sum(np.diff((V > 0).astype(int)) == 1))
        assert n_spikes == 0, f"spikes={n_spikes} at I=1"
