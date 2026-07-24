# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExAdaptation from former test_model_adex.py

"""Focused suite: TestAdExAdaptation from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403


class TestAdExAdaptation:
    def test_w_increments_on_spike(self):
        """Each spike adds b to w."""
        n = AdExNeuron()
        w_before = n.w
        for _ in range(10000):
            if n.step(500.0) == 1:
                # w should have been incremented by b
                assert n.w > w_before
                break
        else:
            pytest.fail("No spike")

    def test_isi_lengthens(self):
        """Adaptation: early ISIs shorter than late ISIs."""
        n = AdExNeuron()
        spikes = _run(n, current=500.0, steps=10000)
        assert len(spikes) >= 10
        isis_arr = np.diff(spikes)
        early = np.mean(isis_arr[:3])
        late = np.mean(isis_arr[-3:])
        assert late > early, f"early={early:.0f}, late={late:.0f}"

    def test_w_decays_between_spikes(self):
        """w decays toward a·(V-V_rest) between spikes (tau_w timescale)."""
        n = AdExNeuron()
        n.w = 50.0
        # Subthreshold: w should decay
        for _ in range(1000):
            n.step(0.0)
        assert n.w < 50.0

    def test_no_adaptation_when_b_zero(self):
        """b=0: no w increment on spike → constant ISI (like EIF)."""
        n = AdExNeuron(b=0.0, a=0.0)
        spikes = _run(n, current=500.0, steps=10000)
        if len(spikes) >= 10:
            isis_arr = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis_arr) / np.mean(isis_arr)
            assert cv < 0.05, f"CV(ISI) = {cv:.4f} without adaptation"

    def test_stronger_adaptation_fewer_spikes(self):
        """Higher b → stronger per-spike w increment → fewer spikes."""
        n_weak = AdExNeuron(b=2.0)
        n_strong = AdExNeuron(b=20.0)
        s_weak = len(_run(n_weak, current=500.0, steps=10000))
        s_strong = len(_run(n_strong, current=500.0, steps=10000))
        assert s_weak > s_strong
