# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMISI from former test_model_spike_response.py

"""Focused suite: TestSRMISI from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403


class TestSRMISI:
    def test_constant_isi(self):
        """At constant suprathreshold input, ISI is perfectly constant."""
        n = SpikeResponseNeuron()
        spikes = _run(n, current=10.0, steps=10000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[2:])
        assert np.all(isis == isis[0]), f"Non-constant ISI: {np.unique(isis)}"

    def test_isi_from_simulation(self):
        """Measure ISI at I=10.0: should be 20 steps (from probing)."""
        n = SpikeResponseNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        measured_isi = int(np.median(np.diff(spikes[2:])))
        # ISI = 20 from probing (refractory recovery takes 19 steps + spike step)
        assert 18 <= measured_isi <= 22, f"ISI = {measured_isi}"

    def test_isi_shortens_with_stronger_input(self):
        n_weak = SpikeResponseNeuron()
        n_strong = SpikeResponseNeuron()
        s_weak = _run(n_weak, current=8.0, steps=5000)
        s_strong = _run(n_strong, current=15.0, steps=5000)
        if len(s_weak) > 5 and len(s_strong) > 5:
            isi_weak = np.median(np.diff(s_weak[2:]))
            isi_strong = np.median(np.diff(s_strong[2:]))
            assert isi_strong < isi_weak

    def test_cv_isi_zero(self):
        n = SpikeResponseNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        isis_arr = np.diff(spikes[2:]).astype(float)
        cv = np.std(isis_arr) / np.mean(isis_arr) if len(isis_arr) > 5 else 0
        assert cv < 0.01
