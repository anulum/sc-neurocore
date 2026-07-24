# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNakerLIFRefractory from former test_model_spinnaker_lif.py

"""Focused suite: TestSpiNNakerLIFRefractory from former test_model_spinnaker_lif.py."""

from __future__ import annotations

from tests.model_spinnaker_lif_support import *  # noqa: F403


class TestSpiNNakerLIFRefractory:
    def test_refractory_blocks_spikes(self):
        """During refractory period (tau_refrac=2), no spikes can occur."""
        n = SpiNNakerLIFNeuron()
        # Drive to spike
        for _ in range(100):
            if n.step(50.0) == 1:
                # Immediately after spike: refrac_count = 2
                assert n.refrac_count == n.tau_refrac
                # Next 2 steps should be blocked
                s1 = n.step(50.0)
                s2 = n.step(50.0)
                assert s1 == 0 and s2 == 0, "Should be refractory"
                return
        raise AssertionError("No spike")

    def test_refractory_reduces_rate(self):
        """Refractory period limits maximum firing rate."""
        n_norefrac = SpiNNakerLIFNeuron(tau_refrac=0.0)
        n_refrac = SpiNNakerLIFNeuron(tau_refrac=5.0)
        s_no = len(_run(n_norefrac, current=50.0, steps=5000))
        s_yes = len(_run(n_refrac, current=50.0, steps=5000))
        assert s_no > s_yes

    def test_refrac_count_decrements(self):
        n = SpiNNakerLIFNeuron(tau_refrac=3.0)
        n.refrac_count = 3.0
        n.step(0.0)
        assert n.refrac_count == 2.0
        n.step(0.0)
        assert n.refrac_count == 1.0

    def test_i_offset_adds_baseline(self):
        """i_offset provides constant baseline current."""
        n = SpiNNakerLIFNeuron(i_offset=25.0)
        spikes = len(_run(n, current=0.0, steps=5000))
        assert spikes > 0, "i_offset should drive spikes even at I=0"
