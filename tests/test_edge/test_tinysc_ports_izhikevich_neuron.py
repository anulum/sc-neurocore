# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIzhikevichNeuron from former test_tinysc_ports.py

"""Focused suite: TestIzhikevichNeuron from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403

class TestIzhikevichNeuron:
    def test_regular_spiking(self):
        n = IzhikevichNeuron.regular_spiking()
        assert n.a_q16 == 1311

    def test_fast_spiking(self):
        n = IzhikevichNeuron.fast_spiking()
        assert n.a_q16 == 6554

    def test_modes_differ(self):
        rs = IzhikevichNeuron.regular_spiking()
        fs = IzhikevichNeuron.fast_spiking()
        assert rs.a_q16 != fs.a_q16

    def test_tick_without_strong_input_does_not_immediately_spike(self):
        n = IzhikevichNeuron.regular_spiking()
        spiked = n.tick(0)
        assert spiked is False
        assert n.spike_count == 0
