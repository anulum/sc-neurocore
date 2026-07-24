# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAkidaIsolation from former test_model_akida_neuron.py

"""Focused suite: TestAkidaIsolation from former test_model_akida_neuron.py."""

from __future__ import annotations

from tests.model_akida_neuron_support import *  # noqa: F403


class TestAkidaIsolation:
    def test_defaults(self):
        n = AkidaNeuron()
        assert n.v == 0 and n.threshold == 100
        assert n.modulation == 0.75
        assert n._rank == 0 and n._spiked is False

    def test_step_returns_binary(self):
        assert AkidaNeuron().step(0) in (0, 1)

    def test_integer_voltage(self):
        """V is integer — neuromorphic hardware constraint."""
        n = AkidaNeuron()
        n.step(50)
        assert isinstance(n.v, int)

    def test_reset_restores_defaults(self):
        n = AkidaNeuron()
        for _ in range(10):
            n.step(50)
        n.reset()
        assert n.v == 0 and n._rank == 0 and n._spiked is False

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AkidaNeuron()
            trace = [(n.step(50), n.v) for _ in range(20)]
            traces.append(trace)
        assert traces[0] == traces[1]
