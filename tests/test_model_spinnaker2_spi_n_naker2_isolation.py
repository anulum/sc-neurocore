# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2Isolation from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2Isolation from former test_model_spinnaker2.py."""

from __future__ import annotations

from tests.model_spinnaker2_support import *  # noqa: F403

class TestSpiNNaker2Isolation:
    def test_defaults(self):
        n = SpiNNaker2Neuron()
        assert n.v == 0 and n.v_threshold == 1024
        assert n.decay_mult == 243 and n.decay_shift == 8
        assert n.refrac_steps == 2

    def test_step_returns_binary(self):
        assert SpiNNaker2Neuron().step(0) in (0, 1)

    def test_integer_types(self):
        n = SpiNNaker2Neuron()
        assert isinstance(n.v, int)
        assert isinstance(n.v_threshold, int)

    def test_state_finite(self):
        n = SpiNNaker2Neuron()
        for _ in range(50000):
            n.step(500)
        # Integer can't be NaN, but check it's reasonable
        assert abs(n.v) < 10**9

    def test_reset(self):
        n = SpiNNaker2Neuron()
        for _ in range(100):
            n.step(500)
        n.reset()
        assert n.v == n.v_rest and n._refrac_count == 0
