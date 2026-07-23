# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2FixedPointDecay from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2FixedPointDecay from former test_model_spinnaker2.py."""

from __future__ import annotations

from tests.model_spinnaker2_support import *  # noqa: F403

class TestSpiNNaker2FixedPointDecay:
    """Core: v = ((v - v_rest) * decay_mult >> decay_shift) + v_rest + I.

    This is fixed-point exponential decay: alpha = decay_mult / 2^decay_shift
    = 243/256 ≈ 0.949, approximating exp(-1/10) ≈ 0.905 with slight error.
    """

    def test_decay_formula(self):
        """One step with I=0: v_new = (v * 243 >> 8) + 0."""
        n = SpiNNaker2Neuron()
        n.v = 1000
        n.step(0)
        expected = (1000 * 243 >> 8) + 0  # = 949
        assert n.v == expected, f"v={n.v}, expected={expected}"

    def test_decay_reduces_voltage(self):
        """Decay with zero input should reduce |v| toward v_rest=0."""
        n = SpiNNaker2Neuron()
        n.v = 500
        n.step(0)
        assert n.v < 500

    def test_effective_alpha(self):
        """alpha_eff = 243/256 ≈ 0.9492."""
        alpha = 243 / 256
        assert 0.94 < alpha < 0.96

    def test_integer_arithmetic_only(self):
        """Verify no float operations: v stays integer."""
        n = SpiNNaker2Neuron()
        for _ in range(100):
            n.step(100)
            assert isinstance(n.v, int), f"v is {type(n.v)}"
